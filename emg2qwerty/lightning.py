# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Sequence
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchmetrics import MetricCollection
import torch.nn.functional as F


from emg2qwerty import utils
from emg2qwerty.charset import charset
from emg2qwerty.data import LabelData, WindowedEMGDataset
from emg2qwerty.metrics import CharacterErrorRates
from emg2qwerty.modules import (
    MultiBandRotationInvariantMLP,
    SpectrogramNorm,
    TDSConvEncoder,
    TDSLSTMEncoder,
    #TDSRNNEncoder,
    #TDSGRUEncoder,
    #TransformerEncoderBlock,
)
from emg2qwerty.transforms import Transform


class WindowedEMGDataModule(pl.LightningDataModule):
    def __init__(
        self,
        window_length: int,
        padding: tuple[int, int],
        batch_size: int,
        num_workers: int,
        train_sessions: Sequence[Path],
        val_sessions: Sequence[Path],
        test_sessions: Sequence[Path],
        train_transform: Transform[np.ndarray, torch.Tensor],
        val_transform: Transform[np.ndarray, torch.Tensor],
        test_transform: Transform[np.ndarray, torch.Tensor],
    ) -> None:
        super().__init__()

        self.window_length = window_length
        self.padding = padding

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_sessions = train_sessions
        self.val_sessions = val_sessions
        self.test_sessions = test_sessions

        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: str | None = None) -> None:
        self.train_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.train_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=True,
                )
                for hdf5_path in self.train_sessions
            ]
        )
        self.val_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.val_transform,
                    window_length=self.window_length,
                    padding=self.padding,
                    jitter=False,
                )
                for hdf5_path in self.val_sessions
            ]
        )
        self.test_dataset = ConcatDataset(
            [
                WindowedEMGDataset(
                    hdf5_path,
                    transform=self.test_transform,
                    # Feed the entire session at once without windowing/padding
                    # at test time for more realism
                    window_length=None,
                    padding=(0, 0),
                    jitter=False,
                )
                for hdf5_path in self.test_sessions
            ]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self) -> DataLoader:
        # Test dataset does not involve windowing and entire sessions are
        # fed at once. Limit batch size to 1 to fit within GPU memory and
        # avoid any influence of padding (while collating multiple batch items)
        # in test scores.
        return DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=WindowedEMGDataset.collate,
            pin_memory=True,
            persistent_workers=True,
        )


#####################################################################################################################
# When trying different model architectures, the relevant encoder parts are commented out and new ones are added    #
# since the overall structure of the project is modular, implementing them this way was more convenient             #
# rather than defining each model as a separate class and using different yaml files. (except the VAE code).        #
#####################################################################################################################

class TDSConvCTCModule(pl.LightningModule): # 
    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,
        mlp_features: Sequence[int],
        block_channels: Sequence[int],
        kernel_width: int,
        optimizer: DictConfig,
        lr_scheduler: DictConfig,
        decoder: DictConfig,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1]

        # Model
        # inputs: (T, N, bands=2, electrode_channels=16, freq)
        self.model = nn.Sequential(
            # (T, N, bands=2, C=16, freq)
            SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS),
            # (T, N, bands=2, mlp_features[-1])
            MultiBandRotationInvariantMLP(
                in_features=in_features,
                mlp_features=mlp_features,
                num_bands=self.NUM_BANDS,
            ),
            # (T, N, num_features)
            nn.Flatten(start_dim=2),

            TDSConvEncoder(
                num_features=num_features,
                block_channels=block_channels,
                kernel_width=kernel_width,
            ),
            TDSLSTMEncoder(
                num_features = num_features,
                lstm_hidden_size = 196,
                num_lstm_layers = 3
            ),
            #TDSGRUEncoder(
            #    num_features=num_features,
            #    gru_hidden_size=196,
            #    num_gru_layers=4,
            #    bidirectional=False
            #),
            #TDSRNNEncoder(
            #    num_features = num_features,
            #    rnn_hidden_size = 128,
            #    num_rnn_layers = 4
            #),
            # (T, N, num_classes)
            nn.Linear(num_features, charset().num_classes),
            nn.LogSoftmax(dim=-1),
        )

        # Criterion
        self.ctc_loss = nn.CTCLoss(blank=charset().null_class)

        # Decoder
        self.decoder = instantiate(decoder)

        # Metrics
        metrics = MetricCollection([CharacterErrorRates()])
        self.metrics = nn.ModuleDict(
            {
                f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
                for phase in ["train", "val", "test"]
            }
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs)

    def _step(
        self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
    ) -> torch.Tensor:
        inputs = batch["inputs"]
        targets = batch["targets"]
        input_lengths = batch["input_lengths"]
        target_lengths = batch["target_lengths"]
        N = len(input_lengths)  # batch_size

        emissions = self.forward(inputs)

        # Shrink input lengths by an amount equivalent to the conv encoder's
        # temporal receptive field to compute output activation lengths for CTCLoss.
        # NOTE: This assumes the encoder doesn't perform any temporal downsampling
        # such as by striding.
        T_diff = inputs.shape[0] - emissions.shape[0]
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss(
            log_probs=emissions,  # (T, N, num_classes)
            targets=targets.transpose(0, 1),  # (T, N) -> (N, T)
            input_lengths=emission_lengths,  # (N,)
            target_lengths=target_lengths,  # (N,)
        )

        # Decode emissions
        predictions = self.decoder.decode_batch(
            emissions=emissions.detach().cpu().numpy(),
            emission_lengths=emission_lengths.detach().cpu().numpy(),
        )

        # Update metrics
        metrics = self.metrics[f"{phase}_metrics"]
        targets = targets.detach().cpu().numpy()
        target_lengths = target_lengths.detach().cpu().numpy()
        for i in range(N):
            # Unpad targets (T, N) for batch entry
            target = LabelData.from_labels(targets[: target_lengths[i], i])
            metrics.update(prediction=predictions[i], target=target)

        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def training_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("train", *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("val", *args, **kwargs)

    def test_step(self, *args, **kwargs) -> torch.Tensor:
        return self._step("test", *args, **kwargs)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def configure_optimizers(self) -> dict[str, Any]:
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.hparams.optimizer,
            lr_scheduler_config=self.hparams.lr_scheduler,
        )

# # tried implementing a transformer into my model

# class LearnablePositionalEncoding(nn.Module):
#     def __init__(self, d_model: int, max_len: int = 2000):
#         super().__init__()
#         self.pos_embedding = nn.Embedding(max_len, d_model)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         T = x.size(0)
#         positions = torch.arange(0, T, device=x.device).unsqueeze(1)  # (T, 1)
#         return x + self.pos_embedding(positions).expand(-1, x.size(1), -1)


# class TransformerEncoderBlock(nn.Module):
#     def __init__(self, d_model=256, nhead=4, dim_feedforward=1024, dropout=0.1, num_layers=2):
#         super().__init__()
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=False,  # We'll keep (T, N, d_model)
#         )
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

#         self.pos_encoding = LearnablePositionalEncoding(d_model)

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = self.pos_encoding(x)  # Add positional info
#         x = self.transformer_encoder(x)  # (T, N, d_model)
#         return x


# class TDSLSTMTransformerModule(pl.LightningModule):
#     NUM_BANDS: ClassVar[int] = 2
#     ELECTRODE_CHANNELS: ClassVar[int] = 16

#     def __init__(
#         self,
#         in_features: int,
#         mlp_features: Sequence[int],
#         block_channels: Sequence[int],
#         kernel_width: int,
#         lstm_hidden_size: int,
#         lstm_layers: int,
#         transformer_d_model: int,
#         transformer_nhead: int,
#         transformer_ff: int,
#         transformer_layers: int,
#         optimizer: DictConfig,
#         lr_scheduler: DictConfig,
#         decoder: DictConfig,
#     ) -> None:
#         super().__init__()
#         self.save_hyperparameters()

#         num_features = self.NUM_BANDS * mlp_features[-1]

#         # spectrogram + MLP 
#         self.spec_norm = SpectrogramNorm(channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS)
#         self.mlp = MultiBandRotationInvariantMLP(
#             in_features=in_features,
#             mlp_features=mlp_features,
#             num_bands=self.NUM_BANDS,
#         )
#         self.flatten = nn.Flatten(start_dim=2)
#         self.tds = TDSConvEncoder( # TDS Conv Encoder
#             num_features=num_features,
#             block_channels=block_channels,
#             kernel_width=kernel_width,
#         )
#         self.lstm = nn.LSTM( # LSTM
#             input_size=num_features,
#             hidden_size=lstm_hidden_size,
#             num_layers=lstm_layers,
#             batch_first=False,
#             bidirectional=False
#         )
#         self.transformer = TransformerEncoderBlock( # Transformer
#             d_model=lstm_hidden_size,
#             nhead=transformer_nhead,
#             dim_feedforward=transformer_ff,
#             num_layers=transformer_layers
#         )
#         self.fc = nn.Linear(lstm_hidden_size, charset().num_classes) # Linear Layer
#         self.log_softmax = nn.LogSoftmax(dim=-1)
#
#         self.ctc_loss = nn.CTCLoss(blank=charset().null_class) # Same layer
#         self.decoder = instantiate(decoder)
#         metrics = MetricCollection([CharacterErrorRates()])
#         self.metrics = nn.ModuleDict({
#             f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
#             for phase in ["train", "val", "test"]
#         })

#     def forward(self, inputs: torch.Tensor) -> torch.Tensor:
#
#         x = self.spec_norm(inputs)   # (T, N, 2, 16, freq)
#         x = self.mlp(x)              # (T, N, 2, mlp_features[-1])
#         x = self.flatten(x)          # (T, N, num_features)
#
#         x = self.tds(x)             # (T, N, num_features)
#         x, _ = self.lstm(x)         # (T, N, lstm_hidden_size)
#
#         # Transformer expects (T, N, d_model) => d_model = lstm_hidden_size
#         x = self.transformer(x)     # (T, N, lstm_hidden_size)
#
#         x = self.fc(x)              # (T, N, num_classes)
#         return self.log_softmax(x)  # (T, N, num_classes)
#    
#     def _step(
#         self, phase: str, batch: dict[str, torch.Tensor], *args, **kwargs
#     ) -> torch.Tensor:
#         inputs = batch["inputs"]
#         targets = batch["targets"]
#         input_lengths = batch["input_lengths"]
#         target_lengths = batch["target_lengths"]
#         N = len(input_lengths)  # batch_size
#
#         emissions = self.forward(inputs)
#
#         T_diff = inputs.shape[0] - emissions.shape[0]
#         emission_lengths = input_lengths - T_diff

#         loss = self.ctc_loss(
#             log_probs=emissions,  # (T, N, num_classes)
#             targets=targets.transpose(0, 1),
#             input_lengths=emission_lengths,
#             target_lengths=target_lengths,
#         )

#         predictions = self.decoder.decode_batch( #d decode
#             emissions=emissions.detach().cpu().numpy(),
#             emission_lengths=emission_lengths.detach().cpu().numpy(),
#         )

#         metrics = self.metrics[f"{phase}_metrics"] # save metrics
#         targets = targets.detach().cpu().numpy()
#         target_lengths = target_lengths.detach().cpu().numpy()
#         for i in range(N):
#             target = LabelData.from_labels(targets[: target_lengths[i], i])
#             metrics.update(prediction=predictions[i], target=target)

#         self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
#         return loss

#     def _epoch_end(self, phase: str) -> None:
#         metrics = self.metrics[f"{phase}_metrics"]
#         self.log_dict(metrics.compute(), sync_dist=True)
#         metrics.reset()

#     def training_step(self, *args, **kwargs) -> torch.Tensor:
#         return self._step("train", *args, **kwargs)

#     def validation_step(self, *args, **kwargs) -> torch.Tensor:
#         return self._step("val", *args, **kwargs)

#     def test_step(self, *args, **kwargs) -> torch.Tensor:
#         return self._step("test", *args, **kwargs)

#     def on_train_epoch_end(self) -> None:
#         self._epoch_end("train")

#     def on_validation_epoch_end(self) -> None:
#         self._epoch_end("val")

#     def on_test_epoch_end(self) -> None:
#         self._epoch_end("test")

#     def configure_optimizers(self) -> dict[str, Any]:
#         return utils.instantiate_optimizer_and_scheduler(
#             self.parameters(),
#             optimizer_config=self.hparams.optimizer,
#             lr_scheduler_config=self.hparams.lr_scheduler,
#         )
#     # training_step, validation_step, test_step, etc. remain identical to your existing code
#     # configure_optimizers remains the same, etc.

# Defined a new Module for TDS + RNN Structure

class TDSRNNCTCModule(pl.LightningModule):
    """
    A variant of TDSConvCTCModule that:
      1) Normalizes spectrogram (SpectrogramNorm)
      2) Applies MultiBandRotationInvariantMLP
      3) Flattens the result
      4) Feeds into TDSConvEncoder
      5) Then runs a single RNN (GRU or LSTM)
      6) Then a Linear -> LogSoftmax
      7) Uses CTCLoss + decoder exactly like TDSConvCTCModule
    """

    NUM_BANDS: ClassVar[int] = 2
    ELECTRODE_CHANNELS: ClassVar[int] = 16

    def __init__(
        self,
        in_features: int,            # this is by default 528 in config file
        mlp_features: Sequence[int], # this is by default 384 in config file
        block_channels: Sequence[int],
        kernel_width: int,
        rnn_hidden_size: int = 128,  # we need to adjust this based on our model config
        optimizer: DictConfig | None = None,
        lr_scheduler: DictConfig | None = None,
        decoder: DictConfig | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        num_features = self.NUM_BANDS * mlp_features[-1] # output dim after flattening MLP

        self.spectrogram_norm = SpectrogramNorm( # spectrogram + MLP
            channels=self.NUM_BANDS * self.ELECTRODE_CHANNELS
        )
        self.mlp = MultiBandRotationInvariantMLP(
            in_features=in_features,
            mlp_features=mlp_features,
            num_bands=self.NUM_BANDS,
        )
        self.flatten = nn.Flatten(start_dim=2)  # flattening again

        self.tds_encoder = TDSConvEncoder( # TDS block that outputs shape [T, N, num_features]
            num_features=num_features,
            block_channels=block_channels,
            kernel_width=kernel_width,
        )


        self.rnn = nn.GRU( # changed this with GRU and LSTM for different architectures
            input_size=num_features,
            hidden_size=rnn_hidden_size,
            batch_first=False  # output shape is [T, N, Feat]
        )

        self.fc = nn.Linear(rnn_hidden_size, charset().num_classes)
        self.log_softmax = nn.LogSoftmax(dim=-1)

        self.ctc_loss = nn.CTCLoss(blank=charset().null_class) # ctc loss and decoder
        self.decoder = instantiate(decoder) if decoder else None

        metrics = MetricCollection([CharacterErrorRates()]) # load the metrics
        self.metrics = nn.ModuleDict({
            f"{phase}_metrics": metrics.clone(prefix=f"{phase}/")
            for phase in ["train", "val", "test"]
        })

        self.opt_conf = optimizer
        self.lr_scheduler_conf = lr_scheduler

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        inputs shape: (T, N, bands=2, electrode_channels=16, freq)
        returns: (T, N, num_classes), after passing TDS + RNN + FC
        """

        x = self.spectrogram_norm(inputs)   # (T, N, 2, 16, freq) spectrogram Norm + MLP
        x = self.mlp(x)                     # (T, N, 2, mlp_features[-1])
        x = self.flatten(x)                 # (T, N, num_features)

        x = self.tds_encoder(x)             # TDS (T, N, num_features)

        rnn_out, _ = self.rnn(x)            # RNN (T, N, rnn_hidden_size)

        logits = self.fc(rnn_out)           # FC + log_softmax (T, N, num_classes)
        return self.log_softmax(logits)     # log-probs (T, N, num_classes)

    def _step(self, phase: str, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Common code for train/val/test step, mirroring TDSConvCTCModule.
        """
        inputs = batch["inputs"]          # shape: (T, N, 2, 16, freq)
        targets = batch["targets"]        # shape: (T, N)
        input_lengths = batch["input_lengths"]  # (N,)
        target_lengths = batch["target_lengths"] # (N,)

        emissions = self(inputs)  # forward pass (T, N, num_classes)
        T_in = inputs.shape[0]
        T_out = emissions.shape[0]

        T_diff = T_in - T_out
        emission_lengths = input_lengths - T_diff

        loss = self.ctc_loss( # CTC loss
            emissions,                      # (T, N, num_classes)
            targets.transpose(0, 1),        # (N, T)
            emission_lengths,               # (N,)
            target_lengths,                 # (N,)
        )

        if self.decoder is not None: # decoding
            predictions = self.decoder.decode_batch(
                emissions=emissions.detach().cpu().numpy(),
                emission_lengths=emission_lengths.detach().cpu().numpy(),
            )
            self._update_metrics(phase, predictions, targets, target_lengths)
        else:
            pass

        N = len(input_lengths)  # batch_size
        self.log(f"{phase}/loss", loss, batch_size=N, sync_dist=True)
        return loss

    def _update_metrics(
        self,
        phase: str,
        predictions: list[Any],
        targets: torch.Tensor,
        target_lengths: torch.Tensor
    ) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        targets_np = targets.detach().cpu().numpy()
        target_lens_np = target_lengths.detach().cpu().numpy()

        for i, pred in enumerate(predictions):
            T_len = target_lens_np[i]
            true_labels = targets_np[:T_len, i]
            metrics.update(prediction=pred, target=true_labels)

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step("train", batch)

    def validation_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step("val", batch)

    def test_step(self, batch, batch_idx) -> torch.Tensor:
        return self._step("test", batch)

    def on_train_epoch_end(self) -> None:
        self._epoch_end("train")

    def on_validation_epoch_end(self) -> None:
        self._epoch_end("val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end("test")

    def _epoch_end(self, phase: str) -> None:
        metrics = self.metrics[f"{phase}_metrics"]
        self.log_dict(metrics.compute(), sync_dist=True)
        metrics.reset()

    def configure_optimizers(self):
        return utils.instantiate_optimizer_and_scheduler(
            self.parameters(),
            optimizer_config=self.opt_conf,
            lr_scheduler_config=self.lr_scheduler_conf,
        )
