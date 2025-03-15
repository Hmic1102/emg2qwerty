# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeVar

import numpy as np
import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
# from .train_vae import VAEModel


TTransformIn = TypeVar("TTransformIn")
TTransformOut = TypeVar("TTransformOut")
Transform = Callable[[TTransformIn], TTransformOut]

@dataclass
class ToTensor:
    """Extracts the specified ``fields`` from a numpy structured array
    and stacks them into a ``torch.Tensor``.

    Following TNC convention as a default, the returned tensor is of shape
    (time, field/batch, electrode_channel).

    Args:
        fields (list): List of field names to be extracted from the passed in
            structured numpy ndarray.
        stack_dim (int): The new dimension to insert while stacking
            ``fields``. (default: 1)
    """

    fields: Sequence[str] = ("emg_left", "emg_right")
    stack_dim: int = 1

    def __call__(self, data: np.ndarray) -> torch.Tensor:
        return torch.stack(
            [torch.as_tensor(data[f]) for f in self.fields], dim=self.stack_dim
        )

@dataclass
class RandomChannelDropoutN:
    """Randomly drops exactly `n` channels in the EMG data.

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies the same dropout mask across all items in the batch. To apply 
    different dropout masks per batch item, use the ``ForEach`` wrapper.

    Args:
        n (int): The number of channels to drop.
        fill_value (float): Value to fill the dropped channels with (default: 0.0).
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    n: int
    fill_value: float = 0.0
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        num_channels = tensor.shape[self.channel_dim]
        assert self.n <= num_channels, f"n ({self.n}) must be <= number of channels ({num_channels})"

        # Randomly select `n` unique channels to drop
        drop_indices = np.random.choice(num_channels, self.n, replace=False)

        # Create a mask initialized to 1 (keep all channels)
        mask = torch.ones(num_channels, device=tensor.device)

        # Set selected channels to 0 (drop)
        mask[drop_indices] = 0  

        # Reshape mask to match tensor dimensions
        mask_shape = [1 if i != self.channel_dim else num_channels for i in range(tensor.ndim)]
        mask = mask.view(mask_shape)

        # Apply mask (set dropped channels to fill_value)
        return tensor * mask + self.fill_value * (1 - mask)

@dataclass
class Downsample:
    """
    A transform that downsamples the time dimension by a given factor (stride).
    E.g., stride=2 => keep every other sample => effectively halves the sampling rate.
    """
    stride: int = 2  # e.g. stride=2 => half the sampling rate
    time_dim: int = 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # shape e.g. (T, N, 2, 16, freq)
        # We want to keep every 'stride' sample along the time_dim
        return tensor.index_select(
            self.time_dim,
            torch.arange(0, tensor.shape[self.time_dim], self.stride, device=tensor.device)
        )

@dataclass
class Lambda:
    """Applies a custom lambda function as a transform.

    Args:
        lambd (lambda): Lambda to wrap within.
    """

    lambd: Transform[Any, Any]

    def __call__(self, data: Any) -> Any:
        return self.lambd(data)


@dataclass
class ForEach:
    """Applies the provided ``transform`` over each item of a batch
    independently. By default, assumes the input is of shape (T, N, ...).

    Args:
        transform (Callable): The transform to apply to each batch item of
            the input tensor.
        batch_dim (int): The bach dimension, i.e., the dim along which to
            unstack/unbind the input tensor prior to mapping over
            ``transform`` and restacking. (default: 1)
    """

    transform: Transform[torch.Tensor, torch.Tensor]
    batch_dim: int = 1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [self.transform(t) for t in tensor.unbind(self.batch_dim)],
            dim=self.batch_dim,
        )


@dataclass
class Compose:
    """Compose a chain of transforms.

    Args:
        transforms (list): List of transforms to compose.
    """

    transforms: Sequence[Transform[Any, Any]]

    def __call__(self, data: Any) -> Any:
        for transform in self.transforms:
            data = transform(data)
        return data


@dataclass
class RandomBandRotation:
    """Applies band rotation augmentation by shifting the electrode channels
    by an offset value randomly chosen from ``offsets``. By default, assumes
    the input is of shape (..., C).

    NOTE: If the input is 3D with batch dim (TNC), then this transform
    applies band rotation for all items in the batch with the same offset.
    To apply different rotations each batch item, use the ``ForEach`` wrapper.

    Args:
        offsets (list): List of integers denoting the offsets by which the
            electrodes are allowed to be shift. A random offset from this
            list is chosen for each application of the transform.
        channel_dim (int): The electrode channel dimension. (default: -1)
    """

    offsets: Sequence[int] = (-1, 0, 1)
    channel_dim: int = -1

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        offset = np.random.choice(self.offsets) if len(self.offsets) > 0 else 0
        return tensor.roll(offset, dims=self.channel_dim)


@dataclass
class TemporalAlignmentJitter:
    """Applies a temporal jittering augmentation that randomly jitters the
    alignment of left and right EMG data by up to ``max_offset`` timesteps.
    The input must be of shape (T, ...).

    Args:
        max_offset (int): The maximum amount of alignment jittering in terms
            of number of timesteps.
        stack_dim (int): The dimension along which the left and right data
            are stacked. See ``ToTensor()``. (default: 1)
    """

    max_offset: int
    stack_dim: int = 1

    def __post_init__(self) -> None:
        assert self.max_offset >= 0

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.shape[self.stack_dim] == 2
        left, right = tensor.unbind(self.stack_dim)

        offset = np.random.randint(-self.max_offset, self.max_offset + 1)
        if offset > 0:
            left = left[offset:]
            right = right[:-offset]
        if offset < 0:
            left = left[:offset]
            right = right[-offset:]

        return torch.stack([left, right], dim=self.stack_dim)

# @dataclass
# class PretrainedVAETransform: # ADDED VAE 
#     """
#     Applies a pretrained VAE to the input tensor.
#     Expects the last dimension = input_dim (e.g. 16).
#     """

#     checkpoint_path: str
#     input_dim: int = 16
#     latent_dim: int = 16

#     def __post_init__(self): # load the same architecture used in train_vae.py
#         self.model = VAEModel(input_dim=self.input_dim, latent_dim=self.latent_dim)
        
#         ckpt = torch.load(self.checkpoint_path, map_location='cpu') # load checkpoint
        
#         if "state_dict" in ckpt:
#             self.model.load_state_dict(ckpt["state_dict"], strict=False)
#         else:
#             self.model.load_state_dict(ckpt, strict=False)

#         self.model.eval()
#         for p in self.model.parameters():
#             p.requires_grad = False

#     def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
#         with torch.no_grad():
#             original_shape = tensor.shape # flatten the shape like needed
#             x = tensor.view(-1, self.input_dim) # make last dim = initial dim
#             decoded, _, _ = self.model(x)
#             decoded = decoded.view(*original_shape)
#         return decoded

@dataclass
class LogSpectrogram:
    """Creates log10-scaled spectrogram from an EMG signal. In the case of
    multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)
    """

    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            # Disable centering of FFT windows to avoid padding inconsistencies
            # between train and test (due to differing window lengths), as well
            # as to be more faithful to real-time/streaming execution.
            center=False,
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.movedim(0, -1)  # (T, ..., C) -> (..., C, T)
        spec = self.spectrogram(x)  # (..., C, freq, T)
        logspec = torch.log10(spec + 1e-6)  # (..., C, freq, T)
        return logspec.movedim(-1, 0)  # (T, ..., C, freq)


@dataclass
class SpecAugment:
    """Applies time and frequency masking as per the paper
    "SpecAugment: A Simple Data Augmentation Method for Automatic Speech
    Recognition, Park et al" (https://arxiv.org/abs/1904.08779).

    Args:
        n_time_masks (int): Maximum number of time masks to apply,
            uniformly sampled from 0. (default: 0)
        time_mask_param (int): Maximum length of each time mask,
            uniformly sampled from 0. (default: 0)
        iid_time_masks (int): Whether to apply different time masks to
            each band/channel (default: True)
        n_freq_masks (int): Maximum number of frequency masks to apply,
            uniformly sampled from 0. (default: 0)
        freq_mask_param (int): Maximum length of each frequency mask,
            uniformly sampled from 0. (default: 0)
        iid_freq_masks (int): Whether to apply different frequency masks to
            each band/channel (default: True)
        mask_value (float): Value to assign to the masked columns (default: 0.)
    """

    n_time_masks: int = 0
    time_mask_param: int = 0
    iid_time_masks: bool = True
    n_freq_masks: int = 0
    freq_mask_param: int = 0
    iid_freq_masks: bool = True
    mask_value: float = 0.0

    def __post_init__(self) -> None:
        self.time_mask = torchaudio.transforms.TimeMasking(
            self.time_mask_param, iid_masks=self.iid_time_masks
        )
        self.freq_mask = torchaudio.transforms.FrequencyMasking(
            self.freq_mask_param, iid_masks=self.iid_freq_masks
        )

    def __call__(self, specgram: torch.Tensor) -> torch.Tensor:
        # (T, ..., C, freq) -> (..., C, freq, T)
        x = specgram.movedim(0, -1)

        # Time masks
        n_t_masks = np.random.randint(self.n_time_masks + 1)
        for _ in range(n_t_masks):
            x = self.time_mask(x, mask_value=self.mask_value)

        # Frequency masks
        n_f_masks = np.random.randint(self.n_freq_masks + 1)
        for _ in range(n_f_masks):
            x = self.freq_mask(x, mask_value=self.mask_value)

        # (..., C, freq, T) -> (T, ..., C, freq)
        return x.movedim(-1, 0)

@dataclass
class AmplitudeScaling:
    """Randomly scales the amplitude of the sEMG signals to simulate
    different muscle contraction strengths.
    
    Helps the model become robust to variations in typing pressure and
    muscle activation levels.

    Args:
        min_scale (float): Minimum scaling factor (default: 0.8).
        max_scale (float): Maximum scaling factor (default: 1.2).

    placed right after to_tensor
    """

    min_scale: float = 0.8
    max_scale: float = 1.2

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        scale = torch.FloatTensor(1).uniform_(self.min_scale, self.max_scale)
        return tensor * scale

from scipy.interpolate import CubicSpline
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class TimeWarping:
    """Applies time warping without changing the shape of the input.
    
    Ensures that the number of time steps remains unchanged after transformation.

    place right after to_tensor
    """
    warp_strength: float = 0.2

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        T = tensor.shape[0]  # Number of time steps
        time_steps = np.arange(T)  # Original time indices
        orig_time = time_steps / T  # Normalize to [0,1]

        # Generate random warping curve
        warp_factor = 1 + np.random.uniform(-self.warp_strength, self.warp_strength)
        warped_time = np.clip(orig_time * warp_factor, 0, 1)  # Apply warping and clip

        # Interpolate using cubic splines while ensuring same number of steps
        warped_tensor = []
        for i in range(tensor.shape[1]):  # Iterate over channels
            cs = CubicSpline(orig_time, tensor[:, i].numpy())  # Create cubic spline
            interpolated_values = cs(orig_time)  # Ensure same time steps
            warped_tensor.append(interpolated_values)

        return torch.tensor(np.stack(warped_tensor, axis=1), dtype=tensor.dtype)
    
def stretch(self, data, rate=1):
    '''
    Time stretch an audio series for a fixed rate using Librosa.
    Args:
        data: clean audio data.
        rate: time stretch rate.

    Returns:
        augmented_data: audio data time-stretched.
    '''
    input_length = len(data)
    # Speed of speech
    augmented_data = librosa.effects.time_stretch(y = data, rate = rate)
    if len(augmented_data) > input_length:
        # Cut the length of the augmented audio to be equal to the original.
        augmented_data = augmented_data[:input_length]
    else:
        # Pad with silence.
        augmented_data = np.pad(augmented_data, (0, max(0, input_length - len(augmented_data))), "constant")
    return augmented_data
    
from dataclasses import dataclass
import torch
import torchaudio
import math


@dataclass
class LogSpectrogramWithPhaseRandomization:
    """
    Creates a log10-scaled spectrogram from an EMG signal. In the case of
    a multi-channeled signal, the channels are treated independently.
    The input must be of shape (T, ...) and the returned spectrogram
    is of shape (T, ..., freq).

    Args:
        n_fft (int): Size of FFT, creates n_fft // 2 + 1 frequency bins.
            (default: 64)
        hop_length (int): Number of samples to stride between consecutive
            STFT windows. (default: 16)

    replace logspec
    """
    n_fft: int = 64
    hop_length: int = 16

    def __post_init__(self) -> None:
        # Set power=None to obtain the complex spectrogram for phase manipulation
        self.spectrogram = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            normalized=True,
            center=False,
            power=None  # Return complex-valued output
        )

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # Rearrange dimensions: (T, ..., C) -> (..., C, T)
        x = tensor.movedim(0, -1)
        # Compute complex spectrogram: shape (..., C, freq, T)
        spec = self.spectrogram(x)
        # Generate random phase uniformly distributed in [0, 2*pi) for each element.
        random_phase = torch.rand_like(spec.real) * 2 * math.pi
        # Replace the phase: combine the original magnitude with the random phase.
        spec_randomized = torch.abs(spec) * torch.exp(1j * random_phase)
        # Compute the log-scaled magnitude.
        logspec = torch.log10(torch.abs(spec_randomized) + 1e-6)
        # Rearrange dimensions back: (..., C, freq, T) -> (T, ..., C, freq)
        return logspec.movedim(-1, 0)
    
from dataclasses import dataclass
import torch

@dataclass
class SpectrogramAugment:
    """Adds frequency-level noise to a log-scaled spectrogram.
    
    This transform takes as input a log-scaled spectrogram (e.g. output from LogSpectrogram)
    with shape (T, ..., C, freq) and adds additive Gaussian noise to each frequency bin.
    The noise is generated with zero mean and a standard deviation specified by `noise_std`
    and is constant across time and channel dimensions.
    
    Args:
        noise_std (float): Standard deviation of the Gaussian noise to add.
            (default: 0.1)
    added right before SpecAugment
    """
    noise_std: float = 0.1

    def __call__(self, spec: torch.Tensor) -> torch.Tensor:
        # spec is expected to be of shape (T, ..., C, freq)
        # Create noise with shape (1, 1, ..., 1, freq) so that it is applied 
        # independently to each frequency bin and broadcasted across other dims.
        noise_shape = [1] * (spec.dim() - 1) + [spec.shape[-1]]
        noise = torch.randn(noise_shape, device=spec.device, dtype=spec.dtype) * self.noise_std
        spec_aug = spec + noise
        return spec_aug
