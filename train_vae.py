# train_vae.py

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import DataLoader

# 1) We'll reuse your "VAE" code from transforms.py but make it a standard nn.Module
#    with a typical forward method that returns decoded, mu, logvar. We'll define it here
#    for clarity, but you can import it from transforms.py if you prefer.

# class VAEModel(torch.nn.Module):
#     def __init__(self, input_dim=16, latent_dim=16):
#         super().__init__()
#         self.input_dim = input_dim
#         self.latent_dim = latent_dim
        
#         # Encoder
#         self.encoder = torch.nn.Sequential(
#             torch.nn.Linear(input_dim, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, 32),
#             torch.nn.ReLU()
#         )
#         self.mu = torch.nn.Linear(32, latent_dim)
#         self.logvar = torch.nn.Linear(32, latent_dim)
        
#         # Decoder
#         self.decoder = torch.nn.Sequential(
#             torch.nn.Linear(latent_dim, 32),
#             torch.nn.ReLU(),
#             torch.nn.Linear(32, 64),
#             torch.nn.ReLU(),
#             torch.nn.Linear(64, input_dim)
#         )
        
#     def reparameterize(self, mu, logvar):
#         std = torch.exp(0.5 * logvar)
#         eps = torch.randn_like(std)
#         return mu + eps * std

#     def forward(self, x):
#         # x shape: (batch, input_dim)
#         encoded = self.encoder(x)
#         mu = self.mu(encoded)
#         logvar = self.logvar(encoded)
#         z = self.reparameterize(mu, logvar)
#         decoded = self.decoder(z)
#         return decoded, mu, logvar


# # 2) Wrap VAEModel in a PyTorch Lightning module to handle training
# class VAELightningModule(pl.LightningModule):
#     def __init__(self, input_dim=16, latent_dim=16, lr=1e-3):
#         super().__init__()
#         self.save_hyperparameters()
#         self.model = VAEModel(input_dim, latent_dim)

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         # batch is a tensor shape (batch_size, input_dim)
#         decoded, mu, logvar = self.model(batch)
        
#         # Reconstruction loss (MSE)
#         recon_loss = F.mse_loss(decoded, batch, reduction='mean')
        
#         # KL divergence
#         kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
#         loss = recon_loss + kld
#         self.log("train_loss", loss)
#         return loss

#     def configure_optimizers(self):
#         return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


# def main():
#     # 3) Build a DataLoader from your sEMG dataset
#     #    We'll do a dummy example below. In your actual code, you can reuse
#     #    the same dataset logic from your main pipeline if you prefer.

#     # Example: random data shaped (num_samples, input_dim=16)
#     # In your real scenario, you might:
#     #   1) Load your .hdf5 or other sEMG data
#     #   2) Flatten or slice out the relevant channels so each sample is shape [16]
#     #   3) Put them in a Dataset, then DataLoader
#     dummy_data = torch.randn(2000, 16)  # e.g., 2000 samples of 16-dim
#     train_loader = DataLoader(dummy_data, batch_size=32, shuffle=True)

#     # 4) Create the LightningModule
#     vae_module = VAELightningModule(input_dim=16, latent_dim=16, lr=1e-3)

#     # 5) Train the VAE
#     trainer = pl.Trainer(max_epochs=15)
#     trainer.fit(vae_module, train_loader)

#     # 6) Save the trained checkpoint
#     trainer.save_checkpoint("vae.ckpt")
#     print("VAE checkpoint saved to vae.ckpt")

# if __name__ == "__main__":
#     main()
