from torch import nn
import torch
import torch.nn.functional as F

class Encoder1DCNN(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 64, kernel_size=5, stride=2, padding=2),  # 200 -> 100
            nn.BatchNorm1d(64), nn.SiLU(),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2), # 100 -> 50
            nn.BatchNorm1d(128), nn.SiLU(),
            nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2), # 50 -> 25
            nn.BatchNorm1d(256), nn.SiLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256), nn.SiLU(),
        )
        self.len_after = 25
        self.fc_mu = nn.Linear(256 * self.len_after, latent_dim)
        self.fc_logvar = nn.Linear(256 * self.len_after, latent_dim)

    def forward(self, x):
        # x: (B,6,200)
        h = self.conv(x)
        h = h.flatten(1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder1DCNN(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.len_after = 25
        self.fc = nn.Linear(latent_dim, 256 * self.len_after)
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(256, 256, kernel_size=4, stride=2, padding=1),  # 25->50
            nn.BatchNorm1d(256), nn.SiLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),  # 50->100
            nn.BatchNorm1d(128), nn.SiLU(),
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=1),   # 100->200
            nn.BatchNorm1d(64), nn.SiLU(),
            nn.Conv1d(64, 6, kernel_size=3, padding=1),
        )

    def forward(self, z):
        h = self.fc(z)
        h = h.view(z.size(0), 256, self.len_after)
        x_recon = self.deconv(h)
        return x_recon


class VAE(nn.Module):
    def __init__(self, latent_dim=32, beta=1.0):
        super().__init__()
        self.encoder = Encoder1DCNN(latent_dim)
        self.decoder = Decoder1DCNN(latent_dim)
        self.latent_dim = latent_dim
        self.beta = beta

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

    @staticmethod
    def loss_fn(x, x_recon, mu, logvar, beta=1.0):
        recon = F.mse_loss(x_recon, x, reduction='mean')
        kld = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return recon + beta * kld, recon.item(), kld.item()
