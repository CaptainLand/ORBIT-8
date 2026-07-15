from __future__ import annotations

import torch
from torch import nn


class ResBlock1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = min(16, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ChartVAE(nn.Module):
    def __init__(self, input_channels: int = 48, base_channels: int = 64, latent_channels: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, base_channels, 5, padding=2),
            ResBlock1D(base_channels),
            nn.Conv1d(base_channels, 96, 4, stride=2, padding=1),
            ResBlock1D(96),
            nn.Conv1d(96, 128, 4, stride=2, padding=1),
            ResBlock1D(128),
            nn.Conv1d(128, 192, 4, stride=2, padding=1),
            ResBlock1D(192),
        )
        self.to_moments = nn.Conv1d(192, latent_channels * 2, 1)
        self.from_latent = nn.Conv1d(latent_channels, 192, 1)
        self.decoder = nn.Sequential(
            ResBlock1D(192),
            nn.ConvTranspose1d(192, 128, 4, stride=2, padding=1),
            ResBlock1D(128),
            nn.ConvTranspose1d(128, 96, 4, stride=2, padding=1),
            ResBlock1D(96),
            nn.ConvTranspose1d(96, base_channels, 4, stride=2, padding=1),
            ResBlock1D(base_channels),
            nn.GroupNorm(min(16, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, input_channels, 5, padding=2),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moments = self.to_moments(self.encoder(x))
        mean, logvar = moments.chunk(2, dim=1)
        return mean, logvar.clamp(-12.0, 8.0)

    @staticmethod
    def sample(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mean
        return mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.from_latent(z))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, logvar = self.encode(x)
        logits = self.decode(self.sample(mean, logvar))
        return logits, mean, logvar

