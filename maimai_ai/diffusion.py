from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def timestep_embedding(timesteps: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=timesteps.device, dtype=torch.float32) / max(1, half - 1)
    )
    angles = timesteps.float()[:, None] * frequencies[None]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dimension % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ConditionedResBlock(nn.Module):
    def __init__(self, channels: int, embedding_dim: int) -> None:
        super().__init__()
        groups = min(16, channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.embedding = nn.Linear(embedding_dim, channels * 2)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, embedding: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        scale, shift = self.embedding(F.silu(embedding)).chunk(2, dim=1)
        h = self.norm2(h) * (1.0 + scale[:, :, None]) + shift[:, :, None]
        return x + self.conv2(F.silu(h))


class AudioResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(min(16, channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(min(16, channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AudioChartDiffusion(nn.Module):
    def __init__(self, latent_channels: int = 16, audio_channels: int = 84, base_channels: int = 64) -> None:
        super().__init__()
        embedding_dim = 256
        self.time_mlp = nn.Sequential(
            nn.Linear(128, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.level_mlp = nn.Sequential(
            nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, embedding_dim)
        )
        self.input = nn.Conv1d(latent_channels, base_channels, 3, padding=1)
        self.audio_encoder = nn.Sequential(
            nn.Conv1d(audio_channels, base_channels, 5, padding=2),
            AudioResBlock(base_channels, 1),
            AudioResBlock(base_channels, 2),
            AudioResBlock(base_channels, 4),
            AudioResBlock(base_channels, 8),
        )
        self.audio_level = nn.Linear(1, base_channels)
        self.rhythm_head = nn.Conv1d(base_channels, 5, 1)
        self.block0a = ConditionedResBlock(base_channels, embedding_dim)
        self.block0b = ConditionedResBlock(base_channels, embedding_dim)

        self.down1 = nn.Conv1d(base_channels, 128, 4, stride=2, padding=1)
        self.audio1 = nn.Conv1d(base_channels, 128, 3, padding=1)
        self.block1a = ConditionedResBlock(128, embedding_dim)
        self.block1b = ConditionedResBlock(128, embedding_dim)

        self.down2 = nn.Conv1d(128, 192, 4, stride=2, padding=1)
        self.audio2 = nn.Conv1d(128, 192, 3, padding=1)
        self.middle1 = ConditionedResBlock(192, embedding_dim)
        self.middle2 = ConditionedResBlock(192, embedding_dim)

        self.up1 = nn.ConvTranspose1d(192, 128, 4, stride=2, padding=1)
        self.merge1 = nn.Conv1d(256, 128, 1)
        self.up_block1 = ConditionedResBlock(128, embedding_dim)
        self.up0 = nn.ConvTranspose1d(128, base_channels, 4, stride=2, padding=1)
        self.merge0 = nn.Conv1d(base_channels * 2, base_channels, 1)
        self.up_block0 = ConditionedResBlock(base_channels, embedding_dim)
        self.output = nn.Sequential(
            nn.GroupNorm(min(16, base_channels), base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, latent_channels, 3, padding=1),
        )

    def encode_audio(self, audio: torch.Tensor, level: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized_level = ((level - 10.0) / 5.0).unsqueeze(1)
        audio0 = self.audio_encoder(audio) + self.audio_level(normalized_level)[:, :, None]
        audio1 = self.audio1(F.avg_pool1d(audio0, 2))
        audio2 = self.audio2(F.avg_pool1d(audio1, 2))
        return audio0, audio1, audio2

    def predict_rhythm(self, audio: torch.Tensor, level: torch.Tensor) -> torch.Tensor:
        audio0, _, _ = self.encode_audio(audio, level)
        return self.rhythm_head(audio0)

    def forward(
        self, noisy_latent: torch.Tensor, timesteps: torch.Tensor, audio: torch.Tensor, level: torch.Tensor
    ) -> torch.Tensor:
        embedding = self.time_mlp(timestep_embedding(timesteps, 128))
        normalized_level = ((level - 10.0) / 5.0).unsqueeze(1)
        embedding = embedding + self.level_mlp(normalized_level)

        audio0, audio1, audio2 = self.encode_audio(audio, level)
        skip0 = self.input(noisy_latent) + audio0
        skip0 = self.block0b(self.block0a(skip0, embedding), embedding)
        skip1 = self.down1(skip0) + audio1
        skip1 = self.block1b(self.block1a(skip1, embedding), embedding)
        h = self.down2(skip1) + audio2
        h = self.middle2(self.middle1(h, embedding), embedding)

        h = self.up1(h)
        h = self.up_block1(self.merge1(torch.cat([h, skip1], dim=1)), embedding)
        h = self.up0(h)
        h = self.up_block0(self.merge0(torch.cat([h, skip0], dim=1)), embedding)
        return self.output(h)


def cosine_schedule(steps: int, device: torch.device | None = None) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1, device=device)
    cumulative = torch.cos(((x / steps) + 0.008) / 1.008 * math.pi * 0.5).square()
    cumulative = cumulative / cumulative[0]
    betas = 1.0 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-5, 0.999)
