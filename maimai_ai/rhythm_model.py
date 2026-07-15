from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1) -> None:
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


class RhythmPlanModel(nn.Module):
    def __init__(self, input_channels: int = 132) -> None:
        super().__init__()
        self.control = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 160))
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, 5, padding=2),
            ResidualBlock(64, 1),
            ResidualBlock(64, 4),
        )
        self.down1 = nn.Sequential(nn.Conv1d(64, 96, 4, stride=2, padding=1), ResidualBlock(96, 1), ResidualBlock(96, 4))
        self.down2 = nn.Sequential(nn.Conv1d(96, 128, 4, stride=2, padding=1), ResidualBlock(128, 1), ResidualBlock(128, 4))
        self.down3 = nn.Sequential(nn.Conv1d(128, 160, 4, stride=2, padding=1), ResidualBlock(160, 1), ResidualBlock(160, 4))
        self.middle = nn.Sequential(ResidualBlock(160, 2), ResidualBlock(160, 8), ResidualBlock(160, 16))
        self.up2 = nn.ConvTranspose1d(160, 128, 4, stride=2, padding=1)
        self.merge2 = nn.Sequential(nn.Conv1d(256, 128, 1), ResidualBlock(128, 2), ResidualBlock(128, 8))
        self.up1 = nn.ConvTranspose1d(128, 96, 4, stride=2, padding=1)
        self.merge1 = nn.Sequential(nn.Conv1d(192, 96, 1), ResidualBlock(96, 2), ResidualBlock(96, 8))
        self.up0 = nn.ConvTranspose1d(96, 64, 4, stride=2, padding=1)
        self.merge0 = nn.Sequential(nn.Conv1d(128, 64, 1), ResidualBlock(64, 2), ResidualBlock(64, 8))
        self.output_norm = nn.GroupNorm(16, 64)
        self.event_head = nn.Conv1d(64, 5, 3, padding=1)
        self.count_head = nn.Conv1d(64, 3, 3, padding=1)
        self.onset_head = nn.Conv1d(64, 1, 3, padding=1)

    def forward(self, audio: torch.Tensor, controls: torch.Tensor) -> dict[str, torch.Tensor]:
        skip0 = self.stem(audio)
        skip1 = self.down1(skip0)
        skip2 = self.down2(skip1)
        h = self.down3(skip2)
        normalized = controls.clone()
        normalized[:, 0] = (normalized[:, 0] - 10.0) / 5.0
        normalized[:, 1] = normalized[:, 1] / 16.0
        h = self.middle(h + self.control(normalized)[:, :, None])
        h = self.merge2(torch.cat([self.up2(h), skip2], dim=1))
        h = self.merge1(torch.cat([self.up1(h), skip1], dim=1))
        h = self.merge0(torch.cat([self.up0(h), skip0], dim=1))
        h = F.silu(self.output_norm(h))
        return {"event": self.event_head(h), "count": self.count_head(h), "onset": self.onset_head(h)}
