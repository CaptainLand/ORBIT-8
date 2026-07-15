from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from maimai_ai.rhythm_model import ResidualBlock
from trans1.model import HybridStack
from v2.rhythm_model import SUBDIVISION_NAMES


class OrbitV2RhythmModel16M(nn.Module):
    """Wider hierarchical rhythm model with a 12-layer 288-dim sequence core."""

    def __init__(self, input_channels: int = 132) -> None:
        super().__init__()
        self.control = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 160))
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, 5, padding=2),
            ResidualBlock(64, 1),
            ResidualBlock(64, 4),
        )
        self.down1 = nn.Sequential(
            nn.Conv1d(64, 96, 4, stride=2, padding=1),
            ResidualBlock(96, 1),
            ResidualBlock(96, 4),
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(96, 128, 4, stride=2, padding=1),
            ResidualBlock(128, 1),
            ResidualBlock(128, 4),
        )
        self.down3 = nn.Sequential(
            nn.Conv1d(128, 160, 4, stride=2, padding=1),
            ResidualBlock(160, 1),
            ResidualBlock(160, 4),
        )
        self.core_in = nn.Conv1d(160, 288, 1)
        self.sequence_core = HybridStack(
            dimension=288,
            layers=12,
            heads=8,
            kv_heads=2,
            feedforward=1152,
            dropout=0.12,
            causal=False,
        )
        self.core_out = nn.Conv1d(288, 160, 1)
        self.up2 = nn.ConvTranspose1d(160, 128, 4, stride=2, padding=1)
        self.merge2 = nn.Sequential(
            nn.Conv1d(256, 128, 1), ResidualBlock(128, 2), ResidualBlock(128, 8)
        )
        self.up1 = nn.ConvTranspose1d(128, 96, 4, stride=2, padding=1)
        self.merge1 = nn.Sequential(
            nn.Conv1d(192, 96, 1), ResidualBlock(96, 2), ResidualBlock(96, 8)
        )
        self.up0 = nn.ConvTranspose1d(96, 64, 4, stride=2, padding=1)
        self.merge0 = nn.Sequential(
            nn.Conv1d(128, 64, 1), ResidualBlock(64, 2), ResidualBlock(64, 8)
        )
        self.output_norm = nn.GroupNorm(16, 64)
        self.event_head = nn.Conv1d(64, 5, 3, padding=1)
        self.count_head = nn.Conv1d(64, 3, 3, padding=1)
        self.onset_head = nn.Conv1d(64, 1, 3, padding=1)
        self.subdivision_head = nn.Conv1d(64, len(SUBDIVISION_NAMES), 3, padding=1)
        self.accent_head = nn.Conv1d(64, 1, 3, padding=1)
        self._initialize_expansion()

    def _initialize_expansion(self) -> None:
        nn.init.zeros_(self.core_in.weight)
        nn.init.zeros_(self.core_in.bias)
        nn.init.zeros_(self.core_out.weight)
        nn.init.zeros_(self.core_out.bias)
        with torch.no_grad():
            identity = torch.arange(160)
            self.core_in.weight[identity, identity, 0] = 1.0
            self.core_out.weight[identity, identity, 0] = 1.0
        for layer in self.sequence_core.layers:
            if hasattr(layer, "attention"):
                nn.init.zeros_(layer.attention.output.weight)
            nn.init.zeros_(layer.feedforward.output.weight)
            if hasattr(layer, "output"):
                nn.init.zeros_(layer.output.weight)

    def forward(self, audio: torch.Tensor, controls: torch.Tensor) -> dict[str, torch.Tensor]:
        skip0 = self.stem(audio)
        skip1 = self.down1(skip0)
        skip2 = self.down2(skip1)
        hidden = self.down3(skip2)
        normalized = controls.clone()
        normalized[:, 0] = (normalized[:, 0] - 10.0) / 5.0
        normalized[:, 1] = normalized[:, 1] / 16.0
        hidden = hidden + self.control(normalized)[:, :, None]
        hidden = self.core_in(hidden)
        tokens = hidden.transpose(1, 2)
        padding = torch.zeros(tokens.shape[:2], device=tokens.device, dtype=torch.bool)
        hidden = self.sequence_core(tokens, padding).transpose(1, 2)
        hidden = self.core_out(hidden)
        hidden = self.merge2(torch.cat([self.up2(hidden), skip2], dim=1))
        hidden = self.merge1(torch.cat([self.up1(hidden), skip1], dim=1))
        hidden = self.merge0(torch.cat([self.up0(hidden), skip0], dim=1))
        hidden = F.silu(self.output_norm(hidden))
        return {
            "event": self.event_head(hidden),
            "count": self.count_head(hidden),
            "onset": self.onset_head(hidden),
            "subdivision": self.subdivision_head(hidden),
            "accent": self.accent_head(hidden),
        }


def inherited_module_names() -> tuple[str, ...]:
    return (
        "control",
        "stem",
        "down1",
        "down2",
        "down3",
        "up2",
        "merge2",
        "up1",
        "merge1",
        "up0",
        "merge0",
        "output_norm",
        "event_head",
        "count_head",
        "onset_head",
        "subdivision_head",
        "accent_head",
    )
