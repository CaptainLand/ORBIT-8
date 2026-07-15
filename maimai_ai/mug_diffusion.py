from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .diffusion import ConditionedResBlock, timestep_embedding


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


class AudioStage(nn.Module):
    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.down = nn.Conv1d(input_channels, output_channels, 4, stride=2, padding=1)
        self.blocks = nn.Sequential(AudioResBlock(output_channels, 1), AudioResBlock(output_channels, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.down(x))


class SelfAttention1D(nn.Module):
    def __init__(self, channels: int, heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(16, channels), channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.norm(x).transpose(1, 2)
        attended, _ = self.attention(sequence, sequence, sequence, need_weights=False)
        return x + attended.transpose(1, 2)


class AudioPyramid(nn.Module):
    def __init__(self, input_channels: int = 132) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 48, 5, padding=2),
            AudioResBlock(48, 1),
            AudioResBlock(48, 4),
        )
        self.stage1 = AudioStage(48, 64)       # 3072 -> 1536
        self.stage2 = AudioStage(64, 64)       # 1536 -> 768
        self.stage3 = AudioStage(64, 96)       # 768 -> 384
        self.stage4 = AudioStage(96, 128)      # 384 -> 192
        self.stage5 = AudioStage(128, 192)     # 192 -> 96
        self.attention1 = SelfAttention1D(128)
        self.attention2 = SelfAttention1D(192)

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.stage2(self.stage1(self.stem(audio)))
        audio0 = self.stage3(h)
        audio1 = self.attention1(self.stage4(audio0))
        audio2 = self.attention2(self.stage5(audio1))
        return audio0, audio1, audio2


class ControlTokenizer(nn.Module):
    """Quantized MuG-style tokens: level, density, hold, slide and break ratios."""

    def __init__(self, dimension: int = 128) -> None:
        super().__init__()
        self.level = nn.Embedding(33, dimension)
        self.density = nn.Embedding(33, dimension)
        self.hold = nn.Embedding(11, dimension)
        self.slide = nn.Embedding(11, dimension)
        self.break_note = nn.Embedding(11, dimension)
        self.null_tokens = nn.Parameter(torch.randn(1, 5, dimension) * 0.02)

    def forward(self, controls: torch.Tensor, drop_mask: torch.Tensor | None = None) -> torch.Tensor:
        indices = (
            (controls[:, 0] * 2.0).round().long().clamp(0, 32),
            (controls[:, 1] / 2.0).round().long().clamp(0, 32),
            (controls[:, 2] * 10.0).round().long().clamp(0, 10),
            (controls[:, 3] * 10.0).round().long().clamp(0, 10),
            (controls[:, 4] * 10.0).round().long().clamp(0, 10),
        )
        tokens = torch.stack(
            [
                self.level(indices[0]),
                self.density(indices[1]),
                self.hold(indices[2]),
                self.slide(indices[3]),
                self.break_note(indices[4]),
            ],
            dim=1,
        )
        if drop_mask is not None:
            tokens = torch.where(drop_mask[:, None, None], self.null_tokens.expand_as(tokens), tokens)
        return tokens


class CrossAttention1D(nn.Module):
    def __init__(self, channels: int, context_channels: int = 128, heads: int = 4) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(16, channels), channels)
        self.attention = nn.MultiheadAttention(
            channels, heads, kdim=context_channels, vdim=context_channels, batch_first=True
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        query = self.norm(x).transpose(1, 2)
        attended, _ = self.attention(query, context, context, need_weights=False)
        return x + attended.transpose(1, 2)


class GlobalSequenceBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(min(16, channels), channels)
        self.gru = nn.GRU(channels, channels // 2, bidirectional=True, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence = self.norm(x).transpose(1, 2)
        sequence, _ = self.gru(sequence)
        return x + sequence.transpose(1, 2)


class MugInspiredAudioChartDiffusion(nn.Module):
    def __init__(self, latent_channels: int = 16, audio_channels: int = 132) -> None:
        super().__init__()
        embedding_dim = 384
        self.time_mlp = nn.Sequential(
            nn.Linear(128, embedding_dim), nn.SiLU(), nn.Linear(embedding_dim, embedding_dim)
        )
        self.controls = ControlTokenizer(128)
        self.audio = AudioPyramid(audio_channels)

        self.input = nn.Conv1d(latent_channels, 96, 3, padding=1)
        self.merge0 = nn.Conv1d(192, 96, 1)
        self.block0a = ConditionedResBlock(96, embedding_dim)
        self.block0b = ConditionedResBlock(96, embedding_dim)
        self.cross0 = CrossAttention1D(96)

        self.down1 = nn.Conv1d(96, 128, 4, stride=2, padding=1)
        self.merge1 = nn.Conv1d(256, 128, 1)
        self.block1a = ConditionedResBlock(128, embedding_dim)
        self.block1b = ConditionedResBlock(128, embedding_dim)
        self.cross1 = CrossAttention1D(128)

        self.down2 = nn.Conv1d(128, 192, 4, stride=2, padding=1)
        self.merge2 = nn.Conv1d(384, 192, 1)
        self.middle1 = ConditionedResBlock(192, embedding_dim)
        self.middle_attention = CrossAttention1D(192)
        self.middle_sequence = GlobalSequenceBlock(192)
        self.middle2 = ConditionedResBlock(192, embedding_dim)

        self.up1 = nn.ConvTranspose1d(192, 128, 4, stride=2, padding=1)
        self.up_merge1 = nn.Conv1d(384, 128, 1)
        self.up_block1a = ConditionedResBlock(128, embedding_dim)
        self.up_block1b = ConditionedResBlock(128, embedding_dim)
        self.up_cross1 = CrossAttention1D(128)

        self.up0 = nn.ConvTranspose1d(128, 96, 4, stride=2, padding=1)
        self.up_merge0 = nn.Conv1d(288, 96, 1)
        self.up_block0a = ConditionedResBlock(96, embedding_dim)
        self.up_block0b = ConditionedResBlock(96, embedding_dim)
        self.up_cross0 = CrossAttention1D(96)
        self.output_norm = nn.GroupNorm(16, 96)
        self.output = nn.Conv1d(96, latent_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

        self.rhythm_control = nn.Linear(128, 96)
        self.rhythm_head = nn.Conv1d(96, 5, 1)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        audio: torch.Tensor,
        controls: torch.Tensor,
        *,
        control_drop_mask: torch.Tensor | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        embedding = self.time_mlp(timestep_embedding(timesteps, 128))
        context = self.controls(controls, control_drop_mask)
        audio0, audio1, audio2 = self.audio(audio)

        skip0 = self.merge0(torch.cat([self.input(noisy_latent), audio0], dim=1))
        skip0 = self.block0b(self.block0a(skip0, embedding), embedding)
        skip0 = self.cross0(skip0, context)

        skip1 = self.merge1(torch.cat([self.down1(skip0), audio1], dim=1))
        skip1 = self.block1b(self.block1a(skip1, embedding), embedding)
        skip1 = self.cross1(skip1, context)

        h = self.merge2(torch.cat([self.down2(skip1), audio2], dim=1))
        h = self.middle1(h, embedding)
        h = self.middle_attention(h, context)
        h = self.middle_sequence(h)
        h = self.middle2(h, embedding)

        h = self.up1(h)
        h = self.up_merge1(torch.cat([h, skip1, audio1], dim=1))
        h = self.up_block1b(self.up_block1a(h, embedding), embedding)
        h = self.up_cross1(h, context)
        h = self.up0(h)
        h = self.up_merge0(torch.cat([h, skip0, audio0], dim=1))
        h = self.up_block0b(self.up_block0a(h, embedding), embedding)
        h = self.up_cross0(h, context)
        prediction = self.output(F.silu(self.output_norm(h)))

        if not return_aux:
            return prediction
        rhythm_context = context if control_drop_mask is None else self.controls(controls)
        control_summary = self.rhythm_control(rhythm_context.mean(dim=1))[:, :, None]
        rhythm_logits = self.rhythm_head(audio0 + control_summary)
        return prediction, rhythm_logits
