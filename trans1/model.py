from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from maimai_ai.arranger import MAX_EVENTS, OPERATORS, OfficialPatternArranger, PlanEncoder
from maimai_ai.patterns import (
    PATTERN_INTERACTION,
    PATTERN_JACK,
    PATTERN_NAMES,
    PATTERN_NONE,
    PATTERN_SWEEP,
)


class RMSNorm(nn.Module):
    def __init__(self, dimension: int, epsilon: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dimension))
        self.epsilon = epsilon

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scale = value.float().pow(2).mean(dim=-1, keepdim=True).add(self.epsilon).rsqrt()
        return (value * scale.to(value.dtype)) * self.weight


def apply_rope(value: torch.Tensor) -> torch.Tensor:
    length = value.shape[-2]
    width = value.shape[-1]
    positions = torch.arange(length, device=value.device, dtype=torch.float32)
    frequencies = 1.0 / (10000 ** (torch.arange(0, width, 2, device=value.device).float() / width))
    angles = positions[:, None] * frequencies[None]
    cosine = angles.cos().to(value.dtype)[None, None]
    sine = angles.sin().to(value.dtype)[None, None]
    even, odd = value[..., 0::2], value[..., 1::2]
    return torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1).flatten(-2)


class GroupedQueryAttention(nn.Module):
    def __init__(self, dimension: int, query_heads: int, key_value_heads: int, dropout: float) -> None:
        super().__init__()
        if dimension % query_heads or query_heads % key_value_heads:
            raise ValueError("Invalid grouped-query attention dimensions")
        self.query_heads = query_heads
        self.key_value_heads = key_value_heads
        self.head_width = dimension // query_heads
        self.dropout = dropout
        self.query = nn.Linear(dimension, query_heads * self.head_width, bias=False)
        self.key = nn.Linear(dimension, key_value_heads * self.head_width, bias=False)
        self.value = nn.Linear(dimension, key_value_heads * self.head_width, bias=False)
        self.output = nn.Linear(dimension, dimension, bias=False)

    def forward(self, hidden: torch.Tensor, padding: torch.Tensor, causal: bool) -> torch.Tensor:
        batch, length, _ = hidden.shape
        query = self.query(hidden).view(batch, length, self.query_heads, self.head_width).transpose(1, 2)
        key = self.key(hidden).view(batch, length, self.key_value_heads, self.head_width).transpose(1, 2)
        value = self.value(hidden).view(batch, length, self.key_value_heads, self.head_width).transpose(1, 2)
        repeat = self.query_heads // self.key_value_heads
        key = key.repeat_interleave(repeat, dim=1)
        value = value.repeat_interleave(repeat, dim=1)
        query = apply_rope(F.normalize(query, dim=-1) * (self.head_width ** 0.5))
        key = apply_rope(F.normalize(key, dim=-1))
        allowed = (~padding)[:, None, None, :].expand(-1, 1, length, -1)
        if causal:
            allowed = allowed & torch.ones(length, length, device=hidden.device, dtype=torch.bool).tril()[None, None]
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=allowed,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, length, -1)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, dimension: int, hidden: int) -> None:
        super().__init__()
        self.input = nn.Linear(dimension, hidden * 2, bias=False)
        self.output = nn.Linear(hidden, dimension, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        gate, content = self.input(value).chunk(2, dim=-1)
        return self.output(F.silu(gate) * content)


class ModernAttentionBlock(nn.Module):
    def __init__(self, dimension: int, heads: int, kv_heads: int, feedforward: int, dropout: float, causal: bool) -> None:
        super().__init__()
        self.causal = causal
        self.attention_norm = RMSNorm(dimension)
        self.attention = GroupedQueryAttention(dimension, heads, kv_heads, dropout)
        self.feedforward_norm = RMSNorm(dimension)
        self.feedforward = SwiGLU(dimension, feedforward)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        value = value + self.dropout(self.attention(self.attention_norm(value), padding, self.causal))
        return value + self.dropout(self.feedforward(self.feedforward_norm(value)))


class GatedSequenceMixer(nn.Module):
    """Linear-cost local sequence mixer used between global attention blocks."""

    def __init__(self, dimension: int, feedforward: int, dropout: float, causal: bool) -> None:
        super().__init__()
        self.causal = causal
        self.norm = RMSNorm(dimension)
        self.input = nn.Linear(dimension, dimension * 2, bias=False)
        self.depthwise = nn.Conv1d(
            dimension, dimension, kernel_size=7, padding=6 if causal else 3,
            groups=dimension, bias=False,
        )
        self.output = nn.Linear(dimension, dimension, bias=False)
        self.feedforward_norm = RMSNorm(dimension)
        self.feedforward = SwiGLU(dimension, feedforward)
        self.dropout = nn.Dropout(dropout)

    def forward(self, value: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        content, gate = self.input(self.norm(value)).chunk(2, dim=-1)
        mixed = self.depthwise(content.transpose(1, 2))
        if self.causal:
            mixed = mixed[..., :value.shape[1]]
        mixed = mixed.transpose(1, 2) * torch.sigmoid(gate)
        mixed = mixed.masked_fill(padding[..., None], 0.0)
        value = value + self.dropout(self.output(F.silu(mixed)))
        return value + self.dropout(self.feedforward(self.feedforward_norm(value)))


class HybridStack(nn.Module):
    def __init__(self, dimension: int, layers: int, heads: int, kv_heads: int, feedforward: int, dropout: float, causal: bool) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            ModernAttentionBlock(dimension, heads, kv_heads, feedforward, dropout, causal)
            if index % 2 == 0 else
            GatedSequenceMixer(dimension, feedforward, dropout, causal)
            for index in range(layers)
        ])
        self.norm = RMSNorm(dimension)

    def forward(self, value: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value, padding)
        return self.norm(value)


class Trans1Arranger(nn.Module):
    """Parallel iterative Transformer arranger for the isolated Trans-1 experiment."""

    def __init__(
        self,
        dimension: int = 128,
        heads: int = 8,
        key_value_heads: int = 2,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        feedforward: int = 512,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.dimension = dimension
        self.plan = PlanEncoder(dimension)
        self.plan_context = HybridStack(
            dimension, encoder_layers, heads, key_value_heads, feedforward, dropout, False
        )
        self.previous_delta = nn.Embedding(9, 32)
        self.pattern = nn.Embedding(len(PATTERN_NAMES), dimension)
        self.decoder_input = nn.Linear(dimension + 32, dimension)
        self.decoder = HybridStack(
            dimension, decoder_layers, heads, key_value_heads, feedforward, dropout, True
        )
        self.delta_head = nn.Linear(dimension, 8)
        self.operator_head = nn.Linear(dimension, len(OPERATORS))
        self.endpoint_head = nn.Linear(dimension, 8)
        self.branch_head = nn.Linear(dimension, 2)
        self.pattern_head = nn.Linear(dimension, len(PATTERN_NAMES))
        nn.init.zeros_(self.pattern.weight)
        nn.init.zeros_(self.pattern_head.weight)
        nn.init.zeros_(self.pattern_head.bias)
        self.pattern_head.bias.data[PATTERN_NONE] = 2.0

    @staticmethod
    def _padding_mask(batch: dict[str, torch.Tensor], length: int) -> torch.Tensor:
        if "mask" in batch:
            return batch["mask"][:, :length] <= 0.5
        valid = batch.get("valid_length")
        if valid is None:
            return torch.zeros(
                batch["tick"].shape[0], length, device=batch["tick"].device, dtype=torch.bool
            )
        positions = torch.arange(length, device=batch["tick"].device)[None]
        return positions >= valid[:, None]

    def encode_plan(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        plan = self.plan(batch)
        length = plan.shape[1]
        padding = self._padding_mask(batch, length)
        return self.plan_context(plan, padding)

    def decode(
        self,
        context: torch.Tensor,
        pattern_tokens: torch.Tensor,
        previous_delta: torch.Tensor,
        padding: torch.Tensor,
    ) -> torch.Tensor:
        conditioned = context + self.pattern(pattern_tokens)
        hidden = self.decoder_input(
            torch.cat([conditioned, self.previous_delta(previous_delta)], dim=-1)
        )
        return self.decoder(hidden, padding)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context = self.encode_plan(batch)
        pattern_logits = self.pattern_head(context)
        pattern_tokens = batch.get("target_pattern", pattern_logits.argmax(dim=-1))
        padding = self._padding_mask(batch, context.shape[1])
        hidden = self.decode(context, pattern_tokens, batch["previous_delta"], padding)
        return {
            "delta": self.delta_head(hidden),
            "operator": self.operator_head(hidden),
            "endpoint": self.endpoint_head(hidden),
            "branch": self.branch_head(hidden),
            "pattern": pattern_logits,
        }

    @staticmethod
    def _segment_pattern_tokens(
        batch: dict[str, torch.Tensor],
        pattern_logits: torch.Tensor,
        pattern_heat: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> torch.Tensor:
        return OfficialPatternArranger._segment_pattern_tokens(batch, pattern_logits, pattern_heat)

    @staticmethod
    def _apply_pattern_constraints(
        delta_logits: torch.Tensor,
        pattern_tokens: torch.Tensor,
        valid_lengths: torch.Tensor,
    ) -> torch.Tensor:
        delta = delta_logits.argmax(dim=-1)
        delta[:, 0] = 0
        for batch_index in range(delta.shape[0]):
            length = int(valid_lengths[batch_index])
            for position in range(1, length):
                pattern = int(pattern_tokens[batch_index, position])
                previous_pattern = int(pattern_tokens[batch_index, position - 1])
                if pattern != previous_pattern or pattern == PATTERN_NONE:
                    continue
                if pattern == PATTERN_INTERACTION:
                    if position >= 2 and int(pattern_tokens[batch_index, position - 2]) == pattern:
                        delta[batch_index, position] = (-delta[batch_index, position - 1]) % 8
                    else:
                        candidates = torch.tensor([1, 2, 6, 7], device=delta.device)
                        scores = delta_logits[batch_index, position, candidates]
                        delta[batch_index, position] = candidates[scores.argmax()]
                elif pattern == PATTERN_SWEEP:
                    if position >= 2 and int(pattern_tokens[batch_index, position - 2]) == pattern:
                        delta[batch_index, position] = delta[batch_index, position - 1]
                    else:
                        candidates = torch.tensor([1, 7], device=delta.device)
                        scores = delta_logits[batch_index, position, candidates]
                        delta[batch_index, position] = candidates[scores.argmax()]
                elif pattern == PATTERN_JACK:
                    delta[batch_index, position] = 0
        return delta

    @torch.inference_mode()
    def generate(
        self,
        batch: dict[str, torch.Tensor],
        first_lane: int = 0,
        operator_bias: torch.Tensor | None = None,
        enable_patterns: bool = True,
        pattern_heat: tuple[float, float, float] = (1.0, 1.0, 1.0),
        refinement_steps: int = 3,
    ) -> dict[str, torch.Tensor]:
        context = self.encode_plan(batch)
        length = context.shape[1]
        padding = self._padding_mask(batch, length)
        valid_lengths = batch.get(
            "valid_length",
            torch.full((context.shape[0],), length, device=context.device, dtype=torch.long),
        )
        pattern_logits = self.pattern_head(context)
        if enable_patterns:
            pattern_tokens = self._segment_pattern_tokens(batch, pattern_logits, pattern_heat)
        else:
            pattern_tokens = torch.zeros(
                pattern_logits.shape[:2], device=context.device, dtype=torch.long
            )

        previous = torch.full(
            (context.shape[0], length), 8, device=context.device, dtype=torch.long
        )
        hidden = context
        delta_logits = torch.zeros(
            context.shape[0], length, 8, device=context.device, dtype=context.dtype
        )
        delta = torch.zeros(context.shape[:2], device=context.device, dtype=torch.long)
        for _ in range(max(1, refinement_steps)):
            hidden = self.decode(context, pattern_tokens, previous, padding)
            delta_logits = self.delta_head(hidden)
            delta = self._apply_pattern_constraints(delta_logits, pattern_tokens, valid_lengths)
            previous.fill_(8)
            previous[:, 1:] = delta[:, :-1]

        lanes = (first_lane + torch.cumsum(delta, dim=1)) % 8
        operator_logits = self.operator_head(hidden)
        if operator_bias is not None:
            operator_logits = operator_logits + operator_bias.to(operator_logits.device)
        return {
            "delta": delta_logits,
            "lane": lanes,
            "operator": operator_logits.argmax(dim=-1),
            "operator_logits": operator_logits,
            "endpoint": self.endpoint_head(hidden).argmax(dim=-1),
            "branch": self.branch_head(hidden).argmax(dim=-1),
            "pattern": pattern_tokens,
            "pattern_logits": pattern_logits,
        }
