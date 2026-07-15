from __future__ import annotations

import torch


def event_and_count_targets(chart: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = chart.reshape(chart.shape[0], 6, 8, chart.shape[-1])
    event = grouped[:, :5].amax(dim=2)
    identity = grouped[:, 0] + grouped[:, 2] + grouped[:, 4]
    count = identity.sum(dim=1).round().long().clamp(0, 2)
    return event, count


def gap_class(gap: int) -> int:
    if gap >= 36:
        return 1
    if 19 <= gap <= 29:
        return 2
    if 14 <= gap <= 18:
        return 3
    if 10 <= gap <= 13:
        return 4
    if 7 <= gap <= 9:
        return 5
    if 4 <= gap <= 6:
        return 6
    return 7


def metrical_targets(
    event: torch.Tensor, count: torch.Tensor, valid_ticks: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    onset = (count > 0) & valid_ticks
    subdivision = torch.zeros_like(count)
    for batch_index in range(onset.shape[0]):
        ticks = torch.nonzero(onset[batch_index], as_tuple=False).flatten().tolist()
        for position, tick in enumerate(ticks):
            gaps = []
            if position:
                gaps.append(tick - ticks[position - 1])
            if position + 1 < len(ticks):
                gaps.append(ticks[position + 1] - tick)
            subdivision[batch_index, tick] = gap_class(min(gaps)) if gaps else 1

    accent = onset.float() * 0.35
    accent = torch.maximum(accent, event[:, 2] * 0.60)
    accent = torch.maximum(accent, event[:, 4] * 0.75)
    accent = torch.maximum(accent, event[:, 1])
    accent = torch.maximum(accent, (count >= 2).float())
    downbeats = torch.zeros_like(accent)
    downbeats[:, ::48] = 0.15
    accent = torch.clamp(accent + downbeats * onset, 0.0, 1.0)
    return subdivision, accent, onset
