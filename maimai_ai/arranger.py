from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from maimai_ai.patterns import (
    MIRROR_MODES,
    MAX_JACK_PATTERN_SHARE,
    PATTERN_INTERACTION,
    PATTERN_JACK,
    PATTERN_NAMES,
    PATTERN_NONE,
    PATTERN_SWEEP,
    PATTERN_TRAINING_WEIGHTS,
    detect_pattern_labels,
    mirror_event,
)


MAX_EVENTS = 384
OPERATORS = ["-", "<", ">", "v", "V", "p", "q", "pp", "qq", "s", "z", "w"]
OPERATOR_TO_ID = {operator: index for index, operator in enumerate(OPERATORS)}
TYPE_TO_ID = {"tap": 0, "hold": 1, "slide": 2}


def operator_calibration_bias(prepared_dir: str | Path) -> torch.Tensor:
    catalog = json.loads((Path(prepared_dir) / "star_catalog.json").read_text(encoding="utf-8"))
    counts = Counter()
    for template in catalog:
        for operator in template["operators"]:
            counts[operator] += template["count"] / len(template["operators"])
    total = sum(counts.values())
    training_weights = [
        min(4.0, math.sqrt(total / (len(OPERATORS) * max(1.0, counts[operator]))))
        for operator in OPERATORS
    ]
    priors = [counts[operator] / total for operator in OPERATORS]
    # Undo weighted-CE distortion, then softly restore the official-chart prior.
    return torch.tensor(
        [-math.log(weight) + 0.55 * math.log(max(prior, 1e-8)) for weight, prior in zip(training_weights, priors)],
        dtype=torch.float32,
    )


def operator_priors(prepared_dir: str | Path, level: float) -> torch.Tensor:
    catalog = json.loads((Path(prepared_dir) / "star_catalog.json").read_text(encoding="utf-8"))
    counts = Counter()
    level_key = f"{level:.1f}"
    for template in catalog:
        weight = template.get("levels", {}).get(level_key, 0)
        if not weight:
            weight = template.get("count_12_15", 0)
        for operator in template["operators"]:
            counts[operator] += weight / len(template["operators"])
    values = torch.tensor([counts[operator] for operator in OPERATORS], dtype=torch.float32)
    return values / values.sum().clamp_min(1.0)


class OraclePlanDataset(Dataset):
    def __init__(
        self,
        prepared_dir: str | Path,
        split: str,
        *,
        corrupt_previous: float = 0.0,
        mirror_augment: bool = False,
    ) -> None:
        self.root = Path(prepared_dir)
        self.rows = [
            json.loads(line)
            for line in (self.root / "windows.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(line)["split"] == split
        ]
        if mirror_augment:
            self.rows = [
                {**row, "mirror_mode": mode, "window_id": f"{row['window_id']}__{mode}"}
                for row in self.rows
                for mode in MIRROR_MODES
            ]
        else:
            self.rows = [{**row, "mirror_mode": "normal"} for row in self.rows]
        self.corrupt_previous = corrupt_previous
        self.event_cache: dict[str, list[dict]] = {}
        self.pattern_cache: dict[str, list[int]] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def _events(self, chart_id: str) -> list[dict]:
        if chart_id not in self.event_cache:
            payload = json.loads((self.root / "events" / f"{chart_id}.json").read_text(encoding="utf-8"))
            self.event_cache[chart_id] = payload["events"]
        return self.event_cache[chart_id]

    def _patterns(self, chart_id: str) -> list[int]:
        if chart_id not in self.pattern_cache:
            self.pattern_cache[chart_id] = detect_pattern_labels(self._events(chart_id))
        return self.pattern_cache[chart_id]

    def _encode_row(self, row: dict) -> dict[str, torch.Tensor | str]:
        source_with_patterns = [
            (mirror_event(event, row["mirror_mode"]), pattern)
            for event, pattern in zip(self._events(row["chart_id"]), self._patterns(row["chart_id"]))
            if row["tick_start"] <= event["tick"] < row["tick_end"]
        ]
        source_with_patterns.sort(key=lambda item: (item[0]["tick"], item[0]["lane"], item[0]["note_type"]))
        source_with_patterns = source_with_patterns[:MAX_EVENTS]
        source = [item[0] for item in source_with_patterns]
        source_patterns = [item[1] for item in source_with_patterns]
        length = len(source)
        tick = np.zeros(MAX_EVENTS, dtype=np.int64)
        event_type = np.zeros(MAX_EVENTS, dtype=np.int64)
        duration = np.zeros(MAX_EVENTS, dtype=np.int64)
        is_break = np.zeros(MAX_EVENTS, dtype=np.int64)
        is_ex = np.zeros(MAX_EVENTS, dtype=np.int64)
        simultaneous = np.zeros(MAX_EVENTS, dtype=np.int64)
        target_delta = np.zeros(MAX_EVENTS, dtype=np.int64)
        target_operator = np.full(MAX_EVENTS, -100, dtype=np.int64)
        target_endpoint = np.full(MAX_EVENTS, -100, dtype=np.int64)
        target_branch = np.full(MAX_EVENTS, -100, dtype=np.int64)
        target_pattern = np.zeros(MAX_EVENTS, dtype=np.int64)
        mask = np.zeros(MAX_EVENTS, dtype=np.float32)
        lane_loss_mask = np.zeros(MAX_EVENTS, dtype=np.float32)

        previous_lane = None
        previous_tick = None
        same_tick_index = 0
        deltas = []
        for position, event in enumerate(source):
            local_tick = int(event["tick"] - row["tick_start"])
            lane = int(event["lane"]) - 1
            tick[position] = local_tick
            event_type[position] = TYPE_TO_ID[event["note_type"]]
            is_break[position] = int(event.get("is_break", False))
            is_ex[position] = int(event.get("is_ex", False))
            if event["tick"] == previous_tick:
                same_tick_index += 1
            else:
                same_tick_index = 0
            simultaneous[position] = min(1, same_tick_index)
            target_pattern[position] = source_patterns[position]
            if event["note_type"] == "hold" and event.get("duration"):
                duration[position] = min(64, int(event["duration"].get("duration_ticks") or 0) // 6)
            elif event["note_type"] == "slide" and event.get("branches"):
                branch = event["branches"][0]
                branch_duration = branch.get("duration") or {}
                duration[position] = min(64, int(branch_duration.get("duration_ticks") or 0) // 6)
                operator = branch.get("operator", "-")
                target_operator[position] = OPERATOR_TO_ID.get(operator, 0)
                target_endpoint[position] = (int(branch.get("end_lane", event["lane"])) - int(event["lane"])) % 8
                target_branch[position] = min(1, len(event["branches"]) - 1)
            if previous_lane is not None:
                delta = (lane - previous_lane) % 8
                target_delta[position] = delta
                lane_loss_mask[position] = 1.0
                deltas.append(delta)
            else:
                deltas.append(0)
            previous_lane = lane
            previous_tick = event["tick"]
            mask[position] = 1.0

        previous_delta = np.full(MAX_EVENTS, 8, dtype=np.int64)  # 8 is BOS/MASK.
        if length > 1:
            previous_delta[1:length] = np.asarray(deltas[:-1], dtype=np.int64)
        if self.corrupt_previous > 0 and length > 1:
            corrupt = np.random.random(length) < self.corrupt_previous
            corrupt[0] = False
            previous_delta[:length][corrupt] = 8

        return {
            "tick": torch.from_numpy(tick),
            "event_type": torch.from_numpy(event_type),
            "duration": torch.from_numpy(duration),
            "is_break": torch.from_numpy(is_break),
            "is_ex": torch.from_numpy(is_ex),
            "simultaneous": torch.from_numpy(simultaneous),
            "previous_delta": torch.from_numpy(previous_delta),
            "level": torch.tensor(float(row["level"] or 0.0), dtype=torch.float32),
            "target_delta": torch.from_numpy(target_delta),
            "target_operator": torch.from_numpy(target_operator),
            "target_endpoint": torch.from_numpy(target_endpoint),
            "target_branch": torch.from_numpy(target_branch),
            "target_pattern": torch.from_numpy(target_pattern),
            "mask": torch.from_numpy(mask),
            "lane_loss_mask": torch.from_numpy(lane_loss_mask),
            "window_id": row["window_id"],
        }

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        return self._encode_row(self.rows[index])


class PlanEncoder(nn.Module):
    def __init__(self, dimension: int = 128) -> None:
        super().__init__()
        self.tick_phase = nn.Embedding(192, 32)
        self.measure = nn.Embedding(16, 16)
        self.event_type = nn.Embedding(3, 24)
        self.duration = nn.Embedding(65, 16)
        self.break_note = nn.Embedding(2, 8)
        self.ex_note = nn.Embedding(2, 8)
        self.simultaneous = nn.Embedding(2, 8)
        self.level = nn.Sequential(nn.Linear(1, 16), nn.SiLU(), nn.Linear(16, 16))
        self.projection = nn.Linear(128, dimension)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        tick = batch["tick"]
        level = ((batch["level"] - 10.0) / 5.0).view(-1, 1, 1).expand(-1, tick.shape[1], -1)
        features = torch.cat(
            [
                self.tick_phase(tick % 192),
                self.measure((tick // 192).clamp(0, 15)),
                self.event_type(batch["event_type"]),
                self.duration(batch["duration"]),
                self.break_note(batch["is_break"]),
                self.ex_note(batch["is_ex"]),
                self.simultaneous(batch["simultaneous"]),
                self.level(level),
            ],
            dim=-1,
        )
        return self.projection(features)


class OfficialPatternArranger(nn.Module):
    def __init__(self, dimension: int = 128) -> None:
        super().__init__()
        self.plan = PlanEncoder(dimension)
        self.plan_context = nn.GRU(
            dimension,
            dimension // 2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.previous_delta = nn.Embedding(9, 32)
        self.pattern = nn.Embedding(len(PATTERN_NAMES), dimension)
        self.decoder = nn.GRU(
            dimension + 32,
            dimension,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
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

    def encode_plan(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        plan = self.plan(batch)
        context, _ = self.plan_context(plan)
        return context

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        context = self.encode_plan(batch)
        pattern_logits = self.pattern_head(context)
        pattern_tokens = batch.get("target_pattern", pattern_logits.argmax(dim=-1))
        conditioned = context + self.pattern(pattern_tokens)
        decoder_input = torch.cat([conditioned, self.previous_delta(batch["previous_delta"])], dim=-1)
        hidden, _ = self.decoder(decoder_input)
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
        tokens = torch.zeros(pattern_logits.shape[:2], device=pattern_logits.device, dtype=torch.long)
        heat = torch.tensor(
            (1.0, *pattern_heat), device=pattern_logits.device, dtype=pattern_logits.dtype
        ).clamp_min(0.0)
        lengths = batch.get("valid_length")
        for batch_index in range(pattern_logits.shape[0]):
            length = int(lengths[batch_index]) if lengths is not None else pattern_logits.shape[1]
            ticks = batch["tick"][batch_index]
            event_types = batch["event_type"][batch_index]
            simultaneous = batch["simultaneous"][batch_index]
            position = 0
            while position + 2 < length:
                if int(event_types[position]) != 0 or int(simultaneous[position]) != 0:
                    position += 1
                    continue
                gap = int(ticks[position + 1] - ticks[position])
                if gap not in {6, 8, 12}:
                    position += 1
                    continue
                end = position + 2
                while end < length:
                    if int(event_types[end]) != 0 or int(simultaneous[end]) != 0:
                        break
                    if int(ticks[end] - ticks[end - 1]) != gap:
                        break
                    end += 1
                run_length = end - position
                if run_length >= 3:
                    if gap <= 8 and run_length >= 4:
                        sweep_heat = float(heat[PATTERN_SWEEP])
                        if sweep_heat >= 1.0 or (
                            sweep_heat > 0.0 and float(torch.rand((), device=pattern_logits.device)) < sweep_heat
                        ):
                            tokens[batch_index, position:end] = PATTERN_SWEEP
                        position = max(position + 1, end)
                        continue
                    weights = torch.tensor(
                        PATTERN_TRAINING_WEIGHTS,
                        device=pattern_logits.device,
                        dtype=pattern_logits.dtype,
                    )
                    calibrated = pattern_logits[batch_index, position:end].mean(dim=0) - weights.log()
                    probabilities = calibrated.softmax(dim=0)
                    probabilities = probabilities * heat
                    if run_length < 4:
                        probabilities[PATTERN_INTERACTION] = 0
                        probabilities[PATTERN_SWEEP] = 0
                    configured_without_jack = probabilities[PATTERN_INTERACTION] + probabilities[PATTERN_SWEEP]
                    jack_share = min(0.10, MAX_JACK_PATTERN_SHARE * float(heat[PATTERN_JACK]))
                    probabilities[PATTERN_JACK] = torch.minimum(
                        probabilities[PATTERN_JACK],
                        configured_without_jack
                        * jack_share
                        / max(1e-8, 1.0 - jack_share),
                    )
                    probabilities = probabilities / probabilities.sum().clamp_min(1e-8)
                    chosen = int(torch.multinomial(probabilities, 1))
                    if chosen != PATTERN_NONE:
                        tokens[batch_index, position:end] = chosen
                position = max(position + 1, end)
        return tokens

    @torch.inference_mode()
    def generate(
        self,
        batch: dict[str, torch.Tensor],
        first_lane: int = 0,
        operator_bias: torch.Tensor | None = None,
        enable_patterns: bool = True,
        pattern_heat: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> dict[str, torch.Tensor]:
        context = self.encode_plan(batch)
        pattern_logits = self.pattern_head(context)
        if enable_patterns:
            pattern_tokens = self._segment_pattern_tokens(batch, pattern_logits, pattern_heat)
        else:
            pattern_tokens = torch.zeros(pattern_logits.shape[:2], device=context.device, dtype=torch.long)
        conditioned = context + self.pattern(pattern_tokens)
        batch_size, length, _ = context.shape
        previous_token = torch.full((batch_size, 1), 8, device=context.device, dtype=torch.long)
        hidden_state = None
        deltas = []
        operators = []
        operator_logits_all = []
        endpoints = []
        branches = []
        chosen_patterns = []
        lanes = torch.full((batch_size,), first_lane, device=context.device, dtype=torch.long)
        generated_lanes = [lanes.clone()]
        for position in range(length):
            decoder_input = torch.cat(
                [conditioned[:, position : position + 1], self.previous_delta(previous_token)], dim=-1
            )
            output, hidden_state = self.decoder(decoder_input, hidden_state)
            delta_logits = self.delta_head(output[:, 0])
            delta = delta_logits.argmax(dim=-1)
            if position == 0:
                delta = torch.zeros_like(delta)
            else:
                for batch_index in range(batch_size):
                    pattern = int(pattern_tokens[batch_index, position])
                    previous_pattern = int(pattern_tokens[batch_index, position - 1])
                    if pattern != previous_pattern or pattern == PATTERN_NONE:
                        continue
                    if pattern == PATTERN_INTERACTION:
                        if position >= 2 and int(pattern_tokens[batch_index, position - 2]) == pattern:
                            delta[batch_index] = (-previous_token[batch_index, 0]) % 8
                        else:
                            candidates = torch.tensor([1, 2, 6, 7], device=delta_logits.device)
                            delta[batch_index] = candidates[delta_logits[batch_index, candidates].argmax()]
                    elif pattern == PATTERN_SWEEP:
                        if position >= 2 and int(pattern_tokens[batch_index, position - 2]) == pattern:
                            delta[batch_index] = previous_token[batch_index, 0]
                        else:
                            candidates = torch.tensor([1, 7], device=delta_logits.device)
                            delta[batch_index] = candidates[delta_logits[batch_index, candidates].argmax()]
                    elif pattern == PATTERN_JACK:
                        delta[batch_index] = 0
                lanes = (lanes + delta) % 8
                generated_lanes.append(lanes.clone())
            deltas.append(delta)
            operator_logits = self.operator_head(output[:, 0])
            if operator_bias is not None:
                operator_logits = operator_logits + operator_bias.to(operator_logits.device)
            operator_logits_all.append(operator_logits)
            operators.append(operator_logits.argmax(dim=-1))
            endpoints.append(self.endpoint_head(output[:, 0]).argmax(dim=-1))
            branches.append(self.branch_head(output[:, 0]).argmax(dim=-1))
            chosen_patterns.append(pattern_tokens[:, position])
            previous_token = delta[:, None]
        lane_tensor = torch.stack(generated_lanes[:length], dim=1)
        return {
            "delta": torch.stack(deltas, dim=1),
            "lane": lane_tensor,
            "operator": torch.stack(operators, dim=1),
            "operator_logits": torch.stack(operator_logits_all, dim=1),
            "endpoint": torch.stack(endpoints, dim=1),
            "branch": torch.stack(branches, dim=1),
            "pattern": torch.stack(chosen_patterns, dim=1),
            "pattern_logits": pattern_logits,
        }
