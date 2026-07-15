from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torchaudio

from maimai_ai.diffusion import cosine_schedule
from maimai_ai.arranger import (
    OPERATORS,
    OfficialPatternArranger,
    operator_calibration_bias,
    operator_priors,
)
from maimai_ai.mug_diffusion import MugInspiredAudioChartDiffusion
from maimai_ai.rhythm_model import RhythmPlanModel
from maimai_ai.sampling import ddim_sample
from maimai_ai.vae import ChartVAE
from maimai_ai.patterns import (
    MAX_JACK_PATTERN_SHARE,
    PATTERN_INTERACTION,
    PATTERN_JACK,
    PATTERN_NAMES,
    PATTERN_SWEEP,
    detect_pattern_labels,
)
from prepare_audio_features import TARGET_SAMPLE_RATE, decode_mono


ROOT = Path(r"D:\trans")
ENGINE_NAME = "ORBIT-8"
ENGINE_VERSION = "v1.7.1"
ENGINE_CREATOR = "SeaLandX"
PREPARED = ROOT / "maimai_finale_dataset" / "prepared_v2"
STAR_CATALOG = ROOT / "maimai_finale_dataset" / "prepared_v1" / "star_catalog.json"
AUDIO_CONFIG = ROOT / "maimai_finale_dataset" / "prepared_audio_orbit_v15" / "config.json"
CHECKPOINT = ROOT / "maimai_audio_diffusion" / "runs" / "finale_mug_maimai_v3" / "best.pt"
VAE_CHECKPOINT = ROOT / "maimai_vae" / "runs" / "finale_vae_v1" / "best.pt"
ARRANGER_CHECKPOINT = ROOT / "maimai_arranger" / "runs" / "orbit_v171_arranger" / "best.pt"
RHYTHM_CHECKPOINT = ROOT / "maimai_rhythm" / "runs" / "orbit_v17_rhythm_consensus" / "best.pt"
TICKS_PER_MEASURE = 192
WINDOW_TICKS = 3072
STRIDE_TICKS = 1536
LONG_OBJECT_RELEASE_TICKS = 12
SLIDE_TAIL_CLEARANCE_TICKS = 32
DEFAULT_SLIDE_RATIO_BOOST = 1.10
EIGHTH_SLIDE_SCORE_BOOST = 1.15
HAND_FLOW_OPTIMIZER = None


@dataclass
class GeneratedEvent:
    tick: int
    lane: int
    kind: str
    score: float
    duration: int = 0
    is_break: bool = False
    is_ex: bool = False
    operator_id: int = 0
    endpoint: int = 4
    branch: int = 0
    pattern_type: int = 0
    slide_template: dict | None = None
    slide_was_slowed: bool = False
    slide_max_speed: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ORBIT-8 v1.7.1 neural maimai chart engine by SeaLandX")
    parser.add_argument("audio", type=Path)
    parser.add_argument("--bpm", type=float, required=True)
    parser.add_argument("--offset", type=float, default=0.0, help="Seconds from audio start to chart tick zero")
    parser.add_argument("--level", type=float, required=True)
    parser.add_argument("--title")
    parser.add_argument("--artist", default="Unknown")
    parser.add_argument("--designer", default="SeaLandX feat. ORBIT-8 v1.7.1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=1.25)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--density", type=float)
    parser.add_argument("--hold-ratio", type=float)
    parser.add_argument("--slide-ratio", type=float)
    parser.add_argument("--break-ratio", type=float)
    parser.add_argument("--interaction-heat", type=float, default=1.0)
    parser.add_argument("--sweep-heat", type=float, default=0.7)
    parser.add_argument("--jack-heat", type=float, default=1.0)
    parser.add_argument("--timing-model", choices=("rhythm", "diffusion"), default="rhythm")
    return parser.parse_args()


def control_defaults(level: float) -> np.ndarray:
    payload = json.loads((PREPARED / "control_defaults.json").read_text(encoding="utf-8"))
    rows = payload["levels"]
    levels = np.asarray([row["level"] for row in rows], dtype=np.float32)
    keys = ("density_per_measure", "hold_ratio", "slide_ratio", "break_ratio")
    values = [float(np.interp(level, levels, [row[key] for row in rows])) for key in keys]
    return np.asarray([level, *values], dtype=np.float32)


def window_starts(total_ticks: int) -> list[int]:
    if total_ticks <= WINDOW_TICKS:
        return [0]
    starts = list(range(0, total_ticks - WINDOW_TICKS + 1, STRIDE_TICKS))
    final_start = math.ceil((total_ticks - WINDOW_TICKS) / TICKS_PER_MEASURE) * TICKS_PER_MEASURE
    if final_start not in starts:
        starts.append(final_start)
    return starts


def extract_log_mel(audio_path: Path, config: dict) -> tuple[np.ndarray, float]:
    waveform, _ = decode_mono(audio_path)
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=int(config["n_fft"]),
        win_length=int(config["n_fft"]),
        hop_length=int(config["hop_length"]),
        f_min=30.0,
        f_max=TARGET_SAMPLE_RATE / 2,
        n_mels=int(config["n_mels"]),
        power=2.0,
        center=True,
    ).cuda()
    with torch.inference_mode():
        mel = transform(waveform.cuda()).squeeze(0)
        log_mel = torch.log(mel.clamp_min(1e-5)).cpu().numpy()
    mean = np.asarray(config["train_mean"], dtype=np.float32)[:, None]
    std = np.asarray(config["train_std"], dtype=np.float32)[:, None]
    return ((log_mel - mean) / std).astype(np.float32), waveform.shape[-1] / TARGET_SAMPLE_RATE


def aligned_audio(log_mel: np.ndarray, tick_times_ms: np.ndarray, config: dict) -> np.ndarray:
    center_ms = (tick_times_ms[:-1] + tick_times_ms[1:]) * 0.5
    positions = center_ms * TARGET_SAMPLE_RATE / (1000.0 * int(config["hop_length"]))
    left = np.floor(positions).astype(np.int64)
    left = np.clip(left, 0, log_mel.shape[1] - 1)
    right = np.minimum(left + 1, log_mel.shape[1] - 1)
    alpha = (positions - left).astype(np.float32)[None]
    features = log_mel[:, left] * (1.0 - alpha) + log_mel[:, right] * alpha
    tick = np.arange(WINDOW_TICKS, dtype=np.float32)
    timing = np.stack(
        [
            np.sin(2 * np.pi * tick / 192.0),
            np.cos(2 * np.pi * tick / 192.0),
            np.sin(2 * np.pi * tick / 768.0),
            np.cos(2 * np.pi * tick / 768.0),
        ]
    )
    return np.concatenate([features, timing]).astype(np.float32)


def select_candidates(
    probabilities: np.ndarray,
    rhythm: np.ndarray,
    channel_start: int,
    rhythm_channel: int,
    count: int,
    total_ticks: int,
    kind: str,
) -> list[GeneratedEvent]:
    if count <= 0:
        return []
    values = probabilities[channel_start : channel_start + 8, :total_ticks].copy()
    rhythm_by_tick = rhythm[rhythm_channel, np.minimum(np.arange(total_ticks) // 8, rhythm.shape[1] - 1)]
    values *= 0.15 + 0.85 * rhythm_by_tick[None]
    order = np.argsort(values.reshape(-1))[::-1]
    selected: list[GeneratedEvent] = []
    lane_ticks: list[list[int]] = [[] for _ in range(8)]
    occupied: dict[int, int] = {}
    for flat in order:
        lane, tick = np.unravel_index(int(flat), values.shape)
        tick = int(round(tick / 2.0) * 2)
        if tick >= total_ticks or occupied.get(tick, 0) >= 2:
            continue
        if any(abs(tick - other) < 4 for other in lane_ticks[lane]):
            continue
        score = float(values[lane, min(tick, values.shape[1] - 1)])
        selected.append(GeneratedEvent(tick=tick, lane=lane + 1, kind=kind, score=score))
        lane_ticks[lane].append(tick)
        occupied[tick] = occupied.get(tick, 0) + 1
        if len(selected) >= count:
            break
    return selected


def hold_duration(probabilities: np.ndarray, event: GeneratedEvent, total_ticks: int) -> int:
    active = probabilities[24 + event.lane - 1]
    end_limit = min(total_ticks, event.tick + 384)
    below = 0
    end = event.tick + 24
    for tick in range(event.tick + 2, end_limit, 2):
        if active[tick] >= 0.45:
            below = 0
            end = tick
        else:
            below += 1
            if below >= 3 and tick >= event.tick + 24:
                break
    return max(24, min(384, end - event.tick))


def rhythm_hold_duration(event_probabilities: np.ndarray, tick: int, total_ticks: int) -> int:
    active = event_probabilities[3]
    end_limit = min(total_ticks, tick + 384)
    below = 0
    end = tick + 24
    for other in range(tick + 2, end_limit, 2):
        if active[other] >= 0.20:
            below = 0
            end = other
        else:
            below += 1
            if below >= 3 and other >= tick + 24:
                break
    return max(24, min(384, end - tick))


def rhythm_plan_events(
    event_probabilities: np.ndarray,
    count_probabilities: np.ndarray,
    onset_probabilities: np.ndarray | None,
    controls: np.ndarray,
    total_ticks: int,
    subdivision_probabilities: np.ndarray | None = None,
    accent_probabilities: np.ndarray | None = None,
) -> list[GeneratedEvent]:
    measures = total_ticks / TICKS_PER_MEASURE
    encoded_total = max(1, round(float(controls[1]) * measures))
    break_count = round(encoded_total * float(controls[4]))
    identity_count = max(1, encoded_total - break_count)
    quotas = {
        "hold": min(identity_count, round(encoded_total * float(controls[2]))),
        "slide": min(identity_count, round(encoded_total * float(controls[3]))),
    }
    quotas["tap"] = max(0, identity_count - quotas["hold"] - quotas["slide"])

    nonzero = 1.0 - count_probabilities[0, :total_ticks]
    onset = onset_probabilities[:total_ticks] if onset_probabilities is not None else np.ones(total_ticks)
    onset_weight = 0.08 + 0.92 * np.square(onset)
    accent = (
        accent_probabilities[:total_ticks]
        if accent_probabilities is not None
        else np.zeros(total_ticks, dtype=np.float32)
    )
    if subdivision_probabilities is not None:
        metrical_presence = 1.0 - subdivision_probabilities[0, :total_ticks]
        onset_weight *= 0.85 + 0.30 * metrical_presence
    scores = {
        "tap": np.maximum(event_probabilities[0, :total_ticks], event_probabilities[1, :total_ticks]) * nonzero,
        "hold": event_probabilities[2, :total_ticks] * nonzero,
        "slide": event_probabilities[4, :total_ticks] * nonzero,
    }
    scores = {kind: values * onset_weight for kind, values in scores.items()}

    grid_ticks = sorted(
        set(range(0, total_ticks, 12))
        | set(range(0, total_ticks, 8))
        | set(range(0, total_ticks, 6))
    )
    allowed_fast_pairs: set[tuple[int, int]] = set()
    if onset_probabilities is not None:
        tap_threshold = float(np.quantile(scores["tap"], 0.75))
        for gap in (6, 8):
            subdivision_class = 6 if gap == 6 else 5
            eligible = []
            for tick in range(0, total_ticks, gap):
                left = max(0, tick - 2)
                right = min(total_ticks, tick + 3)
                structured_fast = (
                    subdivision_probabilities is not None
                    and subdivision_probabilities[subdivision_class, tick] >= 0.30
                )
                eligible.append(
                    onset[tick] >= (0.50 if structured_fast else 0.65)
                    and onset[tick] + 1e-8 >= float(onset[left:right].max())
                    and scores["tap"][tick] >= tap_threshold
                )
            start = 0
            while start < len(eligible):
                if not eligible[start]:
                    start += 1
                    continue
                end = start + 1
                while end < len(eligible) and eligible[end]:
                    end += 1
                if end - start >= 4:
                    run = [index * gap for index in range(start, end)]
                    allowed_fast_pairs.update(zip(run, run[1:]))
                start = end

    occupancy: dict[int, int] = {}
    selected: list[GeneratedEvent] = []
    for kind in ("hold", "slide", "tap"):
        candidate_scores = []
        for tick in grid_ticks:
            left = max(0, tick - 2)
            right = min(total_ticks, tick + 3)
            score = float(scores[kind][left:right].max())
            if tick % 12 == 0:
                grid_penalty = 1.0
            elif tick % 8 == 0:
                grid_penalty = 0.97
            else:
                grid_penalty = 0.92
            local_left = max(0, tick - 4)
            local_right = min(total_ticks, tick + 5)
            is_local_peak = score + 1e-8 >= float(scores[kind][local_left:local_right].max())
            peak_bonus = 1.08 if is_local_peak else 1.0
            if kind == "slide" and tick % 24 == 0:
                peak_bonus *= EIGHTH_SLIDE_SCORE_BOOST
            if kind == "tap" and any(tick in pair for pair in allowed_fast_pairs):
                peak_bonus *= 1.08
            structure_bonus = 1.0
            if subdivision_probabilities is not None:
                structure_bonus += float(
                    0.15 * subdivision_probabilities[3, tick]
                    + 1.20 * subdivision_probabilities[4, tick]
                    + 0.45 * subdivision_probabilities[5, tick]
                    + 0.30 * subdivision_probabilities[6, tick]
                )
            accent_bonus = 1.0 + 0.10 * float(accent[tick])
            candidate_scores.append(
                (score * grid_penalty * peak_bonus * structure_bonus * accent_bonus, tick)
            )
        order = [tick for _, tick in sorted(candidate_scores, reverse=True)]
        candidate_score_by_tick = {tick: score for score, tick in candidate_scores}
        accepted = 0
        for tick in order:
            near_ticks = [other for other in occupancy if 0 < abs(other - tick) < 12]
            if near_ticks and not (
                kind == "tap"
                and all((min(other, tick), max(other, tick)) in allowed_fast_pairs for other in near_ticks)
            ):
                continue
            double_preferred = (
                count_probabilities[2, tick] >= count_probabilities[1, tick] * 0.75
                or (
                    accent_probabilities is not None
                    and accent[tick] >= 0.85
                    and count_probabilities[2, tick] >= count_probabilities[1, tick] * 0.55
                )
            )
            capacity = 2 if double_preferred else 1
            if occupancy.get(tick, 0) >= capacity:
                continue
            selected.append(
                GeneratedEvent(tick=tick, lane=1, kind=kind, score=float(candidate_score_by_tick[tick]))
            )
            occupancy[tick] = occupancy.get(tick, 0) + 1
            accepted += 1
            if accepted >= quotas[kind]:
                break

    selected.sort(key=lambda event: (event.tick, -event.score))
    playable: list[GeneratedEvent] = []
    active_holds: list[int] = []
    for event in selected:
        active_holds = [end for end in active_holds if end > event.tick]
        same_tick = sum(other.tick == event.tick for other in playable[-2:])
        if len(active_holds) + same_tick >= 2:
            continue
        if event.kind == "hold":
            event.duration = rhythm_hold_duration(event_probabilities, event.tick, total_ticks)
            active_holds.append(event.tick + event.duration)
        playable.append(event)

    break_candidates = sorted(
        playable,
        key=lambda event: float(event_probabilities[1, event.tick]) * (1.0 + float(accent[event.tick])),
        reverse=True,
    )
    for event in break_candidates[:break_count]:
        event.is_break = True
    return playable


def clean_irregular_fast_notes(events: list[GeneratedEvent]) -> list[GeneratedEvent]:
    result = list(events)
    while True:
        groups: dict[int, list[GeneratedEvent]] = {}
        for event in result:
            groups.setdefault(event.tick, []).append(event)
        ticks = sorted(groups)
        allowed_pairs: set[tuple[int, int]] = set()
        for gap in (6, 8):
            start = 0
            while start < len(ticks):
                end = start + 1
                while end < len(ticks) and ticks[end] - ticks[end - 1] == gap:
                    end += 1
                run = ticks[start:end]
                if len(run) >= 4 and all(
                    len(groups[tick]) == 1 and groups[tick][0].kind == "tap" for tick in run
                ):
                    allowed_pairs.update(zip(run, run[1:]))
                start = max(start + 1, end)

        offending = next(
            (
                (left, right)
                for left, right in zip(ticks, ticks[1:])
                if right - left < 12 and (left, right) not in allowed_pairs
            ),
            None,
        )
        if offending is None:
            return sorted(result, key=lambda event: (event.tick, -event.score))
        left, right = offending
        left_score = max(event.score for event in groups[left])
        right_score = max(event.score for event in groups[right])
        removed_tick = left if left_score < right_score else right
        result = [event for event in result if event.tick != removed_tick]


def generated_pattern_summary(events: list[GeneratedEvent]) -> tuple[dict[str, int], dict[str, int], float]:
    ordered = sorted(events, key=lambda event: (event.tick, event.lane, event.kind))
    payload = [
        {"tick": event.tick, "lane": event.lane, "note_type": event.kind, "branches": []}
        for event in ordered
    ]
    labels = detect_pattern_labels(payload)
    event_counts = Counter(PATTERN_NAMES[label] for label in labels)
    segment_counts = Counter()
    previous = 0
    for label in labels:
        if label and label != previous:
            segment_counts[PATTERN_NAMES[label]] += 1
        previous = label
    configured = sum(segment_counts.values())
    jack_share = segment_counts["jack"] / max(1, configured)
    return dict(event_counts), dict(segment_counts), jack_share


def limit_jack_patterns(
    events: list[GeneratedEvent], *, remove_excess: bool, max_share: float = MAX_JACK_PATTERN_SHARE
) -> tuple[list[GeneratedEvent], int]:
    result = sorted(events, key=lambda event: (event.tick, event.lane, event.kind))
    changes = 0
    for _ in range(256):
        payload = [
            {"tick": event.tick, "lane": event.lane, "note_type": event.kind, "branches": []}
            for event in result
        ]
        labels = detect_pattern_labels(payload)
        segments = []
        start = 0
        while start < len(labels):
            label = labels[start]
            end = start + 1
            while end < len(labels) and labels[end] == label:
                end += 1
            if label:
                segments.append((label, start, end))
            start = end
        jack_segments = [segment for segment in segments if segment[0] == PATTERN_JACK]
        non_jack_segments = len(segments) - len(jack_segments)
        allowed = math.floor(
            non_jack_segments * max_share / max(1e-8, 1.0 - max_share)
        )
        if len(jack_segments) <= allowed:
            break
        _, segment_start, segment_end = min(
            jack_segments,
            key=lambda segment: sum(result[index].score for index in range(segment[1], segment[2]))
            / (segment[2] - segment[1]),
        )
        indices = list(range(segment_start, segment_end))
        pivot = min(indices, key=lambda index: result[index].score)
        if remove_excess:
            result.pop(pivot)
        else:
            event = result[pivot]
            occupied = {item.lane for item in result if item.tick == event.tick and item is not event}
            candidates = [
                ((event.lane - 2) % 8) + 1,
                (event.lane % 8) + 1,
            ]
            available = [lane for lane in candidates if lane not in occupied]
            if available:
                event.lane = available[0]
            else:
                result.pop(pivot)
        changes += 1
        result.sort(key=lambda event: (event.tick, event.lane, event.kind))
    return result, changes


def max_jack_share_for_bpm(bpm: float, jack_heat: float) -> float:
    if bpm > 200.0:
        return 0.0
    return min(0.10, MAX_JACK_PATTERN_SHARE * jack_heat)


def long_eighth_jack_excess(events: list[GeneratedEvent], max_repeats: int = 6) -> int:
    excess = 0
    for lane in range(1, 9):
        lane_events = sorted(
            (event for event in events if event.kind == "tap" and event.lane == lane),
            key=lambda event: event.tick,
        )
        start = 0
        while start < len(lane_events):
            end = start + 1
            while end < len(lane_events) and lane_events[end].tick - lane_events[end - 1].tick == 24:
                end += 1
            excess += max(0, end - start - max_repeats)
            start = end
    return excess


def break_long_eighth_jacks(
    events: list[GeneratedEvent], max_repeats: int = 6
) -> tuple[list[GeneratedEvent], int]:
    result = sorted(events, key=lambda event: (event.tick, event.lane, event.kind))
    changes = 0
    for lane in range(1, 9):
        lane_events = sorted(
            (event for event in result if event.kind == "tap" and event.lane == lane),
            key=lambda event: event.tick,
        )
        start = 0
        while start < len(lane_events):
            end = start + 1
            while end < len(lane_events) and lane_events[end].tick - lane_events[end - 1].tick == 24:
                end += 1
            run = lane_events[start:end]
            if len(run) > max_repeats:
                neighbor_order = (
                    (lane % 8) + 1,
                    ((lane - 2) % 8) + 1,
                    ((lane + 1) % 8) + 1,
                    ((lane - 3) % 8) + 1,
                )
                for index, event in enumerate(run[max_repeats:]):
                    if index % 2:
                        continue
                    occupied = {
                        other.lane for other in result
                        if other is not event and other.tick == event.tick
                    }
                    target = next((candidate for candidate in neighbor_order if candidate not in occupied), None)
                    if target is not None:
                        event.lane = target
                        changes += 1
            start = end
    result.sort(key=lambda event: (event.tick, event.lane, event.kind))
    return result, changes


def break_irregular_sixteenth_runs(
    events: list[GeneratedEvent], bpm: float
) -> tuple[list[GeneratedEvent], int]:
    result = sorted(events, key=lambda event: (event.tick, event.lane, event.kind))
    removed = 0
    while True:
        payload = [
            {"tick": event.tick, "lane": event.lane, "note_type": event.kind, "branches": []}
            for event in result
        ]
        labels = detect_pattern_labels(payload)
        groups: dict[int, list[int]] = {}
        for index, event in enumerate(result):
            groups.setdefault(event.tick, []).append(index)
        ticks = sorted(groups)
        offending: list[int] | None = None
        start = 0
        while start < len(ticks):
            end = start + 1
            while end < len(ticks) and ticks[end] - ticks[end - 1] == 12:
                end += 1
            run_ticks = ticks[start:end]
            if len(run_ticks) >= 3 and all(
                len(groups[tick]) == 1 and result[groups[tick][0]].kind == "tap"
                for tick in run_ticks
            ):
                indices = [groups[tick][0] for tick in run_ticks]
                run_labels = {labels[index] for index in indices}
                legal = run_labels in ({PATTERN_INTERACTION}, {PATTERN_SWEEP})
                if bpm <= 200.0 and run_labels == {PATTERN_JACK}:
                    legal = True
                if not legal:
                    offending = indices
                    break
            start = end
        if offending is None:
            break
        candidates = offending[1:-1] or offending
        remove_index = min(candidates, key=lambda index: result[index].score)
        result.pop(remove_index)
        removed += 1
    return result, removed


def legalize_events(
    probabilities: np.ndarray,
    rhythm: np.ndarray,
    controls: np.ndarray,
    total_ticks: int,
) -> list[GeneratedEvent]:
    measures = total_ticks / TICKS_PER_MEASURE
    encoded_total = max(1, round(float(controls[1]) * measures))
    break_count = round(encoded_total * float(controls[4]))
    identity_count = max(1, encoded_total - break_count)
    hold_count = min(identity_count, round(encoded_total * float(controls[2])))
    slide_count = min(identity_count - hold_count, round(encoded_total * float(controls[3])))
    tap_count = max(0, identity_count - hold_count - slide_count)

    events = []
    events += select_candidates(probabilities, rhythm, 0, 0, tap_count, total_ticks, "tap")
    events += select_candidates(probabilities, rhythm, 16, 2, hold_count, total_ticks, "hold")
    events += select_candidates(probabilities, rhythm, 32, 4, slide_count, total_ticks, "slide")
    for event in events:
        if event.kind == "hold":
            event.duration = hold_duration(probabilities, event, total_ticks)

    # Resolve collisions between independently selected note types.
    accepted: list[GeneratedEvent] = []
    active_holds: list[int] = []
    for tick in sorted({event.tick for event in events}):
        active_holds = [end for end in active_holds if end > tick]
        available = max(0, 2 - len(active_holds))
        choices = sorted((event for event in events if event.tick == tick), key=lambda item: item.score, reverse=True)
        used_lanes = set()
        for event in choices:
            if available <= 0 or event.lane in used_lanes:
                continue
            accepted.append(event)
            used_lanes.add(event.lane)
            available -= 1
            if event.kind == "hold":
                active_holds.append(event.tick + event.duration)

    break_scores = sorted(
        accepted,
        key=lambda event: float(probabilities[8 + event.lane - 1, event.tick]),
        reverse=True,
    )
    for event in break_scores[:break_count]:
        event.is_break = True
    for event in accepted:
        event.is_ex = probabilities[40 + event.lane - 1, event.tick] >= 0.7
    return sorted(accepted, key=lambda event: (event.tick, event.lane))


def arranger_plan(events: list[GeneratedEvent], start_tick: int, level: float) -> dict[str, torch.Tensor]:
    selected = [event for event in events if start_tick <= event.tick < start_tick + WINDOW_TICKS][:384]
    length = 384
    tick = torch.zeros((1, length), device="cuda", dtype=torch.long)
    event_type = torch.zeros_like(tick)
    duration = torch.zeros_like(tick)
    is_break = torch.zeros_like(tick)
    is_ex = torch.zeros_like(tick)
    simultaneous = torch.zeros_like(tick)
    previous_tick = None
    same_tick = 0
    type_ids = {"tap": 0, "hold": 1, "slide": 2}
    for position, event in enumerate(selected):
        tick[0, position] = event.tick - start_tick
        event_type[0, position] = type_ids[event.kind]
        duration[0, position] = min(64, event.duration // 6)
        is_break[0, position] = int(event.is_break)
        is_ex[0, position] = int(event.is_ex)
        if event.tick == previous_tick:
            same_tick += 1
        else:
            same_tick = 0
        simultaneous[0, position] = min(1, same_tick)
        previous_tick = event.tick
    return {
        "tick": tick,
        "event_type": event_type,
        "duration": duration,
        "is_break": is_break,
        "is_ex": is_ex,
        "simultaneous": simultaneous,
        "level": torch.tensor([level], device="cuda", dtype=torch.float32),
        "valid_length": torch.tensor([len(selected)], device="cuda", dtype=torch.long),
        "selected": selected,
    }


@torch.inference_mode()
def arrange_patterns(
    events: list[GeneratedEvent],
    total_ticks: int,
    level: float,
    seed: int,
    pattern_heat: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> list[GeneratedEvent]:
    if not events:
        return events
    model = OfficialPatternArranger().cuda().eval()
    arranger_state = torch.load(ARRANGER_CHECKPOINT, map_location="cuda", weights_only=False)["model"]
    pattern_enabled = "pattern_head.weight" in arranger_state
    model.load_state_dict(arranger_state, strict=False)
    operator_bias = operator_calibration_bias(PREPARED).cuda()
    identity = {id(event): index for index, event in enumerate(events)}
    assigned: dict[int, dict[str, int]] = {}
    for start in window_starts(total_ticks):
        batch = arranger_plan(events, start, level)
        selected = batch.pop("selected")
        if not selected:
            continue
        generated = model.generate(
            batch,
            first_lane=seed % 8,
            operator_bias=operator_bias,
            enable_patterns=pattern_enabled,
            pattern_heat=pattern_heat,
        )
        overlap = [identity[id(event)] for event in selected if identity[id(event)] in assigned]
        if overlap:
            rotation = min(
                range(8),
                key=lambda value: sum(
                    min(
                        abs((int(generated["lane"][0, position]) + value) % 8 - assigned[identity[id(event)]]["lane"]),
                        8 - abs((int(generated["lane"][0, position]) + value) % 8 - assigned[identity[id(event)]]["lane"]),
                    )
                    for position, event in enumerate(selected)
                    if identity[id(event)] in assigned
                ),
            )
        else:
            rotation = 0
        for position, event in enumerate(selected):
            event_index = identity[id(event)]
            if event_index in assigned:
                continue
            assigned[event_index] = {
                "lane": (int(generated["lane"][0, position]) + rotation) % 8,
                "operator": int(generated["operator"][0, position]),
                "endpoint": int(generated["endpoint"][0, position]),
                "branch": int(generated["branch"][0, position]),
                "pattern": int(generated["pattern"][0, position]),
                "operator_logits": generated["operator_logits"][0, position].float().cpu(),
            }
    for index, event in enumerate(events):
        if index not in assigned:
            continue
        event.lane = assigned[index]["lane"] + 1
        event.operator_id = assigned[index]["operator"]
        event.endpoint = assigned[index]["endpoint"]
        event.branch = assigned[index]["branch"]
        event.pattern_type = assigned[index]["pattern"]

    slide_indices = [index for index, event in enumerate(events) if event.kind == "slide" and index in assigned]
    if slide_indices:
        priors = operator_priors(PREPARED, level)
        expected = priors * len(slide_indices)
        quotas = torch.floor(expected).long()
        remainder = len(slide_indices) - int(quotas.sum())
        if remainder:
            fractional = expected - quotas.float()
            for operator_id in torch.argsort(fractional, descending=True)[:remainder].tolist():
                quotas[operator_id] += 1
        pairs = []
        for event_index in slide_indices:
            logits = assigned[event_index]["operator_logits"]
            for operator_id, score in enumerate(logits.tolist()):
                pairs.append((float(score), event_index, operator_id))
        pairs.sort(reverse=True)
        chosen: dict[int, int] = {}
        for _, event_index, operator_id in pairs:
            if event_index in chosen or quotas[operator_id] <= 0:
                continue
            chosen[event_index] = operator_id
            quotas[operator_id] -= 1
        for event_index in slide_indices:
            if event_index not in chosen:
                available = torch.where(quotas > 0)[0]
                chosen[event_index] = int(available[0]) if len(available) else assigned[event_index]["operator"]
                if len(available):
                    quotas[available[0]] -= 1
            events[event_index].operator_id = chosen[event_index]

    by_tick: dict[int, list[GeneratedEvent]] = {}
    for event in events:
        by_tick.setdefault(event.tick, []).append(event)
    for simultaneous_events in by_tick.values():
        used = set()
        for event in simultaneous_events:
            lane = event.lane
            while lane in used:
                lane = ((lane - 1 + 4) % 8) + 1
                if lane in used:
                    lane = lane % 8 + 1
            event.lane = lane
            used.add(lane)
    return events


def break_eighth_note_orbits(events: list[GeneratedEvent]) -> tuple[list[GeneratedEvent], int]:
    """Break four-note 8th-grid runs that walk around the ring in one direction."""
    result = sorted(events, key=lambda event: (event.tick, event.lane, event.kind))
    by_tick: dict[int, list[GeneratedEvent]] = {}
    for event in result:
        by_tick.setdefault(event.tick, []).append(event)
    singles = [group[0] for _, group in sorted(by_tick.items()) if len(group) == 1]
    changes = 0
    for index in range(3, len(singles)):
        window = singles[index - 3:index + 1]
        if any(right.tick - left.tick != 24 for left, right in zip(window, window[1:])):
            continue
        deltas = [
            (right.lane - left.lane) % 8 for left, right in zip(window, window[1:])
        ]
        if len(set(deltas)) != 1 or deltas[0] not in {1, 7}:
            continue
        direction = 1 if deltas[0] == 1 else -1
        current = window[-1]
        current.lane = ((window[-2].lane - 1 + 3 * direction) % 8) + 1
        changes += 1
    return sorted(result, key=lambda event: (event.tick, event.lane, event.kind)), changes


def choose_slide_template(
    event: GeneratedEvent,
    events: list[GeneratedEvent],
    catalog: list[dict],
    previous_operator: str | None,
    level: float,
) -> dict | None:
    following = next((item for item in events if item.tick > event.tick and item.tick - event.tick <= 192), None)
    desired_operator = OPERATORS[event.operator_id]
    tiers = [
        [item for item in catalog if item["branch_count"] == event.branch + 1 and item["operators"][0] == desired_operator and item["relative_paths"][0][-1] == event.endpoint],
        [item for item in catalog if item["operators"][0] == desired_operator and item["relative_paths"][0][-1] == event.endpoint],
        [item for item in catalog if item["operators"][0] == desired_operator],
    ]
    options = next((tier for tier in tiers if tier), [])
    if not options:
        return None
    def template_score(template: dict) -> float:
        endpoint = ((event.lane - 1 + int(template["relative_paths"][0][-1])) % 8) + 1
        context_distance = 0
        if following:
            raw = abs(endpoint - following.lane)
            context_distance = min(raw, 8 - raw)
        repeat_penalty = 0.4 if template["operators"][0] == previous_operator else 0.0
        return (
            math.log1p(3 * template.get("levels", {}).get(f"{level:.1f}", 0) + template.get("count_12_15", 0))
            - 0.35 * context_distance
            - repeat_penalty
        )
    return max(options, key=template_score)


def assign_slide_templates(events: list[GeneratedEvent], catalog: list[dict], level: float) -> None:
    previous_operator = None
    for event in events:
        if event.kind != "slide":
            continue
        template = choose_slide_template(event, events, catalog, previous_operator, level)
        event.slide_template = copy.deepcopy(template) if template else None
        if event.slide_template:
            previous_operator = event.slide_template["operators"][0]


def _lane_point(lane: int, radius: float = 1.0) -> tuple[float, float]:
    angle = math.pi * 3 / 8 - (lane - 1) * math.pi / 4
    return radius * math.cos(angle), radius * math.sin(angle)


def _polyline_lanes(points: list[tuple[float, float]]) -> set[int]:
    lanes = set()
    for start, end in zip(points, points[1:]):
        for position in np.linspace(0.0, 1.0, 65):
            x = start[0] * (1.0 - position) + end[0] * position
            y = start[1] * (1.0 - position) + end[1] * position
            if math.hypot(x, y) < 0.24:
                continue
            angle = math.atan2(y, x)
            lane = int(round((math.pi * 3 / 8 - angle) / (math.pi / 4))) % 8 + 1
            lanes.add(lane)
    return lanes


def _ring_lanes(start: int, end: int, step: int) -> set[int]:
    result = {start}
    lane = start
    for _ in range(8):
        if lane == end:
            break
        lane = ((lane - 1 + step) % 8) + 1
        result.add(lane)
    return result


def _polyline_length(points: list[tuple[float, float]]) -> float:
    return sum(math.dist(start, end) for start, end in zip(points, points[1:]))


def _ring_steps(start: int, end: int, step: int) -> int:
    lane = start
    for count in range(9):
        if lane == end:
            return count
        lane = ((lane - 1 + step) % 8) + 1
    return 8


def _edge_slide_step(start_lane: int, operator: str) -> int:
    """Resolve simai < / > from its screen-left/screen-right convention."""
    upper_half = start_lane in {7, 8, 1, 2}
    right_step = 1 if upper_half else -1
    return right_step if operator == ">" else -right_step


def _curved_slide_side_lanes(start_lane: int, endpoint: int) -> set[int]:
    """Reserve the outer half occupied by a grand p/q slide for the sliding hand."""
    start = _lane_point(start_lane)
    end = _lane_point(endpoint)
    anchor = (start[0] + end[0], start[1] + end[1])
    if math.hypot(*anchor) < 0.35:
        return _polyline_lanes([start, (0.0, 0.0), end]) | {start_lane, endpoint}
    return {
        lane
        for lane in range(1, 9)
        if _lane_point(lane)[0] * anchor[0] + _lane_point(lane)[1] * anchor[1] >= 0
    } | {start_lane, endpoint}


def slide_branch_length(start_lane: int, operator: str, absolute: list[int]) -> float:
    if not absolute:
        return 0.0
    endpoint = absolute[-1]
    if operator == "-":
        return _polyline_length([_lane_point(start_lane), _lane_point(endpoint)])
    if operator == "v":
        return 2.0
    if operator == "V" and len(absolute) >= 2:
        return _polyline_length([
            _lane_point(start_lane), _lane_point(absolute[-2]), _lane_point(endpoint)
        ])
    if operator in {"s", "z"}:
        direction = 1 if operator == "s" else -1
        middle_start = ((start_lane - 1 + 2 * direction) % 8) + 1
        middle_end = ((endpoint - 1 + 2 * direction) % 8) + 1
        return _polyline_length([
            _lane_point(start_lane),
            _lane_point(middle_start, 0.32),
            _lane_point(middle_end, 0.32),
            _lane_point(endpoint),
        ])
    if operator == "w":
        fan_ends = (((endpoint - 2) % 8) + 1, endpoint, (endpoint % 8) + 1)
        return max(
            _polyline_length([_lane_point(start_lane), _lane_point(fan_end)])
            for fan_end in fan_ends
        )
    if operator in {"p", "pp", "q", "qq"}:
        step = -1 if operator in {"p", "pp"} else 1
        steps = _ring_steps(start_lane, endpoint, step)
        radius = 0.65 if operator in {"p", "q"} else 0.45
        connectors = 0.7 if operator in {"p", "q"} else 1.1
        return connectors + steps * math.pi * radius / 4.0
    if operator in {"<", ">"}:
        step = _edge_slide_step(start_lane, operator)
        return _ring_steps(start_lane, endpoint, step) * math.pi / 4.0
    return _polyline_length([_lane_point(start_lane), _lane_point(endpoint)])


def learn_slide_speed_limits(catalog: list[dict]) -> dict[str, float]:
    samples: dict[str, list[float]] = {operator: [] for operator in OPERATORS}
    all_speeds = []
    for template in catalog:
        for operator, relative_path, duration in zip(
            template["operators"], template["relative_paths"], template["durations"]
        ):
            movement = int(duration.get("duration_ticks") or 0)
            if movement <= 0:
                continue
            absolute = [((int(relative) % 8) + 1) for relative in relative_path]
            length = slide_branch_length(1, operator, absolute)
            if length <= 0:
                continue
            speed = length / movement
            samples.setdefault(operator, []).append(speed)
            all_speeds.append(speed)
    fallback = float(np.quantile(all_speeds, 0.75)) if all_speeds else 0.04
    return {
        operator: float(np.quantile(values, 0.75)) if len(values) >= 5 else fallback
        for operator, values in samples.items()
    }


def normalize_slide_speeds(events: list[GeneratedEvent], catalog: list[dict]) -> None:
    limits = learn_slide_speed_limits(catalog)
    for event in events:
        if event.kind != "slide" or not event.slide_template:
            continue
        final_speeds = []
        for operator, relative_path, duration in zip(
            event.slide_template["operators"],
            event.slide_template["relative_paths"],
            event.slide_template["durations"],
        ):
            absolute = [((event.lane - 1 + int(relative)) % 8) + 1 for relative in relative_path]
            length = slide_branch_length(event.lane, operator, absolute)
            original = max(1, int(duration.get("duration_ticks") or 1))
            long_path_factor = 1.0 + 0.12 * max(0.0, length - 2.0)
            speed_limit = max(1e-6, limits.get(operator, 0.04) / long_path_factor)
            required = max(original, math.ceil(length / speed_limit))
            required = max(6, math.ceil(required / 6) * 6)
            if required > original:
                duration["duration_ticks"] = required
                duration["raw"] = f"192:{required}"
                duration["mode"] = "ratio"
                duration["duration_seconds"] = None
                event.slide_was_slowed = True
            final_speeds.append(length / required)
        event.slide_max_speed = max(final_speeds, default=0.0)


def slide_corridor_lanes(event: GeneratedEvent) -> tuple[set[int], int]:
    template = event.slide_template
    if not template:
        return {event.lane}, event.lane
    corridor = {event.lane}
    final_endpoint = event.lane
    for operator, relative_path in zip(template["operators"], template["relative_paths"]):
        absolute = [((event.lane - 1 + int(relative)) % 8) + 1 for relative in relative_path]
        if not absolute:
            continue
        endpoint = absolute[-1]
        final_endpoint = endpoint
        if operator == "-":
            corridor.update(_polyline_lanes([_lane_point(event.lane), _lane_point(endpoint)]))
        elif operator == "v":
            corridor.update(_polyline_lanes([_lane_point(event.lane), (0.0, 0.0), _lane_point(endpoint)]))
        elif operator == "V" and len(absolute) >= 2:
            corridor.update(_polyline_lanes([_lane_point(event.lane), _lane_point(absolute[-2]), _lane_point(endpoint)]))
        elif operator in ("s", "z"):
            direction = 1 if operator == "s" else -1
            start_angle_lane = ((event.lane - 1 + 2 * direction) % 8) + 1
            end_angle_lane = ((endpoint - 1 + 2 * direction) % 8) + 1
            corridor.update(_polyline_lanes([
                _lane_point(event.lane),
                _lane_point(start_angle_lane, 0.32),
                _lane_point(end_angle_lane, 0.32),
                _lane_point(endpoint),
            ]))
        elif operator == "w":
            for fan_end in (((endpoint - 2) % 8) + 1, endpoint, (endpoint % 8) + 1):
                corridor.update(_polyline_lanes([_lane_point(event.lane), _lane_point(fan_end)]))
        elif operator == "p":
            corridor.update(_ring_lanes(event.lane, endpoint, -1))
        elif operator == "q":
            corridor.update(_ring_lanes(event.lane, endpoint, 1))
        elif operator in ("pp", "qq"):
            corridor.update(_curved_slide_side_lanes(event.lane, endpoint))
        elif operator in ("<", ">"):
            step = _edge_slide_step(event.lane, operator)
            corridor.update(_ring_lanes(event.lane, endpoint, step))
        else:
            corridor.update(_polyline_lanes([_lane_point(event.lane), _lane_point(endpoint)]))
    corridor.add(final_endpoint)
    return corridor, final_endpoint


def slide_tail_lanes(event: GeneratedEvent) -> set[int]:
    template = event.slide_template
    if event.kind != "slide" or not template:
        return {event.lane}
    tails = set()
    for operator, relative_path in zip(template["operators"], template["relative_paths"]):
        absolute = [((event.lane - 1 + int(relative)) % 8) + 1 for relative in relative_path]
        if not absolute:
            continue
        endpoint = absolute[-1]
        if operator == "w":
            tails.update((((endpoint - 2) % 8) + 1, endpoint, (endpoint % 8) + 1))
        else:
            tails.add(endpoint)
    return tails or {event.lane}


def event_hand_cost(event: GeneratedEvent) -> int:
    if event.kind == "slide" and event.slide_template:
        if any(operator == "w" for operator in event.slide_template["operators"]):
            return 2
    return 1


def slide_motion_interval(event: GeneratedEvent) -> tuple[int, int] | None:
    if event.kind != "slide" or not event.slide_template:
        return None
    starts = []
    ends = []
    for duration in event.slide_template["durations"]:
        wait = duration.get("wait_ticks")
        wait = 48 if wait is None else int(wait)
        movement = int(duration.get("duration_ticks") or 0)
        starts.append(event.tick + wait)
        ends.append(event.tick + wait + movement)
    if not starts:
        return None
    return min(starts), max(ends)


def sixteenth_stream_ticks(events: list[GeneratedEvent]) -> set[int]:
    ticks = sorted({event.tick for event in events})
    result = set()
    start = 0
    while start < len(ticks):
        end = start + 1
        while end < len(ticks) and ticks[end] - ticks[end - 1] == 12:
            end += 1
        run = ticks[start:end]
        if len(run) >= 3:
            result.update(run)
        start = max(start + 1, end)
    return result


def playability_conflicts(events: list[GeneratedEvent]) -> dict[str, int]:
    streams = sixteenth_stream_ticks(events)
    slide_path_tap = 0
    long_object_sixteenth = 0
    slide_tails = []
    hand_occupations = []
    taps = [event for event in events if event.kind == "tap"]
    for event in events:
        if event.kind == "hold":
            interval = (event.tick, event.tick + event.duration)
            hand_end = interval[1] + LONG_OBJECT_RELEASE_TICKS
            path_lanes = {event.lane}
        elif event.kind == "slide" and event.slide_template:
            interval = slide_motion_interval(event)
            hand_end = interval[1] + LONG_OBJECT_RELEASE_TICKS if interval else event.tick
            path_lanes, _ = slide_corridor_lanes(event)
            if interval:
                slide_tails.append((interval[1], slide_tail_lanes(event)))
                slide_path_tap += sum(
                    interval[0] <= tap.tick <= interval[1] and tap.lane in path_lanes
                    for tap in taps
                )
        else:
            continue
        if interval and (
            event.tick in streams
            or any(interval[0] <= tick < hand_end for tick in streams)
        ):
            long_object_sixteenth += 1
        if interval:
            hand_occupations.append((event.tick, interval[0], hand_end, event_hand_cost(event)))
    slide_tail_conflicts = sum(
        bool(first_endpoints & second_endpoints)
        and abs(first_tick - second_tick) < SLIDE_TAIL_CLEARANCE_TICKS
        for index, (first_tick, first_endpoints) in enumerate(slide_tails)
        for second_tick, second_endpoints in slide_tails[index + 1:]
    )
    head_costs = Counter()
    for event in events:
        head_costs[event.tick] += event_hand_cost(event)
    max_hand_demand = max(
        (
            count
            + sum(
                cost for head, start, hand_end, cost in hand_occupations
                if head < tick and start <= tick < hand_end
            )
            for tick, count in head_costs.items()
        ),
        default=0,
    )
    return {
        "slide_path_tap_conflicts": slide_path_tap,
        "long_object_sixteenth_conflicts": long_object_sixteenth,
        "slide_tail_clearance_conflicts": slide_tail_conflicts,
        "max_hand_demand": max_hand_demand,
    }


def enforce_hand_capacity(events: list[GeneratedEvent], catalog: list[dict], level: float) -> list[GeneratedEvent]:
    assign_slide_templates(events, catalog, level)
    normalize_slide_speeds(events, catalog)
    stream_ticks = sixteenth_stream_ticks(events)
    accepted: list[GeneratedEvent] = []
    occupations: list[dict] = []
    for event in sorted(events, key=lambda item: (item.tick, -item.score)):
        blocked_lanes = set()
        for occupation in occupations:
            if occupation["kind"] == "hold" and occupation["head"] < event.tick < occupation["hand_end"]:
                blocked_lanes.add(occupation["lane"])
            elif occupation["kind"] == "slide" and occupation["start"] <= event.tick <= occupation["end"]:
                blocked_lanes.update(occupation["path_lanes"])
            if (
                occupation["kind"] == "slide"
                and occupation["end"] < event.tick < occupation["tail_clear"]
            ):
                blocked_lanes.update(occupation["endpoints"])
        blocked_lanes.update(other.lane for other in accepted[-2:] if other.tick == event.tick)
        if event.lane in blocked_lanes:
            available = [lane for lane in range(1, 9) if lane not in blocked_lanes]
            if not available:
                continue
            event.lane = min(
                available,
                key=lambda lane: min(abs(lane - event.lane), 8 - abs(lane - event.lane)),
            )

        active_at_head = sum(
            occupation["hand_cost"]
            for occupation in occupations
            if occupation["start"] <= event.tick < occupation["hand_end"]
            and occupation["head"] < event.tick
        )
        simultaneous_heads = sum(
            event_hand_cost(other) for other in accepted[-2:] if other.tick == event.tick
        )
        current_hand_cost = event_hand_cost(event)
        if active_at_head + simultaneous_heads + current_hand_cost > 2:
            continue

        interval = None
        path_lanes = set()
        endpoint = event.lane
        endpoints = {event.lane}
        if event.kind == "hold":
            interval = (event.tick, event.tick + event.duration)
        elif event.kind == "slide" and event.slide_template:
            interval = slide_motion_interval(event)
            path_lanes, endpoint = slide_corridor_lanes(event)
            endpoints = slide_tail_lanes(event)
            if any(
                occupation["kind"] == "slide"
                and bool(occupation["endpoints"] & endpoints)
                and abs(occupation["end"] - interval[1]) < SLIDE_TAIL_CLEARANCE_TICKS
                for occupation in occupations
            ):
                continue

        if event.kind in {"hold", "slide"} and (
            event.tick in stream_ticks
            or (
                interval
                and any(
                    interval[0] <= tick < interval[1] + LONG_OBJECT_RELEASE_TICKS
                    for tick in stream_ticks
                )
            )
        ):
            continue
        if event.kind == "tap" and event.tick in stream_ticks and any(
            occupation["start"] <= event.tick < occupation["hand_end"]
            for occupation in occupations
        ):
            continue

        if interval and interval[1] > interval[0]:
            hand_end = interval[1] + LONG_OBJECT_RELEASE_TICKS
            boundaries = {interval[0], hand_end}
            boundaries.update(
                item["start"] for item in occupations if interval[0] <= item["start"] <= hand_end
            )
            boundaries.update(
                item["hand_end"] for item in occupations if interval[0] <= item["hand_end"] <= hand_end
            )
            if any(
                current_hand_cost + sum(
                    item["hand_cost"] for item in occupations
                    if item["start"] <= tick < item["hand_end"]
                ) > 2
                for tick in boundaries
            ):
                continue
            occupations.append({
                "start": interval[0], "end": interval[1], "lane": event.lane,
                "hand_end": hand_end,
                "tail_clear": interval[1] + SLIDE_TAIL_CLEARANCE_TICKS,
                "kind": event.kind, "head": event.tick, "path_lanes": path_lanes,
                "endpoint": endpoint, "endpoints": endpoints,
                "hand_cost": current_hand_cost,
            })
        accepted.append(event)
    return sorted(accepted, key=lambda item: (item.tick, item.lane))


def slide_text(
    event: GeneratedEvent,
    events: list[GeneratedEvent],
    catalog: list[dict],
    previous_operator: str | None,
    level: float,
) -> tuple[str, str | None]:
    template = event.slide_template or choose_slide_template(event, events, catalog, previous_operator, level)
    if not template:
        return str(event.lane), previous_operator
    operator = template["operators"][0]
    modifier = ("b" if event.is_break else "") + ("x" if event.is_ex else "")
    branches = []
    for branch_operator, relative_path, duration in zip(
        template["operators"], template["relative_paths"], template["durations"]
    ):
        path = "".join(str(((event.lane - 1 + int(relative)) % 8) + 1) for relative in relative_path)
        branches.append(f"{branch_operator}{path}[{duration['raw']}]")
    return f"{event.lane}{modifier}" + "*".join(branches), operator


def compile_chart(events: list[GeneratedEvent], bpm: float, total_ticks: int, catalog: list[dict], level: float) -> str:
    slots: dict[int, list[str]] = {}
    previous_operator = None
    for event in events:
        modifier = ("b" if event.is_break else "") + ("x" if event.is_ex else "")
        if event.kind == "tap":
            text = f"{event.lane}{modifier}"
        elif event.kind == "hold":
            text = f"{event.lane}{modifier}h[192:{event.duration}]"
        else:
            text, previous_operator = slide_text(event, events, catalog, previous_operator, level)
        slots.setdefault(event.tick // 2, []).append(text)

    slot_count = math.ceil(total_ticks / 2)
    measures = math.ceil(slot_count / 96)
    lines = []
    for measure in range(measures):
        tokens = []
        for local in range(96):
            slot = measure * 96 + local
            token = "/".join(slots.get(slot, []))
            if measure == 0 and local == 0:
                token = f"({bpm:g}){{96}}" + token
            tokens.append(token)
        lines.append(",".join(tokens) + ",")
    return "\n".join(lines) + "\nE"


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if not args.audio.exists():
        raise FileNotFoundError(args.audio)
    if args.audio.suffix.lower() != ".mp3":
        raise ValueError("The first prototype currently requires MP3 input so the output can use track.mp3 directly")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    config = json.loads(AUDIO_CONFIG.read_text(encoding="utf-8"))
    controls = control_defaults(args.level)
    overrides = (args.density, args.hold_ratio, args.slide_ratio, args.break_ratio)
    for index, value in enumerate(overrides, 1):
        if value is not None:
            controls[index] = value
    if args.slide_ratio is None:
        controls[3] = min(0.35, controls[3] * DEFAULT_SLIDE_RATIO_BOOST)
    log_mel, duration_seconds = extract_log_mel(args.audio, config)
    first_ms = args.offset * 1000.0
    usable_ms = max(1.0, duration_seconds * 1000.0 - first_ms)
    total_ticks = max(192, int(usable_ms * args.bpm * TICKS_PER_MEASURE / 240000.0))
    starts = window_starts(total_ticks)
    tick_ms = 240000.0 / (args.bpm * TICKS_PER_MEASURE)

    window_weight = np.maximum(0.1, np.hanning(WINDOW_TICKS)).astype(np.float32)
    rhythm_fast_notes_removed = 0
    hierarchical_rhythm = False
    if args.timing_model == "rhythm":
        model = RhythmPlanModel().cuda().eval()
        rhythm_state = torch.load(RHYTHM_CHECKPOINT, map_location="cuda", weights_only=False)["model"]
        onset_enabled = "onset_head.weight" in rhythm_state
        subdivision_enabled = "subdivision_head.weight" in rhythm_state
        accent_enabled = "accent_head.weight" in rhythm_state
        hierarchical_rhythm = subdivision_enabled and accent_enabled
        model.load_state_dict(rhythm_state, strict=False)
        event_sum = np.zeros((5, max(total_ticks, WINDOW_TICKS)), dtype=np.float64)
        count_sum = np.zeros((3, max(total_ticks, WINDOW_TICKS)), dtype=np.float64)
        onset_sum = np.zeros(max(total_ticks, WINDOW_TICKS), dtype=np.float64)
        subdivision_sum = np.zeros((8, max(total_ticks, WINDOW_TICKS)), dtype=np.float64)
        accent_sum = np.zeros(max(total_ticks, WINDOW_TICKS), dtype=np.float64)
        probability_weight = np.zeros(max(total_ticks, WINDOW_TICKS), dtype=np.float64)
        for batch_start in range(0, len(starts), args.batch_size):
            batch_starts = starts[batch_start : batch_start + args.batch_size]
            features = []
            for start in batch_starts:
                times = first_ms + (start + np.arange(WINDOW_TICKS + 1)) * tick_ms
                features.append(aligned_audio(log_mel, times, config))
            audio = torch.from_numpy(np.stack(features)).cuda()
            control_batch = torch.from_numpy(np.repeat(controls[None], len(batch_starts), axis=0)).cuda()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                output = model(audio, control_batch)
                event_probabilities = torch.sigmoid(output["event"]).float().cpu().numpy()
                count_probabilities = torch.softmax(output["count"], dim=1).float().cpu().numpy()
                onset_probabilities = torch.sigmoid(output["onset"][:, 0]).float().cpu().numpy()
                subdivision_probabilities = (
                    torch.softmax(output["subdivision"], dim=1).float().cpu().numpy()
                    if subdivision_enabled else None
                )
                accent_probabilities = (
                    torch.sigmoid(output["accent"][:, 0]).float().cpu().numpy()
                    if accent_enabled else None
                )
            for index, start in enumerate(batch_starts):
                end = min(start + WINDOW_TICKS, event_sum.shape[1])
                width = end - start
                event_sum[:, start:end] += event_probabilities[index, :, :width] * window_weight[:width]
                count_sum[:, start:end] += count_probabilities[index, :, :width] * window_weight[:width]
                if onset_enabled:
                    onset_sum[start:end] += onset_probabilities[index, :width] * window_weight[:width]
                if subdivision_enabled and subdivision_probabilities is not None:
                    subdivision_sum[:, start:end] += (
                        subdivision_probabilities[index, :, :width] * window_weight[None, :width]
                    )
                if accent_enabled and accent_probabilities is not None:
                    accent_sum[start:end] += accent_probabilities[index, :width] * window_weight[:width]
                probability_weight[start:end] += window_weight[:width]
            print(f"analyzed_windows={min(batch_start + args.batch_size, len(starts))}/{len(starts)}")
        event_probabilities = (
            event_sum[:, :total_ticks] / np.maximum(probability_weight[:total_ticks], 1e-6)
        ).astype(np.float32)
        count_probabilities = (
            count_sum[:, :total_ticks] / np.maximum(probability_weight[:total_ticks], 1e-6)
        ).astype(np.float32)
        onset_probabilities = (
            onset_sum[:total_ticks] / np.maximum(probability_weight[:total_ticks], 1e-6)
        ).astype(np.float32) if onset_enabled else None
        subdivision_probabilities = (
            subdivision_sum[:, :total_ticks] / np.maximum(probability_weight[None, :total_ticks], 1e-6)
        ).astype(np.float32) if subdivision_enabled else None
        accent_probabilities = (
            accent_sum[:total_ticks] / np.maximum(probability_weight[:total_ticks], 1e-6)
        ).astype(np.float32) if accent_enabled else None
        events = rhythm_plan_events(
            event_probabilities,
            count_probabilities,
            onset_probabilities,
            controls,
            total_ticks,
            subdivision_probabilities,
            accent_probabilities,
        )
        events_before_fast_cleanup = len(events)
        events = clean_irregular_fast_notes(events)
        rhythm_fast_notes_removed = events_before_fast_cleanup - len(events)
        timing_checkpoint = RHYTHM_CHECKPOINT
    else:
        checkpoint = torch.load(CHECKPOINT, map_location="cuda", weights_only=False)
        model = MugInspiredAudioChartDiffusion().cuda().eval()
        model.load_state_dict(checkpoint["model"])
        vae = ChartVAE().cuda().eval()
        vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location="cuda", weights_only=False)["model"])
        schedule = torch.cumprod(1.0 - cosine_schedule(1000, torch.device("cuda")), dim=0)
        latent_mean = checkpoint["latent_mean"].cuda()
        latent_std = checkpoint["latent_std"].cuda()
        probability_sum = np.zeros((48, max(total_ticks, WINDOW_TICKS)), dtype=np.float64)
        probability_weight = np.zeros(max(total_ticks, WINDOW_TICKS), dtype=np.float64)
        rhythm_length = math.ceil(max(total_ticks, WINDOW_TICKS) / 8)
        rhythm_sum = np.zeros((5, rhythm_length), dtype=np.float64)
        rhythm_weight = np.zeros(rhythm_length, dtype=np.float64)
        latent_weight = np.maximum(0.1, np.hanning(384)).astype(np.float32)
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        for batch_start in range(0, len(starts), args.batch_size):
            batch_starts = starts[batch_start : batch_start + args.batch_size]
            features = []
            for start in batch_starts:
                times = first_ms + (start + np.arange(WINDOW_TICKS + 1)) * tick_ms
                features.append(aligned_audio(log_mel, times, config))
            audio = torch.from_numpy(np.stack(features)).cuda()
            control_batch = torch.from_numpy(np.repeat(controls[None], len(batch_starts), axis=0)).cuda()
            latent = ddim_sample(
                model, audio, control_batch, schedule, sampling_steps=args.steps,
                control_guidance=args.guidance, generator=generator,
            )
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                logits = vae.decode(latent * latent_std + latent_mean)
                probabilities = torch.sigmoid(logits).float().cpu().numpy()
                audio0, _, _ = model.audio(audio)
                context = model.controls(control_batch)
                rhythm = torch.sigmoid(
                    model.rhythm_head(audio0 + model.rhythm_control(context.mean(dim=1))[:, :, None])
                ).float().cpu().numpy()
            for index, start in enumerate(batch_starts):
                end = min(start + WINDOW_TICKS, probability_sum.shape[1])
                width = end - start
                probability_sum[:, start:end] += probabilities[index, :, :width] * window_weight[:width]
                probability_weight[start:end] += window_weight[:width]
                latent_start = start // 8
                latent_end = min(latent_start + 384, rhythm_length)
                latent_width = latent_end - latent_start
                rhythm_sum[:, latent_start:latent_end] += rhythm[index, :, :latent_width] * latent_weight[:latent_width]
                rhythm_weight[latent_start:latent_end] += latent_weight[:latent_width]
            print(f"generated_windows={min(batch_start + args.batch_size, len(starts))}/{len(starts)}")
        probabilities = (
            probability_sum[:, :total_ticks] / np.maximum(probability_weight[:total_ticks], 1e-6)
        ).astype(np.float32)
        rhythm = (rhythm_sum / np.maximum(rhythm_weight, 1e-6)).astype(np.float32)
        events = legalize_events(probabilities, rhythm, controls, total_ticks)
        timing_checkpoint = CHECKPOINT
    catalog = json.loads(STAR_CATALOG.read_text(encoding="utf-8"))
    pattern_heat = (args.interaction_heat, args.sweep_heat, args.jack_heat)
    if any(not 0.0 <= value <= 2.0 for value in pattern_heat):
        raise ValueError("Pattern heat values must be between 0 and 2")
    events = arrange_patterns(events, total_ticks, args.level, args.seed, pattern_heat)
    events, irregular_sixteenth_removed = break_irregular_sixteenth_runs(events, args.bpm)
    events, eighth_orbits_broken = break_eighth_note_orbits(events)
    jack_max_share = max_jack_share_for_bpm(args.bpm, args.jack_heat)
    events, jack_limit_repositioned = limit_jack_patterns(
        events, remove_excess=False, max_share=jack_max_share
    )
    events_before_hand_filter = len(events)
    events = enforce_hand_capacity(events, catalog, args.level)
    events, long_eighth_repositioned = break_long_eighth_jacks(events)
    handflow_report = None
    if HAND_FLOW_OPTIMIZER is not None:
        events, handflow_report = HAND_FLOW_OPTIMIZER(events)
        events, post_handflow_eighth_changes = break_long_eighth_jacks(events)
        if post_handflow_eighth_changes:
            first_handflow_report = handflow_report
            events, handflow_report = HAND_FLOW_OPTIMIZER(events)
            handflow_report["first_pass"] = first_handflow_report
        long_eighth_repositioned += post_handflow_eighth_changes
    events, jack_limit_removed = limit_jack_patterns(
        events, remove_excess=True, max_share=jack_max_share
    )
    final_pattern_events, final_pattern_segments, final_jack_share = generated_pattern_summary(events)
    final_playability_conflicts = playability_conflicts(events)
    unique_event_ticks = sorted({event.tick for event in events})
    rhythm_gap_histogram = Counter(
        str(right - left) for left, right in zip(unique_event_ticks, unique_event_ticks[1:])
    )
    slowed_slides = sum(event.slide_was_slowed for event in events if event.kind == "slide")
    eighth_note_slides = sum(
        event.kind == "slide" and event.tick % 24 == 0 for event in events
    )
    max_slide_speed = max(
        (event.slide_max_speed for event in events if event.kind == "slide"), default=0.0
    )
    chart = compile_chart(events, args.bpm, total_ticks, catalog, args.level)

    title = args.title or args.audio.stem
    output = args.output or (ROOT / "generated_maimai" / f"{args.audio.stem}_lv{args.level:g}_seed{args.seed}")
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.audio, output / "track.mp3")
    maidata = (
        f"&title={title}\n&artist={args.artist}\n&first={args.offset:g}\n&wholebpm={args.bpm:g}\n"
        f"&lv_5={args.level:g}\n&des_5={args.designer}\n&inote_5={chart}\n"
    )
    (output / "maidata.txt").write_text(maidata, encoding="utf-8", newline="\n")
    type_counts = {kind: sum(event.kind == kind for event in events) for kind in ("tap", "hold", "slide")}
    report = {
        "engine": {
            "name": ENGINE_NAME,
            "version": ENGINE_VERSION,
            "creator": ENGINE_CREATOR,
            "display_name": f"{ENGINE_NAME} {ENGINE_VERSION} by {ENGINE_CREATOR}",
        },
        "model": str(timing_checkpoint),
        "timing_model": args.timing_model,
        "hierarchical_rhythm": hierarchical_rhythm,
        "arranger_model": str(ARRANGER_CHECKPOINT),
        "audio": str(args.audio),
        "duration_seconds": duration_seconds,
        "bpm": args.bpm,
        "offset_seconds": args.offset,
        "level": args.level,
        "controls": {
            "density_per_measure": float(controls[1]),
            "hold_ratio": float(controls[2]),
            "slide_ratio": float(controls[3]),
            "break_ratio": float(controls[4]),
        },
        "seed": args.seed,
        "sampling_steps": args.steps,
        "windows": len(starts),
        "total_ticks": total_ticks,
        "events": len(events),
        "hand_capacity_removed": events_before_hand_filter - len(events),
        "rhythm_fast_notes_removed": rhythm_fast_notes_removed,
        "irregular_sixteenth_removed": irregular_sixteenth_removed,
        "rhythm_gap_histogram": dict(rhythm_gap_histogram.most_common()),
        "event_types": type_counts,
        "breaks": sum(event.is_break for event in events),
        "slide_operators": dict(Counter(OPERATORS[event.operator_id] for event in events if event.kind == "slide")),
        "pattern_events": final_pattern_events,
        "pattern_segments": final_pattern_segments,
        "pattern_heat": {
            "interaction": args.interaction_heat,
            "sweep": args.sweep_heat,
            "jack": args.jack_heat,
        },
        "generation_policy": {
            "default_slide_ratio_boost": DEFAULT_SLIDE_RATIO_BOOST,
            "eighth_note_slide_score_boost": EIGHTH_SLIDE_SCORE_BOOST,
            "eighth_note_orbits_broken": eighth_orbits_broken,
        },
        "handflow": handflow_report,
        "jack_pattern_share": round(final_jack_share, 6),
        "jack_pattern_max_share": round(jack_max_share, 6),
        "jack_limit_repositioned": jack_limit_repositioned,
        "jack_limit_removed": jack_limit_removed,
        "long_eighth_repositioned": long_eighth_repositioned,
        "long_eighth_jack_excess": long_eighth_jack_excess(events),
        **final_playability_conflicts,
        "slides_slowed_by_geometry": slowed_slides,
        "eighth_note_slide_heads": eighth_note_slides,
        "max_normalized_slide_speed": round(max_slide_speed, 6),
        "output": str(output),
    }
    (output / "generation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
