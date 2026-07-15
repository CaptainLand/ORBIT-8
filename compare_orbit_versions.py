from __future__ import annotations

import json
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path

from maimai_ai.patterns import PATTERN_NAMES, PATTERN_SWEEP, detect_pattern_labels
from maimai_ai.simai import parse_chart, parse_fields


ROOT = Path(r"D:\trans")
PREPARED = ROOT / "maimai_finale_dataset" / "prepared_v2"
VERSIONS = {
    "v1": ROOT / "output" / "B.M.S. Strict Hand-Safe",
    "v1.5": ROOT / "output" / "B.M.S. ORBIT-8 v1.5",
    "v1.6": ROOT / "output" / "B.M.S. ORBIT-8 v1.6",
    "v1.7": ROOT / "output" / "B.M.S. ORBIT-8 v1.7 final",
}


def greedy_match(predicted: list[float], expected: list[float], tolerance_ms: float) -> int:
    left = right = matched = 0
    while left < len(predicted) and right < len(expected):
        delta = predicted[left] - expected[right]
        if abs(delta) <= tolerance_ms:
            matched += 1
            left += 1
            right += 1
        elif delta < 0:
            left += 1
        else:
            right += 1
    return matched


def onset_metrics(predicted: list[float], expected: list[float], tolerance_ms: float) -> dict:
    matched = greedy_match(predicted, expected, tolerance_ms)
    precision = matched / max(1, len(predicted))
    recall = matched / max(1, len(expected))
    return {
        "matched": matched,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / max(1e-12, precision + recall), 6),
    }


def peak_heads(events: list[dict], labels: list[int], *, exclude_sweeps: bool) -> int:
    queue = deque()
    peak = 0
    for event, label in sorted(zip(events, labels), key=lambda item: item[0]["time_ms"]):
        if exclude_sweeps and label == PATTERN_SWEEP:
            continue
        time_ms = float(event["time_ms"])
        queue.append(time_ms)
        while queue and queue[0] < time_ms - 1000.0:
            queue.popleft()
        peak = max(peak, len(queue))
    return peak


def pattern_segments(labels: list[int]) -> Counter:
    result = Counter()
    previous = 0
    for label in labels:
        if label and label != previous:
            result[PATTERN_NAMES[label]] += 1
        previous = label
    return result


def evaluate(path: Path, official_onsets: list[float], official_event_count: int) -> dict:
    maidata = path / "maidata.txt"
    chart = parse_chart(parse_fields(maidata.read_text(encoding="utf-8-sig")), 5)
    events = [asdict(event) for event in chart.events]
    labels = detect_pattern_labels(events)
    onsets = sorted({round(float(event["time_ms"]), 6) for event in events})
    ticks = sorted({int(event["tick"]) for event in events})
    gaps = [right - left for left, right in zip(ticks, ticks[1:])]
    simultaneous = Counter(event["tick"] for event in events)
    types = Counter(event["note_type"] for event in events)
    patterns = Counter(PATTERN_NAMES[label] for label in labels)
    segments = pattern_segments(labels)

    invalid_slides = 0
    same_lane_inside_hold = 0
    for event in events:
        if event["note_type"] == "slide":
            for branch in event["branches"]:
                invalid_slides += int(
                    (branch["operator"] == "V" and len(branch["path_lanes"]) != 2)
                    or (branch["operator"] != "V" and len(branch["path_lanes"]) != 1)
                )
        if event["note_type"] == "hold" and event["duration"]:
            end_tick = event["tick"] + event["duration"]["duration_ticks"]
            same_lane_inside_hold += sum(
                other is not event
                and other["lane"] == event["lane"]
                and event["tick"] < other["tick"] < end_tick
                for other in events
            )

    report_path = path / "generation_report.json"
    generation = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    return {
        "folder": str(path),
        "bpm": chart.default_bpm,
        "offset_seconds": chart.first_ms / 1000.0,
        "events": len(events),
        "event_count_error": len(events) - official_event_count,
        "unique_onsets": len(onsets),
        "types": dict(types),
        "onset_20ms": onset_metrics(onsets, official_onsets, 20.0),
        "onset_30ms": onset_metrics(onsets, official_onsets, 30.0),
        "onset_50ms": onset_metrics(onsets, official_onsets, 50.0),
        "patterns": dict(patterns),
        "pattern_segments": dict(segments),
        "max_simultaneous": max(simultaneous.values(), default=0),
        "peak_heads_1s": peak_heads(events, labels, exclude_sweeps=False),
        "peak_non_sweep_heads_1s": peak_heads(events, labels, exclude_sweeps=True),
        "sub_16th_gaps": sum(gap < 12 for gap in gaps),
        "minimum_gap_ticks": min(gaps, default=None),
        "invalid_slides": invalid_slides,
        "same_lane_inside_hold": same_lane_inside_hold,
        "hand_capacity_removed": generation.get("hand_capacity_removed"),
        "rhythm_fast_notes_removed": generation.get("rhythm_fast_notes_removed", 0),
        "terminator_valid": maidata.read_text(encoding="utf-8-sig").strip().endswith("E"),
    }


def main() -> None:
    official = json.loads((PREPARED / "events" / "finale_0004_d4.json").read_text(encoding="utf-8"))
    official_onsets = sorted({round(float(event["time_ms"]), 6) for event in official["events"]})
    result = {
        "benchmark": {
            "song": "B.M.S.",
            "level": 12.6,
            "official_events": len(official["events"]),
            "official_unique_onsets": len(official_onsets),
            "matching": "greedy one-to-one onset matching",
            "pattern_definitions": {
                "interaction": "alternating lanes at 12-tick (16th-note) spacing",
                "jack": "one lane at 12-tick (16th-note) spacing",
                "sweep": "continuous lanes at 12/8/6-tick (16th/24th/32nd) spacing",
                "eighth_notes": "24-tick spacing is never a configuration",
            },
        },
        "versions": {
            version: evaluate(path, official_onsets, len(official["events"]))
            for version, path in VERSIONS.items()
        },
    }
    output = ROOT / "ORBIT-8-VERSION-COMPARISON.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
