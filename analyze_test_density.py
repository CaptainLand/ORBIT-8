from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(r"D:\trans")
DEFAULT_INDEX = ROOT / "maimai_finale_dataset" / "prepared_v2" / "chart_index_12_15.jsonl"
DEFAULT_OUTPUT = ROOT / "maimai_finale_dataset" / "prepared_v2" / "test_density_profile.json"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * fraction
    left = int(math.floor(position))
    right = int(math.ceil(position))
    if left == right:
        return ordered[left]
    weight = position - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def isotonic(values: list[float], weights: list[float]) -> list[float]:
    blocks: list[dict[str, float | int]] = []
    for index, (value, weight) in enumerate(zip(values, weights)):
        blocks.append({"start": index, "end": index, "weight": weight, "sum": value * weight})
        while len(blocks) >= 2:
            left = blocks[-2]
            right = blocks[-1]
            left_value = float(left["sum"]) / float(left["weight"])
            right_value = float(right["sum"]) / float(right["weight"])
            if left_value <= right_value:
                break
            blocks[-2:] = [{
                "start": int(left["start"]),
                "end": int(right["end"]),
                "weight": float(left["weight"]) + float(right["weight"]),
                "sum": float(left["sum"]) + float(right["sum"]),
            }]
    result = [0.0] * len(values)
    for block in blocks:
        value = float(block["sum"]) / float(block["weight"])
        for index in range(int(block["start"]), int(block["end"]) + 1):
            result[index] = value
    return result


def chart_density(row: dict, prepared_root: Path) -> tuple[float, float]:
    payload = json.loads((prepared_root / row["event_path"]).read_text(encoding="utf-8"))
    duration_seconds = max(1, math.ceil(float(payload["duration_ms"]) / 1000.0))
    per_second = [0] * duration_seconds
    for event in payload["events"]:
        second = min(duration_seconds - 1, max(0, int(float(event["time_ms"]) // 1000)))
        per_second[second] += 1
    rms = math.sqrt(sum(count * count for count in per_second) / len(per_second))
    mean = sum(per_second) / len(per_second)
    return rms, mean


def build_profile(index_path: Path) -> dict:
    rows = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [
        row for row in rows
        if 12.0 <= float(row["level"]) <= 15.0
    ]
    prepared_root = index_path.parent
    samples = []
    grouped: dict[float, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        rms, mean = chart_density(row, prepared_root)
        level = round(float(row["level"]), 1)
        grouped[level].append((rms, mean))
        samples.append((level, rms))

    raw_levels = []
    for level in sorted(grouped):
        values = [value[0] for value in grouped[level]]
        means = [value[1] for value in grouped[level]]
        raw_levels.append({
            "level": level,
            "charts": len(values),
            "median_rms_notes_per_second": round(statistics.median(values), 4),
            "mean_rms_notes_per_second": round(statistics.fmean(values), 4),
            "median_plain_notes_per_second": round(statistics.median(means), 4),
            "p25_rms_notes_per_second": round(percentile(values, 0.25), 4),
            "p75_rms_notes_per_second": round(percentile(values, 0.75), 4),
        })

    levels = [round(12.0 + index * 0.1, 1) for index in range(31)]
    smoothed = []
    effective_weights = []
    sigma = 0.28
    for level in levels:
        weights = [math.exp(-0.5 * ((sample_level - level) / sigma) ** 2) for sample_level, _ in samples]
        total_weight = max(1e-8, sum(weights))
        smoothed.append(sum(weight * density for weight, (_, density) in zip(weights, samples)) / total_weight)
        effective_weights.append(total_weight)
    recommended = isotonic(smoothed, effective_weights)

    profile = []
    for level, density in zip(levels, recommended):
        nearby = sum(abs(sample_level - level) <= 0.25 for sample_level, _ in samples)
        exact = len(grouped.get(level, []))
        profile.append({
            "level": level,
            "rms_notes_per_second": round(density, 4),
            "exact_charts": exact,
            "nearby_charts": nearby,
        })

    densities = [density for _, density in samples]
    return {
        "source": str(index_path.relative_to(ROOT)),
        "scope": "all prepared charts across every imported version and data split",
        "level_range": [12.0, 15.0],
        "density_definition": "sqrt(mean(notes_in_each_second ** 2)); tap + hold + slide heads",
        "charts": len(rows),
        "versions": dict(sorted(Counter(row["song_id"].split("_")[0] for row in rows).items())),
        "raw_levels": raw_levels,
        "profile": profile,
        "summary": {
            "median_chart_rms_notes_per_second": round(statistics.median(densities), 4),
            "min_chart_rms_notes_per_second": round(min(densities), 4),
            "max_chart_rms_notes_per_second": round(max(densities), 4),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ORBIT-8 test-set density recommendations")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = build_profile(args.index)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **result["summary"], "charts": result["charts"]}))


if __name__ == "__main__":
    main()
