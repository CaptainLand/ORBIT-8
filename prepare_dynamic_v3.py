from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(r"D:\trans")
SOURCE = ROOT / "maimai_finale_dataset" / "prepared_v2"
OUTPUT = ROOT / "maimai_finale_dataset" / "prepared_v3"
TICKS_PER_MEASURE = 192


def chart_anchors(chart: dict) -> dict[str, list[int]]:
    event_data = json.loads((SOURCE / chart["event_path"]).read_text(encoding="utf-8"))
    events = event_data["events"]
    tick_counts = Counter(int(event["tick"]) for event in events)
    event_ticks = sorted(tick_counts)
    dense = []
    for index, tick in enumerate(event_ticks):
        previous_gap = tick - event_ticks[index - 1] if index else 10**9
        next_gap = event_ticks[index + 1] - tick if index + 1 < len(event_ticks) else 10**9
        if min(previous_gap, next_gap) <= 12:
            dense.append(tick)
    long_object = sorted({
        int(event["tick"]) for event in events if event["note_type"] in {"hold", "slide"}
    })
    double = sorted(tick for tick, count in tick_counts.items() if count >= 2)
    dense_set = set(dense)
    regular = [tick for tick in event_ticks if tick not in dense_set]
    occupied_measures = {tick // TICKS_PER_MEASURE for tick in event_ticks}
    measure_count = max(1, (int(chart["total_ticks"]) + TICKS_PER_MEASURE - 1) // TICKS_PER_MEASURE)
    silence = [
        measure * TICKS_PER_MEASURE + TICKS_PER_MEASURE // 2
        for measure in range(measure_count)
        if measure not in occupied_measures
    ]
    return {
        "regular": regular,
        "dense": dense,
        "long": long_object,
        "double": double,
        "silence": silence,
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = []
    totals = Counter()
    for line in (SOURCE / "chart_index.jsonl").read_text(encoding="utf-8").splitlines():
        chart = json.loads(line)
        anchors = chart_anchors(chart)
        row = {
            "chart_id": chart["chart_id"],
            "song_id": chart["song_id"],
            "split": chart["split"],
            "level": chart["level"],
            "tensor_path": chart["tensor_path"],
            "total_ticks": chart["total_ticks"],
            "anchors": anchors,
        }
        rows.append(row)
        totals[f"charts_{chart['split']}"] += 1
        for name, values in anchors.items():
            totals[f"anchors_{name}"] += len(values)
    with (OUTPUT / "dynamic_index.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    config = {
        "version": 3,
        "source": str(SOURCE),
        "storage_policy": "index-only; charts and audio are loaded from prepared_v2",
        "ticks_per_measure": TICKS_PER_MEASURE,
        "crop_measures": {"8": 0.25, "12": 0.25, "16": 0.50},
        "sample_categories": {
            "regular": 0.60,
            "dense": 0.15,
            "long": 0.10,
            "double": 0.10,
            "silence": 0.05,
        },
        "stats": dict(totals),
    }
    (OUTPUT / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
