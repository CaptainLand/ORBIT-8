from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import torch

from maimai_ai.arranger import OPERATORS, OfficialPatternArranger, OraclePlanDataset, operator_calibration_bias


ROOT = Path(r"D:\trans")
DATASET = ROOT / "maimai_finale_dataset"
PREPARED = DATASET / "prepared_v1"
CHECKPOINT = ROOT / "maimai_arranger" / "runs" / "finale_arranger_v1" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("chart_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-lane", type=int, default=1, choices=range(1, 9))
    return parser.parse_args()


def move_sample(sample: dict) -> dict:
    return {
        key: value.unsqueeze(0).cuda() if torch.is_tensor(value) and value.ndim > 0
        else value.unsqueeze(0).cuda() if torch.is_tensor(value)
        else [value]
        for key, value in sample.items()
    }


def event_key(event: dict) -> tuple:
    return event["tick"], event["lane"], event["note_type"], event.get("raw", "")


def circular_distance(a: int, b: int) -> int:
    raw = abs(a - b)
    return min(raw, 8 - raw)


def best_rotation(predicted: dict[tuple, int], assigned: dict[tuple, int]) -> int:
    overlap = [key for key in predicted if key in assigned]
    if not overlap:
        return 0
    return min(
        range(8),
        key=lambda rotation: sum(
            circular_distance((predicted[key] + rotation) % 8, assigned[key]) for key in overlap
        ),
    )


def choose_template(catalog: list[dict], operator_id: int, endpoint: int, branch_count: int, level: float) -> dict:
    operator = OPERATORS[operator_id]
    tiers = [
        [item for item in catalog if item["branch_count"] == branch_count and item["operators"][0] == operator and item["relative_paths"][0][-1] == endpoint],
        [item for item in catalog if item["operators"][0] == operator and item["relative_paths"][0][-1] == endpoint],
        [item for item in catalog if item["operators"][0] == operator],
        catalog,
    ]
    for options in tiers:
        if options:
            return max(
                options,
                key=lambda item: 3 * item.get("levels", {}).get(f"{level:.1f}", 0) + item.get("count_12_15", 0),
            )
    raise RuntimeError("Star catalog is empty")


def slide_text(start_lane: int, template: dict, is_break: bool, is_ex: bool) -> str:
    modifier = ("b" if is_break else "") + ("x" if is_ex else "")
    branches = []
    for operator, relative_path, duration in zip(
        template["operators"], template["relative_paths"], template["durations"]
    ):
        path = "".join(str(((start_lane - 1 + int(relative)) % 8) + 1) for relative in relative_path)
        branches.append(f"{operator}{path}[{duration['raw']}]")
    return f"{start_lane}{modifier}" + "*".join(branches)


def compile_chart(payload: dict, assigned: dict[tuple, dict], catalog: list[dict]) -> str:
    total_ticks = int(payload["total_ticks"])
    controls: dict[int, list[str]] = {}
    for change in payload["bpm_changes"]:
        controls.setdefault(int(change["tick"]), []).append(f"({change['bpm']:g})")
    controls.setdefault(0, []).append("{192}")

    notes: dict[int, list[str]] = {}
    for event in payload["events"]:
        generated = assigned[event_key(event)]
        lane = generated["lane"] + 1
        modifier = ("b" if event.get("is_break") else "") + ("x" if event.get("is_ex") else "")
        if event["note_type"] == "tap":
            text = f"{lane}{modifier}"
        elif event["note_type"] == "hold":
            text = f"{lane}{modifier}h[{event['duration']['raw']}]"
        else:
            template = choose_template(
                catalog,
                generated["operator"],
                generated["endpoint"],
                generated["branch"] + 1,
                float(payload["level"]),
            )
            text = slide_text(lane, template, bool(event.get("is_break")), bool(event.get("is_ex")))
        notes.setdefault(int(event["tick"]), []).append(text)

    last_tick = max(notes, default=0)
    lines = []
    for measure_start in range(0, last_tick + 192, 192):
        tokens = []
        for tick in range(measure_start, measure_start + 192):
            prefix = "".join(controls.get(tick, []))
            tokens.append(prefix + "/".join(notes.get(tick, [])))
        lines.append(",".join(tokens) + ",")
    return "\n".join(lines) + "\nE"


def main() -> None:
    args = parse_args()
    chart_index = {
        row["chart_id"]: row
        for row in map(json.loads, (PREPARED / "chart_index.jsonl").read_text(encoding="utf-8").splitlines())
    }
    if args.chart_id not in chart_index:
        raise KeyError(args.chart_id)
    source = chart_index[args.chart_id]
    payload = json.loads((PREPARED / source["event_path"]).read_text(encoding="utf-8"))
    dataset = OraclePlanDataset(PREPARED, source["split"])
    indices = [index for index, row in enumerate(dataset.rows) if row["chart_id"] == args.chart_id]
    indices.sort(key=lambda index: dataset.rows[index]["tick_start"])

    model = OfficialPatternArranger().cuda().eval()
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location="cuda", weights_only=False)["model"], strict=False
    )
    operator_bias = operator_calibration_bias(PREPARED).cuda()
    assigned: dict[tuple, dict] = {}
    for index in indices:
        row = dataset.rows[index]
        events = [
            event for event in payload["events"]
            if row["tick_start"] <= event["tick"] < row["tick_end"]
        ][:384]
        sample = move_sample(dataset[index])
        generated = model.generate(sample, first_lane=args.seed_lane - 1, operator_bias=operator_bias)
        predicted = {event_key(event): int(generated["lane"][0, position]) for position, event in enumerate(events)}
        rotation = best_rotation(predicted, {key: value["lane"] for key, value in assigned.items()})
        for position, event in enumerate(events):
            key = event_key(event)
            if key in assigned:
                continue
            assigned[key] = {
                "lane": (int(generated["lane"][0, position]) + rotation) % 8,
                "operator": int(generated["operator"][0, position]),
                "endpoint": int(generated["endpoint"][0, position]),
                "branch": int(generated["branch"][0, position]),
            }

    missing = [event_key(event) for event in payload["events"] if event_key(event) not in assigned]
    if missing:
        raise RuntimeError(f"Unassigned events: {len(missing)}")

    # Two simultaneous events may not occupy the same button.
    by_tick: dict[int, list[tuple]] = {}
    for event in payload["events"]:
        by_tick.setdefault(int(event["tick"]), []).append(event_key(event))
    collision_fixes = 0
    for keys in by_tick.values():
        used = set()
        for key in keys:
            lane = assigned[key]["lane"]
            while lane in used:
                lane = (lane + 4) % 8
                if lane in used:
                    lane = (lane + 1) % 8
                collision_fixes += 1
            assigned[key]["lane"] = lane
            used.add(lane)

    catalog = json.loads((PREPARED / "star_catalog.json").read_text(encoding="utf-8"))
    chart = compile_chart(payload, assigned, catalog)
    maidata_source = DATASET / source["maidata_path"]
    original_fields = {}
    for line in maidata_source.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("&") and "=" in line:
            key, value = line[1:].split("=", 1)
            original_fields[key] = value
    maidata = (
        f"&title={original_fields.get('title', args.chart_id)} Oracle Arranger\n"
        f"&artist={original_fields.get('artist', 'Unknown')}\n"
        f"&first={payload['first_ms'] / 1000:g}\n"
        f"&wholebpm={payload['default_bpm']:g}\n"
        f"&lv_5={payload['level']:g}\n&des_5=Official Pattern Arranger v1\n&inote_5={chart}\n"
    )
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "maidata.txt").write_text(maidata, encoding="utf-8", newline="\n")
    shutil.copy2(DATASET / source["audio_path"], args.output / "track.mp3")
    bg = maidata_source.parent / "bg.png"
    if bg.exists():
        shutil.copy2(bg, args.output / "bg.png")
    operator_counts = Counter(
        OPERATORS[value["operator"]]
        for key, value in assigned.items()
        if next(event for event in payload["events"] if event_key(event) == key)["note_type"] == "slide"
    )
    report = {
        "chart_id": args.chart_id,
        "events": len(payload["events"]),
        "slides": sum(operator_counts.values()),
        "operators": operator_counts,
        "collision_fixes": collision_fixes,
        "output": str(args.output),
    }
    (args.output / "arranger_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, default=dict))


if __name__ == "__main__":
    main()
