from __future__ import annotations

import json
import math
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from maimai_ai.simai import (
    TICKS_PER_MEASURE,
    NoteEvent,
    SimaiParseError,
    canonical_slide_key,
    parse_chart,
    parse_fields,
)


DATASET = Path(r"D:\trans\maimai_finale_dataset")
OUTPUT = DATASET / "prepared_v2"
WINDOW_MEASURES = 16
STRIDE_MEASURES = 8
WINDOW_TICKS = WINDOW_MEASURES * TICKS_PER_MEASURE
STRIDE_TICKS = STRIDE_MEASURES * TICKS_PER_MEASURE

CHANNEL_NAMES = [
    *(f"tap_{lane}" for lane in range(1, 9)),
    *(f"break_{lane}" for lane in range(1, 9)),
    *(f"hold_start_{lane}" for lane in range(1, 9)),
    *(f"hold_active_{lane}" for lane in range(1, 9)),
    *(f"slide_head_{lane}" for lane in range(1, 9)),
    *(f"ex_{lane}" for lane in range(1, 9)),
]


def write_jsonl(path: Path, records) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")


def event_dict(event: NoteEvent) -> dict:
    return asdict(event)


def tensorize(events: list[NoteEvent], total_ticks: int) -> np.ndarray:
    length = max(1, math.ceil((total_ticks + 1) / TICKS_PER_MEASURE) * TICKS_PER_MEASURE)
    tensor = np.zeros((len(CHANNEL_NAMES), length), dtype=np.uint8)

    for event in events:
        lane = event.lane - 1
        if event.tick >= length:
            raise ValueError(f"Event outside chart: tick={event.tick}, length={length}")

        if event.note_type == "tap":
            tensor[lane, event.tick] = 1
        elif event.note_type == "hold":
            tensor[16 + lane, event.tick] = 1
            duration = event.duration.duration_ticks if event.duration else 0
            end = min(length, event.tick + max(0, duration) + 1)
            tensor[24 + lane, event.tick + 1:end] = 1
        elif event.note_type == "slide":
            tensor[32 + lane, event.tick] = 1
        else:
            raise ValueError(f"Unsupported note type: {event.note_type}")

        if event.is_break:
            tensor[8 + lane, event.tick] = 1
        if event.is_ex:
            tensor[40 + lane, event.tick] = 1

    return tensor


def padded_tick_times(tick_times: list[float], required_length: int) -> np.ndarray:
    values = np.asarray(tick_times, dtype=np.float64)
    if len(values) >= required_length:
        return values[:required_length]
    step = values[-1] - values[-2] if len(values) > 1 else 1.0
    extra = values[-1] + step * np.arange(1, required_length - len(values) + 1)
    return np.concatenate([values, extra])


def window_starts(total_ticks: int) -> list[int]:
    if total_ticks <= WINDOW_TICKS:
        return [0]
    starts = list(range(0, total_ticks - WINDOW_TICKS + 1, STRIDE_TICKS))
    final_start = math.ceil((total_ticks - WINDOW_TICKS) / TICKS_PER_MEASURE) * TICKS_PER_MEASURE
    if final_start not in starts:
        starts.append(final_start)
    return starts


def nearest_context(events: list[NoteEvent], index: int) -> dict:
    event = events[index]
    previous = next((item for item in reversed(events[:index]) if item.tick < event.tick), None)
    following = next((item for item in events[index + 1:] if item.tick > event.tick), None)
    return {
        "previous_tick_delta": None if previous is None else event.tick - previous.tick,
        "previous_lane": None if previous is None else previous.lane,
        "previous_type": None if previous is None else previous.note_type,
        "next_tick_delta": None if following is None else following.tick - event.tick,
        "next_lane": None if following is None else following.lane,
        "next_type": None if following is None else following.note_type,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"Prepared output already exists: {OUTPUT}")

    temp = DATASET / "prepared_v2.tmp"
    if temp.exists():
        shutil.rmtree(temp)
    events_dir = temp / "events"
    tensors_dir = temp / "tensors"
    events_dir.mkdir(parents=True)
    tensors_dir.mkdir(parents=True)

    source_charts = [
        json.loads(line)
        for line in (DATASET / "charts_12_15.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    field_cache = {}
    chart_index = []
    window_index = []
    star_occurrences = []
    star_material: dict[str, dict] = {}
    qa_counts = Counter()
    skipped_charts = []

    for source in source_charts:
        maidata_path = DATASET / source["maidata_path"]
        fields = field_cache.setdefault(
            maidata_path,
            parse_fields(maidata_path.read_text(encoding="utf-8-sig")),
        )
        try:
            chart = parse_chart(fields, source["difficulty_index"], ignore_touch=True)
        except SimaiParseError as error:
            qa_counts["skipped_incompatible_charts"] += 1
            skipped_charts.append({
                "chart_id": source["chart_id"],
                "song_id": source["song_id"],
                "level": source["level"],
                "reason": str(error),
            })
            continue
        events = sorted(chart.events, key=lambda event: (event.tick, event.lane, event.note_type))
        tensor = tensorize(events, chart.total_ticks)

        event_path = events_dir / f"{source['chart_id']}.json"
        event_payload = {
            "chart_id": source["chart_id"],
            "song_id": source["song_id"],
            "split": source["split"],
            "level": source["level"],
            "level_text": source["level_text"],
            "difficulty_index": source["difficulty_index"],
            "first_ms": chart.first_ms,
            "default_bpm": chart.default_bpm,
            "total_ticks": chart.total_ticks,
            "duration_ms": round(chart.tick_times_ms[-1] - chart.first_ms, 6),
            "bpm_changes": chart.bpm_changes,
            "division_changes": chart.division_changes,
            "events": [event_dict(event) for event in events],
        }
        event_path.write_text(
            json.dumps(event_payload, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

        starts = window_starts(chart.total_ticks)
        required_times = starts[-1] + WINDOW_TICKS + 1
        times = padded_tick_times(chart.tick_times_ms, required_times)
        tensor_windows = []
        time_windows = []
        valid_ticks = []

        for row, start in enumerate(starts):
            end = start + WINDOW_TICKS
            window = np.zeros((len(CHANNEL_NAMES), WINDOW_TICKS), dtype=np.uint8)
            available = max(0, min(WINDOW_TICKS, tensor.shape[1] - start))
            if available:
                window[:, :available] = tensor[:, start:start + available]
            tensor_windows.append(window)
            time_windows.append(times[start:end + 1].astype(np.float32))
            valid = max(0, min(WINDOW_TICKS, chart.total_ticks - start))
            valid_ticks.append(valid)
            window_index.append(
                {
                    "window_id": f"{source['chart_id']}_w{row:03d}",
                    "chart_id": source["chart_id"],
                    "song_id": source["song_id"],
                    "split": source["split"],
                    "level": source["level"],
                    "tensor_path": f"tensors/{source['chart_id']}.npz",
                    "tensor_row": row,
                    "tick_start": start,
                    "tick_end": end,
                    "valid_ticks": valid,
                    "audio_start_ms": round(float(times[start]), 6),
                    "audio_end_ms": round(float(times[end]), 6),
                }
            )

        tensor_path = tensors_dir / f"{source['chart_id']}.npz"
        np.savez_compressed(
            tensor_path,
            chart=np.stack(tensor_windows),
            tick_time_ms=np.stack(time_windows),
            valid_ticks=np.asarray(valid_ticks, dtype=np.int32),
            start_ticks=np.asarray(starts, dtype=np.int32),
        )

        type_counts = Counter(event.note_type for event in events)
        simultaneous = Counter(event.tick for event in events)
        max_simultaneous = max(simultaneous.values(), default=0)
        qa_counts.update(type_counts)
        qa_counts["charts"] += 1
        qa_counts["windows"] += len(starts)
        if max_simultaneous > 2:
            qa_counts["over_two_simultaneous_ticks"] += sum(
                count > 2 for count in simultaneous.values()
            )

        chart_index.append(
            {
                **source,
                "event_path": f"events/{source['chart_id']}.json",
                "tensor_path": f"tensors/{source['chart_id']}.npz",
                "total_ticks": chart.total_ticks,
                "window_count": len(starts),
                "event_count": len(events),
                "tap_count_parsed": type_counts["tap"],
                "hold_count_parsed": type_counts["hold"],
                "slide_count_parsed": type_counts["slide"],
                "max_simultaneous": max_simultaneous,
            }
        )

        for index, event in enumerate(events):
            if event.note_type != "slide":
                continue
            operator_tokens = re.findall(r"pp|qq|[<>^vVpqszw-]", event.raw.split("[")[0])
            is_chain = len(operator_tokens) > len(event.branches)
            key = canonical_slide_key(event)
            occurrence = {
                "chart_id": source["chart_id"],
                "song_id": source["song_id"],
                "split": source["split"],
                "level": source["level"],
                "level_text": source["level_text"],
                "tick": event.tick,
                "time_ms": event.time_ms,
                "lane": event.lane,
                "is_break": event.is_break,
                "is_ex": event.is_ex,
                "branch_count": len(event.branches),
                "canonical_key": key,
                "raw": event.raw,
                "branches": [asdict(branch) for branch in event.branches],
                **nearest_context(events, index),
            }
            star_occurrences.append(occurrence)

            if is_chain:
                qa_counts["chain_slides_excluded_from_catalog"] += 1
                continue

            material = star_material.setdefault(
                key,
                {
                    "canonical_key": key,
                    "count": 0,
                    "count_12_15": 0,
                    "branch_count": len(event.branches),
                    "operators": [branch.operator for branch in event.branches],
                    "relative_paths": [
                        [((lane - event.lane) % 8) for lane in branch.path_lanes]
                        for branch in event.branches
                    ],
                    "durations": [asdict(branch.duration) for branch in event.branches],
                    "levels": Counter(),
                    "break_count": 0,
                    "examples": [],
                },
            )
            material["count"] += 1
            if source["level"] is not None and 12 <= source["level"] <= 15.9:
                material["count_12_15"] += 1
            material["levels"][source["level_text"]] += 1
            material["break_count"] += int(event.is_break)
            if event.raw not in material["examples"] and len(material["examples"]) < 5:
                material["examples"].append(event.raw)

    for material in star_material.values():
        material["levels"] = dict(sorted(material["levels"].items()))
        material["frequency"] = material["count"] / max(1, len(star_occurrences))
        material["frequency_12_15"] = material["count_12_15"] / max(
            1,
            sum(
                1 for occurrence in star_occurrences
                if occurrence["level"] is not None and 12 <= occurrence["level"] <= 15.9
            ),
        )

    catalog = sorted(star_material.values(), key=lambda item: (-item["count"], item["canonical_key"]))
    write_jsonl(temp / "chart_index.jsonl", chart_index)
    write_jsonl(temp / "windows.jsonl", window_index)
    write_jsonl(
        temp / "chart_index_12_15.jsonl",
        (row for row in chart_index if row["level"] is not None and 12 <= row["level"] <= 15.9),
    )
    write_jsonl(
        temp / "windows_12_15.jsonl",
        (row for row in window_index if row["level"] is not None and 12 <= row["level"] <= 15.9),
    )
    write_jsonl(temp / "star_occurrences.jsonl", star_occurrences)
    (temp / "star_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    schema = {
        "version": 2,
        "ticks_per_measure": TICKS_PER_MEASURE,
        "window_measures": WINDOW_MEASURES,
        "stride_measures": STRIDE_MEASURES,
        "window_ticks": WINDOW_TICKS,
        "channels": CHANNEL_NAMES,
        "tensor_dtype": "uint8",
        "tensor_layout": "window,channel,tick",
        "touch_supported": False,
        "touch_policy": "TOUCH and TOUCH HOLD are removed during parsing",
        "level_filter": [12.0, 15.9],
    }
    (temp / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (temp / "qa_summary.json").write_text(
        json.dumps(dict(sorted(qa_counts.items())), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (temp / "skipped_charts.json").write_text(
        json.dumps(skipped_charts, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    temp.rename(OUTPUT)
    print(
        f"Prepared {qa_counts['charts']} charts, {qa_counts['windows']} windows, "
        f"and {len(catalog)} slide templates in {OUTPUT}"
    )


if __name__ == "__main__":
    main()
