from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np


DATASET = Path(r"D:\trans\maimai_finale_dataset")
PREPARED = DATASET / "prepared_v1"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def main() -> None:
    source = {row["chart_id"]: row for row in read_jsonl(DATASET / "charts.jsonl")}
    charts = read_jsonl(PREPARED / "chart_index.jsonl")
    windows = read_jsonl(PREPARED / "windows.jsonl")
    target_charts = read_jsonl(PREPARED / "chart_index_12_15.jsonl")
    target_windows = read_jsonl(PREPARED / "windows_12_15.jsonl")
    occurrences = read_jsonl(PREPARED / "star_occurrences.jsonl")
    catalog = json.loads((PREPARED / "star_catalog.json").read_text(encoding="utf-8"))
    schema = json.loads((PREPARED / "schema.json").read_text(encoding="utf-8"))

    assert len(charts) == len(source) == 241
    assert len(target_charts) == 88
    assert len(windows) == 2736
    assert len(target_windows) == 1031
    assert len(occurrences) == 11580
    assert len(catalog) == 867
    assert sum(row["count"] for row in catalog) == len(occurrences)
    assert abs(sum(row["frequency"] for row in catalog) - 1.0) < 1e-9

    window_rows = Counter(row["tensor_path"] for row in windows)
    totals = Counter()
    max_simultaneous = 0

    for chart in charts:
        original = source[chart["chart_id"]]
        assert chart["tap_count_parsed"] == original["tap_count"]
        assert chart["hold_count_parsed"] == original["hold_count"]
        assert chart["slide_count_parsed"] == original["slide_count"]
        assert chart["max_simultaneous"] <= 2
        max_simultaneous = max(max_simultaneous, chart["max_simultaneous"])

        event_file = PREPARED / chart["event_path"]
        event_data = json.loads(event_file.read_text(encoding="utf-8"))
        assert len(event_data["events"]) == chart["event_count"]
        assert all(event["note_type"] in {"tap", "hold", "slide"} for event in event_data["events"])
        totals.update(event["note_type"] for event in event_data["events"])

        tensor_path = PREPARED / chart["tensor_path"]
        with np.load(tensor_path) as data:
            tensor = data["chart"]
            times = data["tick_time_ms"]
            valid = data["valid_ticks"]
            starts = data["start_ticks"]
            assert tensor.dtype == np.uint8
            assert tensor.shape[1:] == (len(schema["channels"]), schema["window_ticks"])
            assert times.shape == (tensor.shape[0], schema["window_ticks"] + 1)
            assert valid.shape == starts.shape == (tensor.shape[0],)
            assert tensor.shape[0] == chart["window_count"]
            assert tensor.shape[0] == window_rows[chart["tensor_path"]]
            assert np.all(np.diff(times, axis=1) > 0)
            assert np.all((valid >= 0) & (valid <= schema["window_ticks"]))

    assert totals == Counter({"tap": 89412, "slide": 11580, "hold": 9025})
    result = {
        "status": "ok",
        "charts": len(charts),
        "windows": len(windows),
        "charts_12_15": len(target_charts),
        "windows_12_15": len(target_windows),
        "events": sum(totals.values()),
        "event_types": dict(totals),
        "slide_occurrences": len(occurrences),
        "slide_templates": len(catalog),
        "max_simultaneous": max_simultaneous,
        "tensor_shape_per_window": [len(schema["channels"]), schema["window_ticks"]],
    }
    (PREPARED / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
