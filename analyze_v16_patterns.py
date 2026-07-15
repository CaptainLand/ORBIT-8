from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from maimai_ai.patterns import MIRROR_MODES, PATTERN_NAMES, detect_pattern_labels, mirror_event


PREPARED = Path(r"D:\trans\maimai_finale_dataset\prepared_v2")


def count_segments(labels: list[int]) -> Counter:
    result = Counter()
    previous = 0
    for label in labels:
        if label and label != previous:
            result[PATTERN_NAMES[label]] += 1
        previous = label
    return result


def main() -> None:
    rows = [json.loads(line) for line in (PREPARED / "chart_index.jsonl").read_text(encoding="utf-8").splitlines()]
    event_counts = Counter()
    segment_counts = Counter()
    split_charts = Counter()
    charts_with_patterns = Counter()
    mirror_mismatches = []

    for row in rows:
        payload = json.loads((PREPARED / row["event_path"]).read_text(encoding="utf-8"))
        events = payload["events"]
        labels = detect_pattern_labels(events)
        split = row["split"]
        split_charts[split] += 1
        local = Counter(PATTERN_NAMES[label] for label in labels if label)
        event_counts.update(local)
        segments = count_segments(labels)
        segment_counts.update(segments)
        if local:
            charts_with_patterns[split] += 1

        expected = Counter(labels)
        for mode in MIRROR_MODES[1:]:
            transformed = [(mirror_event(event, mode), label) for event, label in zip(events, labels)]
            transformed.sort(key=lambda item: (item[0]["tick"], item[0]["lane"], item[0]["note_type"]))
            mirrored_events = [item[0] for item in transformed]
            if Counter(detect_pattern_labels(mirrored_events)) != expected:
                mirror_mismatches.append({"chart_id": row["chart_id"], "mode": mode})

    windows = [json.loads(line) for line in (PREPARED / "windows.jsonl").read_text(encoding="utf-8").splitlines()]
    train_windows = sum(row["split"] == "train" for row in windows)
    report = {
        "version": "ORBIT-8 v1.7.1",
        "source_charts": len(rows),
        "split_charts": dict(split_charts),
        "charts_with_patterns": dict(charts_with_patterns),
        "pattern_events": dict(event_counts),
        "pattern_segments": dict(segment_counts),
        "mirror_modes": list(MIRROR_MODES),
        "train_windows_original": train_windows,
        "train_windows_augmented": train_windows * len(MIRROR_MODES),
        "mirror_consistency_mismatches": mirror_mismatches,
    }
    output = PREPARED / "v171_pattern_qa.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
