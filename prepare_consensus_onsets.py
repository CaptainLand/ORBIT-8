from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


PREPARED = Path(r"D:\trans\maimai_finale_dataset\prepared_v2")
CLUSTER_TOLERANCE_MS = 18.0


def main() -> None:
    rows = [json.loads(line) for line in (PREPARED / "chart_index.jsonl").read_text(encoding="utf-8").splitlines()]
    by_song: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_song[row["song_id"]].append(row)

    songs = {}
    total_clusters = 0
    multi_supported = 0
    for song_id, charts in by_song.items():
        observations = []
        for chart in charts:
            payload = json.loads((PREPARED / chart["event_path"]).read_text(encoding="utf-8"))
            times = sorted({
                round(float(event["time_ms"]), 3)
                for event in payload["events"]
                if event["note_type"] in {"tap", "hold", "slide"}
            })
            observations.extend((time_ms, chart["chart_id"]) for time_ms in times)
        observations.sort()

        clusters = []
        for time_ms, chart_id in observations:
            if not clusters or abs(time_ms - statistics.median(clusters[-1]["times"])) > CLUSTER_TOLERANCE_MS:
                clusters.append({"times": [time_ms], "charts": {chart_id}})
            else:
                clusters[-1]["times"].append(time_ms)
                clusters[-1]["charts"].add(chart_id)

        onsets = []
        for cluster in clusters:
            support = len(cluster["charts"])
            ratio = support / len(charts)
            confidence = 0.65 + 0.35 * ratio
            onsets.append({
                "time_ms": round(float(statistics.median(cluster["times"])), 3),
                "support": support,
                "chart_count": len(charts),
                "confidence": round(confidence, 4),
            })
            multi_supported += int(support >= 2)
        total_clusters += len(onsets)
        songs[song_id] = onsets

    report = {
        "version": 1,
        "cluster_tolerance_ms": CLUSTER_TOLERANCE_MS,
        "songs": len(songs),
        "onset_clusters": total_clusters,
        "multi_chart_supported_clusters": multi_supported,
        "onsets": songs,
    }
    output = PREPARED / "consensus_onsets.json"
    output.write_text(json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "onsets"}, indent=2))


if __name__ == "__main__":
    main()
