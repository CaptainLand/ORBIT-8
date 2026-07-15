from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import soundfile as sf


ROOT = Path(r"D:\trans\maimai_finale_dataset")
SONGS = ROOT / "songs"
FIELD_RE = re.compile(r"(?m)^&([^=\r\n]+)=(.*)$")
CONTROL_RE = re.compile(r"\([^)]*\)|\{[^}]*\}|\[[^]]*\]")
SLIDE_RE = re.compile(r"[<>^vVpqszw-]")
TOUCH_RE = re.compile(r"(?:[ABDE][1-8]|C)(?:[bhxf]|)", re.IGNORECASE)


def parse_fields(text: str) -> dict[str, str]:
    matches = list(FIELD_RE.finditer(text))
    fields: dict[str, str] = {}
    for index, match in enumerate(matches):
        key = match.group(1).strip()
        start = match.start(2)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        fields[key] = text[start:end].strip()
    return fields


def numeric_level(value: str) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", value)
    return float(match.group()) if match else None


def analyze_chart(chart: str, first_ms: float, default_bpm: float) -> dict:
    chart = re.split(r"(?m)^E\s*$", chart, maxsplit=1)[0]
    tokens = chart.replace("\r", "").replace("\n", "").split(",")
    bpm = default_bpm if default_bpm > 0 else 120.0
    division = 4
    time_ms = first_ms
    counts = Counter()
    slide_shapes = Counter()
    bpm_values: list[float] = []

    for token in tokens:
        for value in re.findall(r"\(([-+]?\d+(?:\.\d+)?)\)", token):
            bpm = float(value)
            bpm_values.append(bpm)
        for value in re.findall(r"\{(\d+)\}", token):
            division = int(value)

        cleaned = CONTROL_RE.sub("", token).strip()
        if cleaned:
            components = [part for part in cleaned.split("/") if part]
            for component in components:
                if TOUCH_RE.search(component):
                    counts["touch"] += 1
                    continue

                lane_heads = re.findall(r"(?<![A-Za-z0-9])[1-8]", component)
                if not lane_heads:
                    continue

                if "h" in component.lower():
                    counts["hold"] += 1
                elif SLIDE_RE.search(component):
                    counts["slide"] += 1
                    for shape in SLIDE_RE.findall(component):
                        slide_shapes[shape] += 1
                else:
                    counts["tap"] += 1
                if "b" in component.lower():
                    counts["break"] += 1
                if "x" in component.lower():
                    counts["ex"] += 1

            if components:
                counts["note_slots"] += 1
                if len(components) > 1:
                    counts["multi_slots"] += 1

        if bpm > 0 and division > 0:
            time_ms += 240000.0 / (bpm * division)

    return {
        "note_slots": counts["note_slots"],
        "multi_slots": counts["multi_slots"],
        "tap_count": counts["tap"],
        "hold_count": counts["hold"],
        "slide_count": counts["slide"],
        "touch_count": counts["touch"],
        "break_count": counts["break"],
        "ex_count": counts["ex"],
        "slide_shapes": dict(sorted(slide_shapes.items())),
        "bpm_values": sorted(set(bpm_values or [default_bpm])),
        "estimated_end_ms": round(time_ms, 3),
        "syntax_balanced": all(
            chart.count(left) == chart.count(right)
            for left, right in (("[", "]"), ("(", ")"), ("{", "}"))
        ),
    }


def main() -> None:
    sources = {}
    for line in (ROOT / "source_manifest.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        sources[record["song_id"]] = record

    songs = []
    charts = []
    issues = []
    overall_shapes = Counter()

    ranked_ids = sorted(
        sources,
        key=lambda song_id: hashlib.sha256(song_id.encode()).hexdigest(),
    )
    train_end = round(len(ranked_ids) * 0.8)
    validation_end = train_end + round(len(ranked_ids) * 0.1)
    split_map = {
        song_id: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, song_id in enumerate(ranked_ids)
    }

    for song_id, source in sorted(sources.items()):
        song_dir = ROOT / source["directory"]
        maidata_path = song_dir / "maidata.txt"
        audio_path = song_dir / "track.mp3"
        text = maidata_path.read_text(encoding="utf-8-sig")
        fields = parse_fields(text)

        first = float(fields.get("first", "0") or 0)
        whole_bpm = float(fields.get("wholebpm", "0") or 0)
        audio_info = sf.info(audio_path)
        split = split_map[song_id]
        song_charts = []

        for difficulty in range(1, 8):
            key = f"inote_{difficulty}"
            if key not in fields or not fields[key].strip():
                continue
            level_text = fields.get(f"lv_{difficulty}", "")
            stats = analyze_chart(fields[key], first * 1000.0, whole_bpm)
            overall_shapes.update(stats["slide_shapes"])
            chart_id = f"{song_id}_d{difficulty}"
            chart_record = {
                "chart_id": chart_id,
                "song_id": song_id,
                "split": split,
                "difficulty_index": difficulty,
                "level": numeric_level(level_text),
                "level_text": level_text,
                "designer": fields.get(f"des_{difficulty}", ""),
                "maidata_path": source["files"]["maidata.txt"]["path"],
                "audio_path": source["files"]["track.mp3"]["path"],
                "first_seconds": first,
                "whole_bpm": whole_bpm,
                **stats,
            }
            charts.append(chart_record)
            song_charts.append(chart_id)

            if stats["touch_count"]:
                issues.append(f"{chart_id}: contains {stats['touch_count']} touch notes")
            if not stats["syntax_balanced"]:
                issues.append(f"{chart_id}: unbalanced brackets")
            if stats["note_slots"] == 0:
                issues.append(f"{chart_id}: empty chart")

        songs.append(
            {
                "song_id": song_id,
                "split": split,
                "title": fields.get("title", ""),
                "artist": fields.get("artist", ""),
                "version": source.get("version", fields.get("version", "FiNALE")),
                "shortid": fields.get("shortid", ""),
                "genre": fields.get("genre", ""),
                "first_seconds": first,
                "whole_bpm": whole_bpm,
                "audio_seconds": round(audio_info.duration, 6),
                "audio_samplerate": audio_info.samplerate,
                "audio_channels": audio_info.channels,
                "directory": source["directory"],
                "charts": song_charts,
            }
        )

    def write_jsonl(path: Path, records: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    write_jsonl(ROOT / "songs.jsonl", songs)
    write_jsonl(ROOT / "charts.jsonl", charts)
    target_charts = [
        chart for chart in charts
        if chart["level"] is not None and 12.0 <= chart["level"] <= 15.9
    ]
    write_jsonl(ROOT / "charts_12_15.jsonl", target_charts)

    split_counts = Counter(song["split"] for song in songs)
    level_counts = Counter(chart["level_text"] for chart in charts)
    total_touch = sum(chart["touch_count"] for chart in charts)
    total_slides = sum(chart["slide_count"] for chart in charts)
    report = [
        "# ORBIT-8 Multiversion Dataset Report",
        "",
        f"- Songs: {len(songs)}",
        f"- Charts: {len(charts)}",
        f"- Level 12-15 charts: {len(target_charts)}",
        f"- Splits by song: train={split_counts['train']}, validation={split_counts['validation']}, test={split_counts['test']}",
        f"- Slides: {total_slides}",
        f"- Touch notes: {total_touch}",
        f"- QA issues: {len(issues)}",
        "",
        "## Levels",
        "",
        "| Level | Charts |",
        "| --- | ---: |",
        *[f"| {level or '(blank)'} | {count} |" for level, count in sorted(level_counts.items())],
        "",
        "## Slide Shapes",
        "",
        "| Shape | Count |",
        "| --- | ---: |",
        *[f"| `{shape}` | {count} |" for shape, count in overall_shapes.most_common()],
        "",
        "## Issues",
        "",
        *(issues or ["None."]),
        "",
    ]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Indexed {len(songs)} songs and {len(charts)} charts; issues={len(issues)}")


if __name__ == "__main__":
    main()
