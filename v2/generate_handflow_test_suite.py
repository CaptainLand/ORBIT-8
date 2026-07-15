from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(r"D:\trans")
OUTPUT_ROOT = ROOT / "v2" / "output" / "HandFlow 14.9 Strict Test Suite r5"
SONGS = (
    {
        "title": "Schwarzschild",
        "audio": ROOT / "maimai_finale_dataset" / "songs" / "finale_0020_Schwarzschild" / "track.mp3",
        "bpm": 188.0,
        "seed": 20260711,
        "profile": "tap-dense / high-speed",
    },
    {
        "title": "the EmpErroR",
        "audio": ROOT / "maimai_finale_dataset" / "songs" / "finale_0021_the EmpErroR" / "track.mp3",
        "bpm": 240.0,
        "seed": 20260712,
        "profile": "very-high-speed / hold-slide",
    },
    {
        "title": "Alea jacta est!",
        "audio": ROOT / "maimai_finale_dataset" / "songs" / "finale_0002_Alea jacta est!" / "track.mp3",
        "bpm": 162.0,
        "seed": 20260713,
        "profile": "dense / lower-BPM endurance",
    },
    {
        "title": "FFT",
        "audio": ROOT / "maimai_finale_dataset" / "songs" / "finale_0009_FFT" / "track.mp3",
        "bpm": 180.0,
        "seed": 20260714,
        "profile": "slide-dense",
    },
    {
        "title": "TiamaTF minor",
        "audio": ROOT / "maimai_finale_dataset" / "songs" / "finale_0022_TiamaTF minor" / "track.mp3",
        "bpm": 215.0,
        "seed": 20260715,
        "profile": "hold-slide / validation-song",
    },
)


def summarize(song: dict, report: dict) -> dict:
    handflow = report.get("handflow") or {}
    final = handflow.get("final_assignment") or handflow.get("optimized") or {}
    return {
        "title": song["title"],
        "profile": song["profile"],
        "bpm": song["bpm"],
        "level": report.get("level"),
        "events": report.get("events"),
        "event_types": report.get("event_types"),
        "pattern_segments": report.get("pattern_segments"),
        "handflow_feasible": final.get("feasible"),
        "handflow_cost": final.get("cost"),
        "crossings": final.get("crossings"),
        "rapid_posture_changes": final.get("rapid_posture_changes"),
        "backhand_actions": final.get("backhand_actions"),
        "max_normalized_hand_speed": final.get("max_normalized_hand_speed"),
        "lane_changes": len(handflow.get("changes", [])),
        "handflow_dropped": len(handflow.get("dropped", [])),
        "jack_pattern_share": report.get("jack_pattern_share"),
        "jack_pattern_max_share": report.get("jack_pattern_max_share"),
        "long_eighth_repositioned": report.get("long_eighth_repositioned"),
        "long_eighth_jack_excess": report.get("long_eighth_jack_excess"),
        "irregular_sixteenth_removed": report.get("irregular_sixteenth_removed"),
        "slide_path_tap_conflicts": report.get("slide_path_tap_conflicts"),
        "long_object_sixteenth_conflicts": report.get("long_object_sixteenth_conflicts"),
        "slide_tail_clearance_conflicts": report.get("slide_tail_clearance_conflicts"),
        "max_hand_demand": report.get("max_hand_demand"),
        "output": report.get("output"),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for song in SONGS:
        if not song["audio"].exists():
            raise FileNotFoundError(song["audio"])
        output = OUTPUT_ROOT / song["title"]
        command = [
            sys.executable,
            "-m",
            "v2.generate_16m_handflow",
            str(song["audio"]),
            "--bpm", str(song["bpm"]),
            "--offset", "0",
            "--level", "14.9",
            "--title", f"{song['title']} ORBIT-8 HandFlow Test",
            "--artist", "Unknown",
            "--seed", str(song["seed"]),
            "--output", str(output),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        report = json.loads((output / "generation_report.json").read_text(encoding="utf-8"))
        summaries.append(summarize(song, report))
    payload = {
        "engine": "ORBIT-8 v2 16M HandFlow",
        "difficulty": 14.9,
        "songs": summaries,
    }
    (OUTPUT_ROOT / "suite_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
