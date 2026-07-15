from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from v2.generate_handflow_test_suite import SONGS, summarize


ROOT = Path(r"D:\trans")
OUTPUT_ROOT = ROOT / "v22" / "output" / "HandFlow 14.9 Test Suite"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for song in SONGS:
        output = OUTPUT_ROOT / song["title"]
        command = [
            sys.executable,
            "-m",
            "v22.generate_v22",
            str(song["audio"]),
            "--bpm",
            str(song["bpm"]),
            "--offset",
            "0",
            "--level",
            "14.9",
            "--title",
            f"{song['title']} ORBIT-8 v2.2 Test",
            "--artist",
            "Unknown",
            "--seed",
            str(song["seed"]),
            "--output",
            str(output),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        report = json.loads((output / "generation_report.json").read_text(encoding="utf-8"))
        summaries.append(summarize(song, report))
    payload = {
        "engine": "ORBIT-8 v2.2 Calibrated HandFlow",
        "difficulty": 14.9,
        "songs": summaries,
    }
    (OUTPUT_ROOT / "suite_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
