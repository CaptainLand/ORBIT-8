from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


PREPARED = Path(r"D:\trans\maimai_finale_dataset\prepared_v2")


def main() -> None:
    rows = [json.loads(line) for line in (PREPARED / "windows.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [row for row in rows if row["split"] == "train" and row["level"]]
    cache: dict[str, np.ndarray] = {}
    values: dict[float, list[list[float]]] = defaultdict(list)
    for row in rows:
        path = row["tensor_path"]
        if path not in cache:
            with np.load(PREPARED / path) as data:
                cache[path] = data["chart"].copy()
        chart = cache[path][row["tensor_row"]]
        valid_ticks = int(row["valid_ticks"])
        chart = chart[:, :valid_ticks]
        counts = np.asarray([chart[group * 8 : (group + 1) * 8].sum() for group in range(5)], dtype=np.float64)
        primary = max(1.0, counts[0] + counts[1] + counts[2] + counts[4])
        measures = max(1.0, valid_ticks / 192.0)
        values[float(row["level"])].append(
            [primary / measures, counts[2] / primary, counts[4] / primary, counts[1] / primary]
        )

    records = []
    for level in sorted(values):
        array = np.asarray(values[level])
        records.append(
            {
                "level": level,
                "windows": len(array),
                "density_per_measure": float(np.median(array[:, 0])),
                "hold_ratio": float(np.median(array[:, 1])),
                "slide_ratio": float(np.median(array[:, 2])),
                "break_ratio": float(np.median(array[:, 3])),
            }
        )
    output = PREPARED / "control_defaults.json"
    output.write_text(json.dumps({"source": "train split medians", "levels": records}, indent=2) + "\n", encoding="utf-8")
    print(f"output={output} levels={len(records)} range={records[0]['level']}-{records[-1]['level']}")


if __name__ == "__main__":
    main()
