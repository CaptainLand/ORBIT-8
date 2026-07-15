from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .arranger import OraclePlanDataset
from .patterns import MIRROR_MODES


TICKS_PER_MEASURE = 192


class DynamicOraclePlanDataset(OraclePlanDataset):
    """Samples variable-length arranger crops from the prepared-v3 anchor pool."""

    def __init__(
        self,
        prepared_dir: str | Path,
        dynamic_dir: str | Path,
        split: str = "train",
        *,
        samples_per_epoch: int = 16_208,
        corrupt_previous: float = 0.15,
    ) -> None:
        super().__init__(prepared_dir, split, corrupt_previous=corrupt_previous)
        self.dynamic_root = Path(dynamic_dir)
        self.dynamic_config = json.loads(
            (self.dynamic_root / "config.json").read_text(encoding="utf-8")
        )
        dynamic_rows = [
            json.loads(line)
            for line in (self.dynamic_root / "dynamic_index.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(line)["split"] == split
        ]
        self.samples_per_epoch = samples_per_epoch
        self.rows_by_category = {
            category: [row for row in dynamic_rows if row["anchors"][category]]
            for category in self.dynamic_config["sample_categories"]
        }
        self.category_names = tuple(self.dynamic_config["sample_categories"])
        self.category_probabilities = np.asarray(
            [self.dynamic_config["sample_categories"][name] for name in self.category_names],
            dtype=np.float64,
        )
        self.crop_measures = np.asarray(
            [int(value) for value in self.dynamic_config["crop_measures"]], dtype=np.int64
        )
        self.crop_probabilities = np.asarray(
            list(self.dynamic_config["crop_measures"].values()), dtype=np.float64
        )
        self.sample_plan = []
        for _ in range(samples_per_epoch):
            category = str(np.random.choice(self.category_names, p=self.category_probabilities))
            rows = self.rows_by_category[category]
            self.sample_plan.append((category, rows[np.random.randint(len(rows))]))
        self.sample_plan.sort(key=lambda item: (item[1]["chart_id"], item[0]))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        category, source = self.sample_plan[index]
        crop_measures = int(np.random.choice(self.crop_measures, p=self.crop_probabilities))
        crop_ticks = crop_measures * TICKS_PER_MEASURE
        anchor = int(np.random.choice(source["anchors"][category]))
        max_start = max(0, int(source["total_ticks"]) - crop_ticks)
        start = max(0, min(max_start, anchor - crop_ticks // 2))
        start = (start // TICKS_PER_MEASURE) * TICKS_PER_MEASURE
        if np.random.random() < 0.5:
            start = max(0, min(max_start, start + np.random.randint(-1, 2) * TICKS_PER_MEASURE))
            start = (start // TICKS_PER_MEASURE) * TICKS_PER_MEASURE
        row = {
            "chart_id": source["chart_id"],
            "tick_start": start,
            "tick_end": min(int(source["total_ticks"]), start + crop_ticks),
            "level": source["level"],
            "mirror_mode": str(np.random.choice(MIRROR_MODES)),
            "window_id": f"{source['chart_id']}__dynamic_{start}_{crop_measures}_{category}",
        }
        return self._encode_row(row)
