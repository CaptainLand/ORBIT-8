from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FinaleWindowDataset(Dataset):
    def __init__(
        self,
        prepared_dir: str | Path,
        split: str,
        *,
        levels_12_15: bool = False,
        augment: bool = False,
        cache_size: int = 12,
    ) -> None:
        self.root = Path(prepared_dir)
        index_name = "windows_12_15.jsonl" if levels_12_15 else "windows.jsonl"
        self.rows = []
        for line in (self.root / index_name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row["split"] == split:
                self.rows.append(row)
        self.augment = augment
        self.cache_size = cache_size
        self.cache: OrderedDict[Path, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def _load_chart(self, relative_path: str) -> np.ndarray:
        path = self.root / relative_path
        if path in self.cache:
            value = self.cache.pop(path)
            self.cache[path] = value
            return value
        with np.load(path) as data:
            value = data["chart"].copy()
        self.cache[path] = value
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return value

    @staticmethod
    def _augment_lanes(x: np.ndarray) -> np.ndarray:
        groups = x.reshape(6, 8, x.shape[-1])
        rotation = np.random.randint(0, 8)
        groups = np.roll(groups, rotation, axis=1)
        if np.random.random() < 0.5:
            mirror = np.asarray([0, 7, 6, 5, 4, 3, 2, 1])
            groups = groups[:, mirror, :]
        return groups.reshape(48, x.shape[-1]).copy()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        chart = self._load_chart(row["tensor_path"])[row["tensor_row"]]
        if self.augment:
            chart = self._augment_lanes(chart)
        return {
            "chart": torch.from_numpy(chart.astype(np.float32, copy=False)),
            "level": torch.tensor(float(row["level"] or 0.0), dtype=torch.float32),
            "window_id": row["window_id"],
        }

