from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

from .audio_dataset import AudioChartWindowDataset


WINDOW_TICKS = 3072
TICKS_PER_MEASURE = 192


class DynamicAudioChartDataset(AudioChartWindowDataset):
    """Builds aligned training crops from full charts without duplicating audio."""

    def __init__(
        self,
        prepared_dir: str | Path,
        audio_dir: str | Path,
        dynamic_dir: str | Path,
        split: str = "train",
        *,
        samples_per_epoch: int = 4052,
        augment: bool = True,
        cache_size: int = 8,
        seed: int = 20260703,
        cache_bucket_size: int = 96,
    ) -> None:
        super().__init__(
            prepared_dir,
            audio_dir,
            split,
            augment=False,
            cache_size=cache_size,
            audio_per_tick=True,
        )
        self.dynamic_root = Path(dynamic_dir)
        self.dynamic_config = json.loads(
            (self.dynamic_root / "config.json").read_text(encoding="utf-8")
        )
        self.rows = [
            json.loads(line)
            for line in (self.dynamic_root / "dynamic_index.jsonl").read_text(encoding="utf-8").splitlines()
            if json.loads(line)["split"] == split
        ]
        self.samples_per_epoch = samples_per_epoch
        self.dynamic_augment = augment
        self.seed = int(seed)
        self.epoch = 0
        self.cache_bucket_size = max(1, int(cache_bucket_size))
        self.full_chart_cache: OrderedDict[Path, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self.rows_by_category = {
            category: [row for row in self.rows if row["anchors"][category]]
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
        self.sample_plan: list[tuple[str, dict]] = []
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Resample source charts each epoch while retaining bounded cache locality."""
        self.epoch = int(epoch)
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003)
        plan: list[tuple[str, dict]] = []
        for _ in range(self.samples_per_epoch):
            category = str(rng.choice(self.category_names, p=self.category_probabilities))
            rows = self.rows_by_category[category]
            plan.append((category, rows[int(rng.integers(len(rows)))]))
        rng.shuffle(plan)
        buckets = [
            plan[start : start + self.cache_bucket_size]
            for start in range(0, len(plan), self.cache_bucket_size)
        ]
        for bucket in buckets:
            bucket.sort(key=lambda item: (item[1]["song_id"], item[1]["chart_id"]))
        rng.shuffle(buckets)
        self.sample_plan = [item for bucket in buckets for item in bucket]

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + self.epoch * 1_000_003 + int(index) * 97)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _load_full_chart(self, relative_path: str, total_ticks: int) -> tuple[np.ndarray, np.ndarray]:
        path = self.root / relative_path
        if path in self.full_chart_cache:
            value = self.full_chart_cache.pop(path)
            self.full_chart_cache[path] = value
            return value
        with np.load(path) as data:
            windows = data["chart"]
            timing = data["tick_time_ms"]
            valid_ticks = data["valid_ticks"]
            starts = data["start_ticks"]
            chart = np.zeros((48, total_ticks), dtype=np.uint8)
            tick_time_ms = np.zeros(total_ticks + 1, dtype=np.float32)
            for window, times, valid, start in zip(windows, timing, valid_ticks, starts):
                start = int(start)
                valid = min(int(valid), total_ticks - start)
                if valid <= 0:
                    continue
                chart[:, start : start + valid] = np.maximum(
                    chart[:, start : start + valid], window[:, :valid]
                )
                tick_time_ms[start : start + valid + 1] = times[: valid + 1]
        value = (chart, tick_time_ms)
        self.full_chart_cache[path] = value
        while len(self.full_chart_cache) > self.cache_size:
            self.full_chart_cache.popitem(last=False)
        return value

    @staticmethod
    def _padded_crop(
        chart: np.ndarray, tick_time_ms: np.ndarray, start: int, valid_ticks: int
    ) -> tuple[np.ndarray, np.ndarray]:
        result_chart = np.zeros((48, WINDOW_TICKS), dtype=np.uint8)
        result_chart[:, :valid_ticks] = chart[:, start : start + valid_ticks]
        result_times = np.empty(WINDOW_TICKS + 1, dtype=np.float32)
        source_times = tick_time_ms[start : start + valid_ticks + 1]
        result_times[: valid_ticks + 1] = source_times
        tick_ms = float(np.median(np.diff(source_times))) if len(source_times) > 1 else 10.0
        if valid_ticks < WINDOW_TICKS:
            result_times[valid_ticks + 1 :] = source_times[-1] + tick_ms * np.arange(
                1, WINDOW_TICKS - valid_ticks + 1, dtype=np.float32
            )
        return result_chart, result_times

    def _augment_audio(self, audio: np.ndarray) -> np.ndarray:
        result = audio.copy()
        mel = result[: self.n_mels]
        if np.random.random() < 0.45:
            mel *= np.random.uniform(0.92, 1.08)
        if np.random.random() < 0.35:
            tilt = np.linspace(-1.0, 1.0, self.n_mels, dtype=np.float32)[:, None]
            mel += tilt * np.random.uniform(-0.12, 0.12)
        if np.random.random() < 0.30:
            mel += np.random.normal(0.0, 0.012, mel.shape).astype(np.float32)
        if np.random.random() < 0.25:
            width = np.random.randint(2, 10)
            start = np.random.randint(0, self.n_mels - width + 1)
            mel[start : start + width] = 0.0
        if np.random.random() < 0.20:
            width = np.random.randint(12, 49)
            start = np.random.randint(0, WINDOW_TICKS - width + 1)
            mel[:, start : start + width] = 0.0
        return result

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        category, row = self.sample_plan[index]
        rng = self._rng(index)
        crop_measures = int(rng.choice(self.crop_measures, p=self.crop_probabilities))
        crop_ticks = crop_measures * TICKS_PER_MEASURE
        anchor = int(rng.choice(row["anchors"][category]))
        max_start = max(0, int(row["total_ticks"]) - crop_ticks)
        start = max(0, min(max_start, anchor - crop_ticks // 2))
        start = (start // TICKS_PER_MEASURE) * TICKS_PER_MEASURE
        if rng.random() < 0.5:
            start = max(0, min(max_start, start + int(rng.integers(-1, 2)) * TICKS_PER_MEASURE))
            start = (start // TICKS_PER_MEASURE) * TICKS_PER_MEASURE
        valid_ticks = min(crop_ticks, int(row["total_ticks"]) - start)

        full_chart, full_times = self._load_full_chart(row["tensor_path"], int(row["total_ticks"]))
        chart, tick_time_ms = self._padded_crop(full_chart, full_times, start, valid_ticks)
        chart = self._augment_lanes(chart)
        audio = self._aligned_audio(row["song_id"], tick_time_ms)
        if self.dynamic_augment:
            audio = self._augment_audio(audio)
        if valid_ticks < WINDOW_TICKS:
            audio[:, valid_ticks:] = 0.0

        onset_target = np.zeros(WINDOW_TICKS, dtype=np.float32)
        onset_peak = np.zeros(WINDOW_TICKS, dtype=np.float32)
        tick_points = tick_time_ms[:-1]
        for onset in self.consensus_onsets.get(row["song_id"], []):
            time_ms = float(onset["time_ms"])
            if time_ms < tick_points[0] - 30.0 or time_ms > tick_points[valid_ticks - 1] + 30.0:
                continue
            right = int(np.searchsorted(tick_points[:valid_ticks], time_ms))
            candidates = [value for value in (right - 1, right) if 0 <= value < valid_ticks]
            if not candidates:
                continue
            nearest = min(candidates, key=lambda value: abs(float(tick_points[value]) - time_ms))
            if abs(float(tick_points[nearest]) - time_ms) > 30.0:
                continue
            confidence = float(onset["confidence"])
            onset_peak[nearest] = max(onset_peak[nearest], confidence)
            for delta, weight in ((-2, 0.2), (-1, 0.55), (0, 1.0), (1, 0.55), (2, 0.2)):
                target = nearest + delta
                if 0 <= target < valid_ticks:
                    onset_target[target] = max(onset_target[target], confidence * weight)

        valid_mask = np.zeros(384, dtype=np.float32)
        valid_mask[: (valid_ticks + 7) // 8] = 1.0
        valid_chart = chart[:, :valid_ticks]
        event_counts = np.asarray(
            [valid_chart[group * 8 : (group + 1) * 8].sum() for group in range(5)],
            dtype=np.float32,
        )
        primary_notes = max(1.0, event_counts[0] + event_counts[1] + event_counts[2] + event_counts[4])
        measures = max(1.0, valid_ticks / TICKS_PER_MEASURE)
        controls = np.asarray(
            [
                float(row["level"] or 0.0),
                primary_notes / measures,
                event_counts[2] / primary_notes,
                event_counts[4] / primary_notes,
                event_counts[1] / primary_notes,
            ],
            dtype=np.float32,
        )
        return {
            "chart": torch.from_numpy(chart.astype(np.float32, copy=False)),
            "audio": torch.from_numpy(audio),
            "level": torch.tensor(float(row["level"] or 0.0), dtype=torch.float32),
            "valid_mask": torch.from_numpy(valid_mask),
            "controls": torch.from_numpy(controls),
            "onset_target": torch.from_numpy(onset_target),
            "onset_peak": torch.from_numpy(onset_peak),
            "window_id": f"{row['chart_id']}_dynamic_{start}_{crop_measures}",
            "song_id": row["song_id"],
            "sample_kind": category,
            "crop_measures": torch.tensor(crop_measures),
        }
