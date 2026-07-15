from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import FinaleWindowDataset


class AudioChartWindowDataset(FinaleWindowDataset):
    def __init__(
        self,
        prepared_dir: str | Path,
        audio_dir: str | Path,
        split: str,
        *,
        levels_12_15: bool = False,
        augment: bool = False,
        cache_size: int = 256,
        audio_per_tick: bool = False,
    ) -> None:
        super().__init__(
            prepared_dir,
            split,
            levels_12_15=levels_12_15,
            augment=augment,
            cache_size=cache_size,
        )
        self.audio_root = Path(audio_dir)
        self.audio_per_tick = audio_per_tick
        config = json.loads((self.audio_root / "config.json").read_text(encoding="utf-8"))
        self.hop_length = int(config["hop_length"])
        self.sample_rate = int(config["sample_rate"])
        self.n_mels = int(config["n_mels"])
        self.audio_mean = np.asarray(config["train_mean"], dtype=np.float32)[:, None]
        self.audio_std = np.asarray(config["train_std"], dtype=np.float32)[:, None]
        self.audio_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.timing_cache: OrderedDict[Path, np.ndarray] = OrderedDict()
        consensus_path = self.root / "consensus_onsets.json"
        self.consensus_onsets = (
            json.loads(consensus_path.read_text(encoding="utf-8"))["onsets"]
            if consensus_path.exists()
            else {}
        )

    def _load_audio(self, song_id: str) -> np.ndarray:
        if song_id in self.audio_cache:
            value = self.audio_cache.pop(song_id)
            self.audio_cache[song_id] = value
            return value
        with np.load(self.audio_root / "songs" / f"{song_id}.npz") as data:
            value = data["log_mel"].astype(np.float32)
        value = (value - self.audio_mean) / self.audio_std
        self.audio_cache[song_id] = value
        while len(self.audio_cache) > self.cache_size:
            self.audio_cache.popitem(last=False)
        return value

    def _load_timing(self, relative_path: str) -> np.ndarray:
        path = self.root / relative_path
        if path in self.timing_cache:
            value = self.timing_cache.pop(path)
            self.timing_cache[path] = value
            return value
        with np.load(path) as data:
            value = data["tick_time_ms"].copy()
        self.timing_cache[path] = value
        while len(self.timing_cache) > self.cache_size:
            self.timing_cache.popitem(last=False)
        return value

    def _aligned_audio(self, song_id: str, tick_time_ms: np.ndarray) -> np.ndarray:
        mel = self._load_audio(song_id)
        if self.audio_per_tick:
            center_times_ms = (tick_time_ms[:-1] + tick_time_ms[1:]) * 0.5
            measure_period = 192.0
            phrase_period = 768.0
        else:
            # The VAE downsamples exactly 8 chart ticks into one latent position.
            center_times_ms = tick_time_ms[4:-1:8]
            measure_period = 24.0
            phrase_period = 96.0
        frame_positions = center_times_ms * (self.sample_rate / (1000.0 * self.hop_length))
        left = np.floor(frame_positions).astype(np.int64)
        left = np.clip(left, 0, mel.shape[1] - 1)
        right = np.minimum(left + 1, mel.shape[1] - 1)
        alpha = (frame_positions - left).astype(np.float32)[None, :]
        aligned = mel[:, left] * (1.0 - alpha) + mel[:, right] * alpha

        positions = np.arange(aligned.shape[1], dtype=np.float32)
        measure_phase = 2.0 * np.pi * positions / measure_period
        phrase_phase = 2.0 * np.pi * positions / phrase_period
        timing = np.stack(
            [np.sin(measure_phase), np.cos(measure_phase), np.sin(phrase_phase), np.cos(phrase_phase)]
        )
        return np.concatenate([aligned, timing], axis=0).astype(np.float32, copy=False)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.rows[index]
        chart_data = self._load_chart(row["tensor_path"])
        chart = chart_data[row["tensor_row"]]
        if self.augment:
            chart = self._augment_lanes(chart)
        tick_time_ms = self._load_timing(row["tensor_path"])[row["tensor_row"]].astype(np.float32)
        audio = self._aligned_audio(row["song_id"], tick_time_ms)
        onset_target = np.zeros(len(tick_time_ms) - 1, dtype=np.float32)
        onset_peak = np.zeros_like(onset_target)
        tick_points = tick_time_ms[:-1]
        for onset in self.consensus_onsets.get(row["song_id"], []):
            time_ms = float(onset["time_ms"])
            if time_ms < tick_points[0] - 30.0 or time_ms > tick_points[-1] + 30.0:
                continue
            right = int(np.searchsorted(tick_points, time_ms))
            candidates = [value for value in (right - 1, right) if 0 <= value < len(tick_points)]
            nearest = min(candidates, key=lambda value: abs(float(tick_points[value]) - time_ms))
            if abs(float(tick_points[nearest]) - time_ms) > 30.0:
                continue
            confidence = float(onset["confidence"])
            onset_peak[nearest] = max(onset_peak[nearest], confidence)
            for delta, weight in ((-2, 0.2), (-1, 0.55), (0, 1.0), (1, 0.55), (2, 0.2)):
                target = nearest + delta
                if 0 <= target < len(onset_target):
                    onset_target[target] = max(onset_target[target], confidence * weight)
        if self.augment and self.audio_per_tick:
            if np.random.random() < 0.2:
                width = np.random.randint(1, 9)
                start = np.random.randint(0, self.n_mels - width + 1)
                audio[start : start + width] = 0.0
            if np.random.random() < 0.2:
                shift = np.random.randint(-4, 5)
                if shift:
                    shifted = np.zeros_like(audio[: self.n_mels])
                    if shift > 0:
                        shifted[shift:] = audio[: self.n_mels - shift]
                    else:
                        shifted[:shift] = audio[-shift : self.n_mels]
                    audio[: self.n_mels] = shifted
        valid_latents = min(384, (int(row["valid_ticks"]) + 7) // 8)
        valid_mask = np.zeros(384, dtype=np.float32)
        valid_mask[:valid_latents] = 1.0
        valid_ticks = int(row["valid_ticks"])
        valid_chart = chart[:, :valid_ticks]
        event_counts = np.asarray(
            [valid_chart[group * 8 : (group + 1) * 8].sum() for group in range(5)], dtype=np.float32
        )
        primary_notes = max(1.0, event_counts[0] + event_counts[1] + event_counts[2] + event_counts[4])
        measures = max(1.0, valid_ticks / 192.0)
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
            "window_id": row["window_id"],
            "song_id": row["song_id"],
        }
