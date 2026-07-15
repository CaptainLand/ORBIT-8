from __future__ import annotations

import math
from pathlib import Path

import joblib
import librosa
import numpy as np


SAMPLE_RATE = 22050
HOP_LENGTH = 256
MODEL_PATH = Path(r"D:\trans\maimai_finale_dataset\bpm_ranker.joblib")
FEATURE_NAMES = [
    "bpm_scaled",
    "log_bpm",
    "log_ratio_raw_fast",
    "log_ratio_raw_coarse",
    "autocorr_beat",
    "autocorr_half",
    "autocorr_double",
    "beat_strength_mean",
    "beat_strength_median",
    "beat_strength_q75",
    "beat_coverage",
    "interval_cv",
    "onset_density",
    "duration_scaled",
    "prior_90",
    "prior_120",
    "prior_150",
    "prior_180",
    "prior_210",
    "prior_240",
]


def _autocorrelation_value(autocorrelation: np.ndarray, bpm: float, multiplier: float = 1.0) -> float:
    lag = SAMPLE_RATE * 60.0 / (HOP_LENGTH * bpm) * multiplier
    if lag < 1 or lag >= len(autocorrelation) - 1:
        return 0.0
    left = int(math.floor(lag))
    alpha = lag - left
    return float(autocorrelation[left] * (1.0 - alpha) + autocorrelation[left + 1] * alpha)


def _phase_from_beats(beat_times: np.ndarray, bpm: float) -> float:
    if len(beat_times) < 2:
        return 0.0
    period = 60.0 / bpm
    index = np.arange(len(beat_times), dtype=np.float64)
    intercept = float(np.polyfit(index, beat_times, 1)[1])
    phase = (intercept + period * 0.5) % period - period * 0.5
    if abs(phase) < 0.05:
        phase = 0.0
    return phase


def extract_candidates(audio_path: str | Path) -> tuple[list[dict], dict]:
    y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    if not np.any(np.abs(y) > 1e-7):
        raise ValueError("Audio contains no detectable signal")
    duration = len(y) / sr
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP_LENGTH, aggregate=np.median)
    coarse_onset = onset[::2]
    raw_fast = float(np.asarray(librosa.feature.tempo(
        onset_envelope=onset, sr=sr, hop_length=HOP_LENGTH, max_tempo=360
    )).reshape(-1)[0])
    raw_coarse = float(np.asarray(librosa.feature.tempo(
        onset_envelope=coarse_onset, sr=sr, hop_length=HOP_LENGTH * 2, max_tempo=360
    )).reshape(-1)[0])
    if raw_fast <= 0 or raw_coarse <= 0:
        raise ValueError("Could not estimate tempo")

    max_lag = int(SAMPLE_RATE * 60 / (HOP_LENGTH * 40)) + 2
    autocorrelation = librosa.autocorrelate(onset, max_size=max_lag).astype(np.float64)
    autocorrelation /= max(1e-9, float(np.max(autocorrelation[1:])))
    times = librosa.times_like(onset, sr=sr, hop_length=HOP_LENGTH)
    onset_scale = max(1e-6, float(np.percentile(onset, 95)))
    onset_density = float(np.mean(onset > np.percentile(onset, 75)))

    seeds = []
    for raw in (raw_fast, raw_coarse):
        for factor in (0.5, 2 / 3, 1.0, 1.5, 2.0):
            bpm = raw * factor
            if 55 <= bpm <= 320:
                seeds.append(bpm)

    candidates = []
    for seed in seeds:
        _, beat_times = librosa.beat.beat_track(
            onset_envelope=onset,
            sr=sr,
            hop_length=HOP_LENGTH,
            bpm=seed,
            tightness=100,
            trim=True,
            units="time",
        )
        beat_times = np.asarray(beat_times, dtype=np.float64)
        if len(beat_times) >= 8:
            slope = float(np.polyfit(np.arange(len(beat_times)), beat_times, 1)[0])
            refined = 60.0 / slope if slope > 0 else seed
            if not 0.92 * seed <= refined <= 1.08 * seed:
                refined = seed
        else:
            refined = seed
        sampled = np.interp(beat_times, times, onset, left=0, right=0) / onset_scale
        intervals = np.diff(beat_times)
        expected_beats = max(1.0, duration * refined / 60.0)
        feature = [
            refined / 300.0,
            math.log(max(refined, 1.0)) / 6.0,
            math.log2(refined / raw_fast),
            math.log2(refined / raw_coarse),
            _autocorrelation_value(autocorrelation, refined),
            _autocorrelation_value(autocorrelation, refined, 2.0),
            _autocorrelation_value(autocorrelation, refined, 0.5),
            float(np.mean(sampled)) if len(sampled) else 0.0,
            float(np.median(sampled)) if len(sampled) else 0.0,
            float(np.percentile(sampled, 75)) if len(sampled) else 0.0,
            min(1.5, len(beat_times) / expected_beats),
            float(np.std(intervals) / max(1e-6, np.mean(intervals))) if len(intervals) else 1.0,
            onset_density,
            min(duration, 300.0) / 300.0,
        ]
        feature.extend(math.exp(-0.5 * ((refined - center) / 18.0) ** 2) for center in (90, 120, 150, 180, 210, 240))
        candidates.append(
            {
                "bpm": refined,
                "offset": _phase_from_beats(beat_times, refined),
                "beat_count": int(len(beat_times)),
                "feature": feature,
                "beat_strength": float(np.mean(sampled)) if len(sampled) else 0.0,
            }
        )

    candidates.sort(key=lambda item: item["bpm"])
    deduplicated = []
    for candidate in candidates:
        existing = next((item for item in deduplicated if abs(item["bpm"] - candidate["bpm"]) / item["bpm"] < 0.018), None)
        if existing is None:
            deduplicated.append(candidate)
        elif candidate["beat_strength"] > existing["beat_strength"]:
            deduplicated[deduplicated.index(existing)] = candidate
    return deduplicated, {
        "duration_seconds": duration,
        "raw_tempo_fast": raw_fast,
        "raw_tempo_coarse": raw_coarse,
    }


def analyze_audio(audio_path: str | Path, model_path: str | Path = MODEL_PATH) -> dict:
    candidates, metadata = extract_candidates(audio_path)
    package = joblib.load(model_path)
    model = package["model"]
    matrix = np.asarray([candidate["feature"] for candidate in candidates], dtype=np.float32)
    probabilities = model.predict_proba(matrix)[:, 1]
    order = np.argsort(probabilities)[::-1]
    ranked = []
    for index in order[:3]:
        candidate = candidates[int(index)]
        ranked.append(
            {
                "bpm": round(float(candidate["bpm"]), 3),
                "offset": round(float(candidate["offset"]), 4),
                "score": round(float(probabilities[index]), 4),
                "beat_count": candidate["beat_count"],
            }
        )
    if not ranked:
        raise ValueError("No tempo candidates were produced")
    if len(ranked) == 1:
        confidence = ranked[0]["score"]
    else:
        margin = max(0.0, ranked[0]["score"] - ranked[1]["score"])
        confidence = min(1.0, 0.55 * ranked[0]["score"] + 0.9 * margin)
    return {
        **metadata,
        "bpm": ranked[0]["bpm"],
        "offset": ranked[0]["offset"],
        "confidence": round(confidence, 4),
        "candidates": ranked,
    }
