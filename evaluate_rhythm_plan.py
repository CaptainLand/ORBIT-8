from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from maimai_ai.audio_dataset import AudioChartWindowDataset
from maimai_ai.rhythm_model import RhythmPlanModel
from train_rhythm_plan import AUDIO, PREPARED, targets


def collect(split: str, checkpoint: Path, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = AudioChartWindowDataset(PREPARED, AUDIO, split, audio_per_tick=True, cache_size=16)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=0)
    model = RhythmPlanModel().cuda().eval()
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=False)["model"])
    predictions, expected, masks = [], [], []
    with torch.inference_mode():
        for batch in loader:
            output = model(batch["audio"].cuda(), batch["controls"].cuda())
            target, _ = targets(batch["chart"])
            predictions.append(torch.sigmoid(output["event"]).cpu().numpy())
            expected.append(target.numpy())
            masks.append(batch["valid_mask"].repeat_interleave(8, dim=1).numpy().astype(bool))
    return np.concatenate(predictions), np.concatenate(expected), np.concatenate(masks)


def tolerant_counts(predicted: np.ndarray, target: np.ndarray, mask: np.ndarray, tolerance: int) -> tuple[int, int, int]:
    tp = fp = fn = 0
    for row in range(predicted.shape[0]):
        valid_length = int(mask[row].sum())
        for channel in range(predicted.shape[1]):
            prediction_ticks = np.flatnonzero(predicted[row, channel, :valid_length])
            target_ticks = np.flatnonzero(target[row, channel, :valid_length] > 0.5)
            prediction_index = target_index = matched = 0
            while prediction_index < len(prediction_ticks) and target_index < len(target_ticks):
                prediction_tick = int(prediction_ticks[prediction_index])
                target_tick = int(target_ticks[target_index])
                if prediction_tick < target_tick - tolerance:
                    fp += 1
                    prediction_index += 1
                elif target_tick < prediction_tick - tolerance:
                    fn += 1
                    target_index += 1
                else:
                    matched += 1
                    prediction_index += 1
                    target_index += 1
            tp += matched
            fp += len(prediction_ticks) - prediction_index
            fn += len(target_ticks) - target_index
    return tp, fp, fn


def score(probabilities: np.ndarray, target: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float]:
    predicted = probabilities >= threshold
    result = {}
    for tolerance in (0, 1, 2, 4):
        tp, fp, fn = tolerant_counts(predicted, target, mask, tolerance)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        result[f"f1_tolerance_{tolerance}"] = 2 * precision * recall / max(1e-12, precision + recall)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    validation = collect("validation", args.checkpoint, args.batch_size)
    candidates = []
    for threshold in np.arange(0.20, 0.71, 0.05):
        metrics = score(*validation, float(threshold))
        candidates.append({"threshold": round(float(threshold), 2), **metrics})
    best = max(candidates, key=lambda row: row["f1_tolerance_2"])
    test = score(*collect("test", args.checkpoint, args.batch_size), best["threshold"])
    print(json.dumps({"checkpoint": str(args.checkpoint), "validation": candidates, "selected": best, "test": test}, indent=2))


if __name__ == "__main__":
    main()
