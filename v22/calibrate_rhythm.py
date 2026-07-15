from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train_rhythm_plan import make_loader, seed_everything
from v2.rhythm_model_16m import OrbitV2RhythmModel16M
from v2.targets import event_and_count_targets


ROOT = Path(r"D:\trans")
SOURCE = ROOT / "v2" / "runs" / "orbit_v2_16m_calibrated" / "best.pt"
OUTPUT = ROOT / "v22" / "runs" / "orbit_v22_rhythm_calibrated.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate ORBIT-8 v2.2 output logits")
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.set_defaults(max_train_batches=None, max_val_batches=None, gradient_accumulation=1)
    return parser.parse_args()


def binary_counts(probability: np.ndarray, target: np.ndarray, threshold: float) -> tuple[int, int, int]:
    predicted = probability >= threshold
    return (
        int(np.logical_and(predicted, target).sum()),
        int(np.logical_and(predicted, np.logical_not(target)).sum()),
        int(np.logical_and(np.logical_not(predicted), target).sum()),
    )


def f1(counts: tuple[int, int, int]) -> float:
    true_positive, false_positive, false_negative = counts
    return 2.0 * true_positive / max(1, 2 * true_positive + false_positive + false_negative)


def onset_f1(
    batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]], threshold: float
) -> float:
    predicted_count = expected_count = matched_predicted = matched_expected = 0
    for probability, target, valid in batches:
        local_maximum = probability >= F.max_pool1d(
            probability[:, None], 5, stride=1, padding=2
        )[:, 0]
        predicted = (probability >= threshold) & local_maximum & valid
        expected = (target >= 0.6) & valid
        expected_near = F.max_pool1d(
            expected.float()[:, None], 5, stride=1, padding=2
        )[:, 0] > 0
        predicted_near = F.max_pool1d(
            predicted.float()[:, None], 5, stride=1, padding=2
        )[:, 0] > 0
        predicted_count += int(predicted.sum())
        expected_count += int(expected.sum())
        matched_predicted += int((predicted & expected_near).sum())
        matched_expected += int((expected & predicted_near).sum())
    precision = matched_predicted / max(1, predicted_count)
    recall = matched_expected / max(1, expected_count)
    return 2.0 * precision * recall / max(1e-12, precision + recall)


def logit(value: float) -> float:
    return math.log(value / (1.0 - value))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    checkpoint = torch.load(args.source, map_location="cpu", weights_only=False)
    model = OrbitV2RhythmModel16M().cuda().eval()
    model.load_state_dict(checkpoint["model"])
    loader = make_loader("validation", args)

    event_probabilities: list[list[np.ndarray]] = [[] for _ in range(5)]
    event_targets: list[list[np.ndarray]] = [[] for _ in range(5)]
    onset_batches: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    with torch.inference_mode():
        for batch in loader:
            audio = batch["audio"].cuda(non_blocking=True)
            controls = batch["controls"].cuda(non_blocking=True)
            valid = batch["valid_mask"].repeat_interleave(8, dim=1).bool()
            chart = batch["chart"].cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(audio, controls)
            event_target, _ = event_and_count_targets(chart)
            event_probability = torch.sigmoid(output["event"]).cpu().numpy()
            event_target = event_target.cpu().numpy() > 0.5
            valid_numpy = valid.numpy()
            for event_index in range(5):
                event_probabilities[event_index].append(event_probability[:, event_index][valid_numpy])
                event_targets[event_index].append(event_target[:, event_index][valid_numpy])
            onset_batches.append(
                (
                    torch.sigmoid(output["onset"][:, 0]).float().cpu(),
                    batch["onset_peak"].float().cpu(),
                    valid,
                )
            )

    event_probability_arrays = [np.concatenate(values) for values in event_probabilities]
    event_target_arrays = [np.concatenate(values) for values in event_targets]
    event_grid = np.arange(0.20, 0.801, 0.025)
    event_thresholds = []
    event_class_f1 = []
    for probability, target in zip(event_probability_arrays, event_target_arrays):
        candidates = [(float(threshold), f1(binary_counts(probability, target, threshold))) for threshold in event_grid]
        threshold, score = max(candidates, key=lambda item: item[1])
        event_thresholds.append(threshold)
        event_class_f1.append(score)

    onset_candidates = [
        (float(threshold), onset_f1(onset_batches, float(threshold)))
        for threshold in np.arange(0.20, 0.601, 0.01)
    ]
    onset_threshold, onset_score = max(onset_candidates, key=lambda item: item[1])

    state = {key: value.clone() for key, value in checkpoint["model"].items()}
    event_bias = state["event_head.bias"]
    for index, threshold in enumerate(event_thresholds):
        event_bias[index] += -logit(threshold)
    state["onset_head.bias"][0] += logit(0.35) - logit(onset_threshold)
    calibration = {
        "split": "validation",
        "windows": len(loader.dataset),
        "event_thresholds": event_thresholds,
        "event_class_f1": event_class_f1,
        "onset_threshold": onset_threshold,
        "onset_f1": onset_score,
        "method": "thresholds absorbed into output-head biases",
    }
    payload = {
        "model": state,
        "epoch": "v2.2-logit-calibration",
        "selection_score": checkpoint.get("selection_score"),
        "args": {
            "model": "ORBIT-8 v2.2 rhythm 16M calibrated",
            "source_checkpoint": str(args.source),
            "calibration": calibration,
        },
    }
    torch.save(payload, args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps(calibration, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(calibration))


if __name__ == "__main__":
    main()
