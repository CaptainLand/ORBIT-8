from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from maimai_ai.dataset import FinaleWindowDataset
from maimai_ai.vae import ChartVAE


PREPARED = Path(r"D:\trans\maimai_finale_dataset\prepared_v1")
GROUPS = {
    "tap": slice(0, 8),
    "break": slice(8, 16),
    "hold_start": slice(16, 24),
    "hold_active": slice(24, 32),
    "slide_head": slice(32, 40),
}


def scores(counts: Counter) -> dict[str, float | int]:
    precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
    recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": counts["tp"],
        "fp": counts["fp"],
        "fn": counts["fn"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    dataset = FinaleWindowDataset(PREPARED, "test", augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = ChartVAE().cuda().eval()
    model.load_state_dict(checkpoint["model"])

    counts = {name: Counter() for name in GROUPS}
    overall = Counter()
    with torch.no_grad():
        for batch in loader:
            target = batch["chart"].cuda(non_blocking=True) > 0.5
            with torch.autocast("cuda", dtype=torch.float16):
                mean, _ = model.encode(target.float())
                predicted = torch.sigmoid(model.decode(mean)) >= args.threshold
            for name, channel_slice in GROUPS.items():
                actual_group = target[:, channel_slice]
                predicted_group = predicted[:, channel_slice]
                counts[name]["tp"] += int((predicted_group & actual_group).sum())
                counts[name]["fp"] += int((predicted_group & ~actual_group).sum())
                counts[name]["fn"] += int((~predicted_group & actual_group).sum())

            event_actual = torch.cat([target[:, 0:8], target[:, 16:24], target[:, 32:40]], dim=1)
            event_predicted = torch.cat(
                [predicted[:, 0:8], predicted[:, 16:24], predicted[:, 32:40]], dim=1
            )
            overall["tp"] += int((event_predicted & event_actual).sum())
            overall["fp"] += int((event_predicted & ~event_actual).sum())
            overall["fn"] += int((~event_predicted & event_actual).sum())

    result = {
        "checkpoint": str(args.checkpoint),
        "split": "test",
        "test_windows": len(dataset),
        "threshold": args.threshold,
        "event_identity": scores(overall),
        "groups": {name: scores(value) for name, value in counts.items()},
    }
    output = args.checkpoint.parent / "test_metrics.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
