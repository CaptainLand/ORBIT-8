from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_rhythm_plan import make_loader, seed_everything
from v2.rhythm_model import OrbitV2RhythmModel
from v2.train import run_epoch


ROOT = Path(r"D:\trans")
DEFAULT_CHECKPOINT = ROOT / "v2" / "runs" / "orbit_v2_hierarchical_v1" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ORBIT-8 v2 rhythm planner")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "v2" / "test_metrics.json")
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--max-val-batches", type=int)
    parser.set_defaults(max_train_batches=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    loader = make_loader(args.split, args)
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model = OrbitV2RhythmModel().cuda().eval()
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        metrics = run_epoch(model, loader, None, None, args)
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "windows": len(loader.dataset),
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
