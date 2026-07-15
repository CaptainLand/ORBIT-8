from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_rhythm_plan import make_loader, seed_everything
from v2.rhythm_model_16m import OrbitV2RhythmModel16M
from v2.train_16m import run_epoch


ROOT = Path(r"D:\trans")
DEFAULT_CHECKPOINT = ROOT / "v2" / "runs" / "orbit_v2_16m_dynamic_v3" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ORBIT-8 v2 16M")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--output", type=Path, default=ROOT / "v2" / "16m_test_metrics.json")
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--max-val-batches", type=int)
    parser.set_defaults(max_train_batches=None, gradient_accumulation=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    loader = make_loader(args.split, args)
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model = OrbitV2RhythmModel16M().cuda().eval()
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        metrics = run_epoch(model, loader, None, None, args)
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "split": args.split,
        "windows": len(loader.dataset),
        "metrics": metrics,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
