from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from train_arranger import make_loader, operator_weights, run_epoch, seed_everything
from trans1.model import Trans1Arranger
from v22.train_arranger import selection_score


ROOT = Path(r"D:\trans")
DEFAULT_CHECKPOINTS = (
    ROOT / "trans1" / "runs" / "trans1_dynamic_v2" / "best.pt",
    ROOT / "v22" / "runs" / "orbit_v22_arranger" / "best.pt",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare arrangers in free-running mode")
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--output", type=Path, default=ROOT / "v22" / "runs" / "arranger_test.json")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.set_defaults(max_train_batches=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    loader = make_loader(args.split, args)
    weights = operator_weights()
    results = []
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = Trans1Arranger().cuda().eval()
        model.load_state_dict(checkpoint["model"])
        with torch.no_grad():
            metrics = run_epoch(
                model,
                loader,
                None,
                None,
                args,
                weights,
                teacher_forcing_ratio=0.0,
            )
        results.append(
            {
                "checkpoint": str(path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "selection_score": selection_score(metrics),
                "metrics": metrics,
            }
        )
        del model
        torch.cuda.empty_cache()
    payload = {
        "split": args.split,
        "windows": len(loader.dataset),
        "evaluation_mode": "free-running patterns and previous deltas",
        "results": results,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
