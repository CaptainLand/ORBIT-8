from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from maimai_ai.dynamic_arranger_dataset import DynamicOraclePlanDataset
from train_arranger import make_loader, operator_weights, run_epoch, seed_everything
from trans1.model import Trans1Arranger


ROOT = Path(r"D:\trans")
PREPARED = ROOT / "maimai_finale_dataset" / "prepared_v2"
DYNAMIC = ROOT / "maimai_finale_dataset" / "prepared_v3"
RUN_ROOT = ROOT / "v22" / "runs"
SOURCE = ROOT / "trans1" / "runs" / "trans1_dynamic_v2" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ORBIT-8 v2.2 arranger")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=4e-5)
    parser.add_argument("--samples-per-epoch", type=int, default=24_312)
    parser.add_argument("--teacher-forcing-start", type=float, default=0.90)
    parser.add_argument("--teacher-forcing-end", type=float, default=0.30)
    parser.add_argument("--run-name", default="orbit_v22_arranger")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def teacher_forcing_ratio(epoch: int, args: argparse.Namespace) -> float:
    progress = (epoch - 1) / max(1, args.epochs - 1)
    return args.teacher_forcing_start + progress * (
        args.teacher_forcing_end - args.teacher_forcing_start
    )


def selection_score(metrics: dict[str, float]) -> float:
    pattern_f1 = metrics["pattern_f1"]
    pattern_quality = 0.5 * (pattern_f1["interaction"] + pattern_f1["sweep"])
    return (
        metrics["lane_accuracy"]
        + 0.25 * metrics["operator_accuracy"]
        + 0.15 * metrics["endpoint_accuracy"]
        + 0.05 * metrics["branch_accuracy"]
        + 0.15 * pattern_quality
        - 0.03 * metrics["loss"]
    )


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    run_dir = RUN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "model": "ORBIT-8 v2.2 Trans arranger",
        "source_checkpoint": str(SOURCE),
        "validation_mode": "free-running patterns and previous deltas",
        "training_changes": [
            "epoch-wise source resampling",
            "scheduled sampling for pattern and lane history",
            "pattern precision/F1 checkpoint selection",
            "endpoint and branch validation",
        ],
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    train_dataset = DynamicOraclePlanDataset(
        PREPARED,
        DYNAMIC,
        samples_per_epoch=args.samples_per_epoch,
        corrupt_previous=0.10,
        seed=args.seed,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = make_loader("validation", args)
    model = Trans1Arranger()
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    model.load_state_dict(source["model"])
    model = model.cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, args.epochs) / args.epochs)),
    )
    scaler = torch.amp.GradScaler("cuda")
    weights = operator_weights()
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"dynamic_train_samples={len(train_dataset)} fixed_val_windows={len(val_loader.dataset)}")

    best_score = float("-inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        ratio = teacher_forcing_ratio(epoch, args)
        started = time.time()
        train = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            args,
            weights,
            teacher_forcing_ratio=ratio,
        )
        with torch.no_grad():
            validation = run_epoch(
                model,
                val_loader,
                None,
                scaler,
                args,
                weights,
                teacher_forcing_ratio=0.0,
            )
        score = selection_score(validation)
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 3),
            "teacher_forcing_ratio": ratio,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "selection_score": score,
            "train": train,
            "validation": validation,
            "max_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        }
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(json.dumps(record))
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "selection_score": score,
            "args": config,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if score > best_score:
            best_score = score
            stale_epochs = 0
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= args.patience:
            print(f"early_stopping epoch={epoch} best_selection_score={best_score:.6f}")
            break
    print(f"run_dir={run_dir} best_selection_score={best_score:.6f}")


if __name__ == "__main__":
    main()
