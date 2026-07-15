from __future__ import annotations

import argparse
import json
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
RUN_ROOT = ROOT / "trans1" / "runs"
WARM_START = RUN_ROOT / "trans1_hybrid_v1" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Trans-1 with dynamic prepared-v3 crops")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--samples-per-epoch", type=int, default=16_208)
    parser.add_argument("--run-name", default="trans1_dynamic_v2")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def selection_score(metrics: dict[str, float]) -> float:
    recall = metrics["pattern_recall"]
    pattern_quality = 0.5 * (recall["interaction"] + recall["sweep"])
    return (
        metrics["lane_accuracy"]
        + 0.35 * metrics["operator_accuracy"]
        + 0.20 * pattern_quality
        - 0.05 * metrics["loss"]
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
        "model": "ORBIT-8 Trans-1 Dynamic v2",
        "training_data": str(DYNAMIC),
        "validation_data": str(PREPARED / "windows.jsonl"),
        "warm_start": str(WARM_START),
        "crop_measures": [8, 12, 16],
        "mirror_modes": 4,
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    train_dataset = DynamicOraclePlanDataset(
        PREPARED,
        DYNAMIC,
        samples_per_epoch=args.samples_per_epoch,
        corrupt_previous=0.15,
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
    checkpoint = torch.load(WARM_START, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model = model.cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.01, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    weights = operator_weights()
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"dynamic_train_samples={len(train_dataset)} fixed_val_windows={len(val_loader.dataset)}")

    best_score = float("-inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train = run_epoch(model, train_loader, optimizer, scaler, args, weights)
        with torch.no_grad():
            validation = run_epoch(model, val_loader, None, scaler, args, weights)
        score = selection_score(validation)
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 3),
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
            break
    print(f"run_dir={run_dir} best_selection_score={best_score:.6f}")


if __name__ == "__main__":
    main()
