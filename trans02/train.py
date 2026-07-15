from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from train_rhythm_plan import make_loader, run_epoch, seed_everything
from trans02.rhythm_model import Trans02RhythmModel


ROOT = Path(r"D:\trans")
RUN_ROOT = ROOT / "trans02" / "runs"
WARM_START = ROOT / "maimai_rhythm" / "runs" / "orbit_v17_rhythm_consensus" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ORBIT-8 Trans-02 timing model")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--run-name", default="trans02_rhythm_hybrid_v1")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--from-scratch", action="store_true")
    return parser.parse_args()


def warm_start(model: Trans02RhythmModel) -> list[str]:
    source = torch.load(WARM_START, map_location="cpu", weights_only=False)["model"]
    target = model.state_dict()
    compatible = {
        key: value for key, value in source.items()
        if key in target and value.shape == target[key].shape
    }
    model.load_state_dict(compatible, strict=False)
    return sorted(compatible)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    run_dir = RUN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "model": "ORBIT-8 Trans-02",
        "architecture": "convolutional transient U-Net + RMSNorm/RoPE/GQA/SwiGLU hybrid core",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    train_loader = make_loader("train", args)
    val_loader = make_loader("validation", args)
    model = Trans02RhythmModel()
    loaded = [] if args.from_scratch else warm_start(model)
    model = model.cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.005, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"warm_started_tensors={len(loaded)}")
    print(f"train_windows={len(train_loader.dataset)} val_windows={len(val_loader.dataset)}")

    best_f1 = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train = run_epoch(model, train_loader, optimizer, scaler, args)
        with torch.no_grad():
            validation = run_epoch(model, val_loader, None, scaler, args)
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 3),
            "learning_rate": optimizer.param_groups[0]["lr"],
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
            "args": config,
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if validation["onset_f1_t2"] > best_f1:
            best_f1 = validation["onset_f1_t2"]
            stale_epochs = 0
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            stale_epochs += 1
        scheduler.step()
        if stale_epochs >= args.patience:
            print(f"early_stopping epoch={epoch} best_validation_onset_f1={best_f1:.6f}")
            break
    print(f"run_dir={run_dir} best_validation_onset_f1={best_f1:.6f}")


if __name__ == "__main__":
    main()
