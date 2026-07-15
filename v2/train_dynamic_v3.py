from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from maimai_ai.dynamic_audio_dataset import DynamicAudioChartDataset
from train_rhythm_plan import make_loader, seed_everything
from v2.rhythm_model import OrbitV2RhythmModel
from v2.train import run_epoch, selection_score, warm_start


ROOT = Path(r"D:\trans")
PREPARED = ROOT / "maimai_finale_dataset" / "prepared_v2"
AUDIO = ROOT / "maimai_finale_dataset" / "prepared_audio_orbit_v15"
DYNAMIC = ROOT / "maimai_finale_dataset" / "prepared_v3"
RUN_ROOT = ROOT / "v2" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ORBIT-8 v2 with prepared_v3 dynamic crops")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--schedule-epochs", type=int, default=10)
    parser.add_argument("--samples-per-epoch", type=int, default=4052)
    parser.add_argument("--run-name", default="orbit_v2_dynamic_v3_trial")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    run_dir = RUN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "model": "ORBIT-8 v2 dynamic-v3 trial",
        "training_data": str(DYNAMIC),
        "validation_data": "prepared_v2 fixed windows",
        "controlled_warm_start": "Trans-02 best checkpoint",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    train_dataset = DynamicAudioChartDataset(
        PREPARED,
        AUDIO,
        DYNAMIC,
        samples_per_epoch=args.samples_per_epoch,
        augment=True,
        cache_size=16,
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
    model = OrbitV2RhythmModel()
    loaded = warm_start(model)
    model = model.cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=0.005, betas=(0.9, 0.95)
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.schedule_epochs)
    scaler = torch.amp.GradScaler("cuda")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"warm_started_tensors={len(loaded)}")
    print(f"dynamic_train_samples={len(train_dataset)} fixed_val_windows={len(val_loader.dataset)}")

    best_score = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train = run_epoch(model, train_loader, optimizer, scaler, args)
        with torch.no_grad():
            validation = run_epoch(model, val_loader, None, scaler, args)
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
