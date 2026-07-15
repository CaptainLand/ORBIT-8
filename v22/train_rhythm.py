from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from maimai_ai.dynamic_audio_dataset import DynamicAudioChartDataset
from train_rhythm_plan import make_loader, seed_everything
from v2.rhythm_model import OrbitV2RhythmModel
from v2.rhythm_model_16m import OrbitV2RhythmModel16M
from v2.train_16m import run_epoch


ROOT = Path(r"D:\trans")
PREPARED = ROOT / "maimai_finale_dataset" / "prepared_v2"
DYNAMIC = ROOT / "maimai_finale_dataset" / "prepared_v3"
AUDIO = ROOT / "maimai_finale_dataset" / "prepared_audio_orbit_v15"
RUN_ROOT = ROOT / "v22" / "runs"
SOURCE = ROOT / "v2" / "runs" / "orbit_v2_16m_calibrated" / "best.pt"
TEACHER = ROOT / "v2" / "runs" / "orbit_v2_hierarchical_v1" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ORBIT-8 v2.2 rhythm model")
    parser.add_argument("--epochs", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=8e-6)
    parser.add_argument("--inherited-learning-rate", type=float, default=5e-6)
    parser.add_argument("--distill-event-weight", type=float, default=0.12)
    parser.add_argument("--distill-count-weight", type=float, default=0.05)
    parser.add_argument("--distill-onset-weight", type=float, default=0.18)
    parser.add_argument("--distill-confidence", type=float, default=0.35)
    parser.add_argument("--samples-per-epoch", type=int, default=6080)
    parser.add_argument("--run-name", default="orbit_v22_rhythm_16m")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--patience", type=int, default=5)
    return parser.parse_args()


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["onset_f1_t2"]
        + 0.15 * metrics["event_f1"]
        + 0.08 * metrics["dense_f1"]
        + 0.05 * metrics["subdivision_accuracy"]
        - 0.10 * metrics["note_count_relative_error"]
    )


def distill_scale(epoch: int, epochs: int) -> float:
    return max(0.0, 1.0 - (epoch - 1) / max(1.0, epochs * 0.60))


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    run_dir = RUN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "model": "ORBIT-8 v2.2 rhythm 16M",
        "source_checkpoint": str(SOURCE),
        "teacher_checkpoint": str(TEACHER),
        "training_changes": [
            "epoch-wise source resampling",
            "confidence-masked distillation",
            "distillation annealed to zero",
            "density-aware checkpoint selection",
        ],
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    train_dataset = DynamicAudioChartDataset(
        PREPARED,
        AUDIO,
        DYNAMIC,
        samples_per_epoch=args.samples_per_epoch,
        augment=True,
        cache_size=24,
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

    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    model = OrbitV2RhythmModel16M()
    model.load_state_dict(source["model"])
    teacher = OrbitV2RhythmModel()
    teacher.load_state_dict(torch.load(TEACHER, map_location="cpu", weights_only=False)["model"])
    teacher = teacher.cuda().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model = model.cuda()

    new_names = {"sequence_core", "core_in", "core_out"}
    new_parameters, inherited_parameters = [], []
    for name, parameter in model.named_parameters():
        (new_parameters if name.split(".")[0] in new_names else inherited_parameters).append(parameter)
    optimizer = torch.optim.AdamW(
        [
            {"params": new_parameters, "lr": args.learning_rate},
            {"params": inherited_parameters, "lr": args.inherited_learning_rate},
        ],
        weight_decay=0.005,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, args.epochs) / args.epochs)),
    )
    scaler = torch.amp.GradScaler("cuda")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"dynamic_train_samples={len(train_dataset)} fixed_val_windows={len(val_loader.dataset)}")

    best_score = float("-inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        train_dataset.set_epoch(epoch)
        scale = distill_scale(epoch, args.epochs)
        started = time.time()
        train = run_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            args,
            teacher,
            distill_scale=scale,
            distill_confidence=args.distill_confidence,
        )
        with torch.no_grad():
            validation = run_epoch(model, val_loader, None, scaler, args)
        score = selection_score(validation)
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 3),
            "distill_scale": scale,
            "learning_rates": [group["lr"] for group in optimizer.param_groups],
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
