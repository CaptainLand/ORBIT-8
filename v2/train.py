from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

from train_rhythm_plan import BinaryMetrics, OnsetMetrics, make_loader, seed_everything
from v2.rhythm_model import OrbitV2RhythmModel
from v2.targets import event_and_count_targets, metrical_targets


ROOT = Path(r"D:\trans")
RUN_ROOT = ROOT / "v2" / "runs"
WARM_START = ROOT / "trans02" / "runs" / "trans02_rhythm_hybrid_v1" / "best.pt"
POSITIVE_WEIGHTS = [4.0, 12.0, 12.0, 1.5, 10.0]
SUBDIVISION_WEIGHTS = [0.2, 0.8, 1.0, 1.3, 1.5, 1.8, 2.0, 1.2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ORBIT-8 v2 hierarchical timing model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=6e-5)
    parser.add_argument("--run-name", default="orbit_v2_hierarchical_v1")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--from-scratch", action="store_true")
    return parser.parse_args()


def warm_start(model: OrbitV2RhythmModel) -> list[str]:
    source = torch.load(WARM_START, map_location="cpu", weights_only=False)["model"]
    target = model.state_dict()
    compatible = {
        key: value for key, value in source.items()
        if key in target and value.shape == target[key].shape
    }
    model.load_state_dict(compatible, strict=False)
    return sorted(compatible)


def run_epoch(model, loader, optimizer, scaler, args) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = Counter()
    event_metrics = BinaryMetrics()
    onset_metrics = OnsetMetrics()
    max_batches = args.max_train_batches if training else args.max_val_batches
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        audio = batch["audio"].cuda(non_blocking=True)
        controls = batch["controls"].cuda(non_blocking=True)
        chart = batch["chart"].cuda(non_blocking=True)
        valid_latent = batch["valid_mask"].cuda(non_blocking=True)
        valid_ticks = valid_latent.repeat_interleave(8, dim=1) > 0.5
        onset_target = batch["onset_target"].cuda(non_blocking=True)
        event_target, count_target = event_and_count_targets(chart)
        subdivision_target, accent_target, chart_onset = metrical_targets(
            event_target, count_target, valid_ticks
        )

        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.float16):
            output = model(audio, controls)
            positive = torch.tensor(POSITIVE_WEIGHTS, device="cuda").view(1, 5, 1)
            event_raw = F.binary_cross_entropy_with_logits(
                output["event"], event_target, reduction="none"
            )
            event_raw = torch.where(event_target > 0.5, event_raw * positive, event_raw)
            event_loss = (event_raw * valid_ticks[:, None]).sum() / (
                valid_ticks.sum() * 5
            ).clamp_min(1)

            count_raw = F.cross_entropy(
                output["count"],
                count_target,
                reduction="none",
                weight=torch.tensor([1.0, 8.0, 15.0], device="cuda"),
            )
            count_loss = (count_raw * valid_ticks).sum() / valid_ticks.sum().clamp_min(1)

            onset_raw = F.binary_cross_entropy_with_logits(
                output["onset"][:, 0], onset_target, reduction="none"
            )
            onset_focus = torch.where(onset_target > 0.05, 4.0, 1.0)
            onset_loss = (onset_raw * onset_focus * valid_ticks).sum() / valid_ticks.sum().clamp_min(1)

            subdivision_raw = F.cross_entropy(
                output["subdivision"],
                subdivision_target,
                reduction="none",
                weight=torch.tensor(SUBDIVISION_WEIGHTS, device="cuda"),
            )
            subdivision_loss = (subdivision_raw * chart_onset).sum() / chart_onset.sum().clamp_min(1)

            accent_raw = F.binary_cross_entropy_with_logits(
                output["accent"][:, 0], accent_target, reduction="none"
            )
            accent_focus = torch.where(chart_onset, 3.0, 0.35)
            accent_loss = (accent_raw * accent_focus * valid_ticks).sum() / valid_ticks.sum().clamp_min(1)

            loss = (
                event_loss
                + 0.35 * count_loss
                + 0.50 * onset_loss
                + 0.35 * subdivision_loss
                + 0.20 * accent_loss
            )

        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        predicted_subdivision = output["subdivision"].argmax(1)
        dense_expected = (subdivision_target >= 4) & (subdivision_target <= 6) & chart_onset
        dense_predicted = (predicted_subdivision >= 4) & (predicted_subdivision <= 6) & chart_onset
        totals["loss"] += float(loss.detach())
        totals["event_loss"] += float(event_loss.detach())
        totals["count_loss"] += float(count_loss.detach())
        totals["onset_loss"] += float(onset_loss.detach())
        totals["subdivision_loss"] += float(subdivision_loss.detach())
        totals["accent_loss"] += float(accent_loss.detach())
        totals["count_correct"] += int(((output["count"].argmax(1) == count_target) & valid_ticks).sum())
        totals["count_total"] += int(valid_ticks.sum())
        totals["subdivision_correct"] += int(((predicted_subdivision == subdivision_target) & chart_onset).sum())
        totals["subdivision_total"] += int(chart_onset.sum())
        totals["dense_tp"] += int((dense_expected & dense_predicted).sum())
        totals["dense_fp"] += int((~dense_expected & dense_predicted).sum())
        totals["dense_fn"] += int((dense_expected & ~dense_predicted).sum())
        totals["batches"] += 1
        event_metrics.update(output["event"], event_target, valid_ticks)
        onset_metrics.update(output["onset"][:, 0], batch["onset_peak"].cuda(non_blocking=True), valid_ticks)

    batches = max(1, totals["batches"])
    dense_precision = totals["dense_tp"] / max(1, totals["dense_tp"] + totals["dense_fp"])
    dense_recall = totals["dense_tp"] / max(1, totals["dense_tp"] + totals["dense_fn"])
    result = {
        "loss": totals["loss"] / batches,
        "event_loss": totals["event_loss"] / batches,
        "count_loss": totals["count_loss"] / batches,
        "onset_loss": totals["onset_loss"] / batches,
        "subdivision_loss": totals["subdivision_loss"] / batches,
        "accent_loss": totals["accent_loss"] / batches,
        "count_accuracy": totals["count_correct"] / max(1, totals["count_total"]),
        "subdivision_accuracy": totals["subdivision_correct"] / max(1, totals["subdivision_total"]),
        "dense_precision": dense_precision,
        "dense_recall": dense_recall,
        "dense_f1": 2 * dense_precision * dense_recall / max(1e-12, dense_precision + dense_recall),
        "batches": totals["batches"],
    }
    result.update(event_metrics.result())
    result.update(onset_metrics.result())
    return result


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["onset_f1_t2"]
        + 0.15 * metrics["subdivision_accuracy"]
        + 0.10 * metrics["dense_f1"]
        + 0.10 * metrics["event_f1"]
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
        "model": "ORBIT-8 v2",
        "architecture": "hierarchical hybrid Transformer rhythm planner",
        "selection": "onset_f1 + subdivision_accuracy + dense_f1 + event_f1",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    train_loader = make_loader("train", args)
    val_loader = make_loader("validation", args)
    model = OrbitV2RhythmModel()
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
            print(f"early_stopping epoch={epoch} best_selection_score={best_score:.6f}")
            break
    print(f"run_dir={run_dir} best_selection_score={best_score:.6f}")


if __name__ == "__main__":
    main()
