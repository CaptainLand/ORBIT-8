from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from maimai_ai.dynamic_audio_dataset import DynamicAudioChartDataset
from train_rhythm_plan import BinaryMetrics, OnsetMetrics, make_loader, seed_everything
from v2.rhythm_model import OrbitV2RhythmModel
from v2.rhythm_model_16m import OrbitV2RhythmModel16M, inherited_module_names
from v2.targets import event_and_count_targets, metrical_targets
from v2.train import POSITIVE_WEIGHTS, SUBDIVISION_WEIGHTS


ROOT = Path(r"D:\trans")
PREPARED = ROOT / "maimai_finale_dataset" / "prepared_v2"
AUDIO = ROOT / "maimai_finale_dataset" / "prepared_audio_orbit_v15"
DYNAMIC = ROOT / "maimai_finale_dataset" / "prepared_v3"
RUN_ROOT = ROOT / "v2" / "runs"
TEACHER_CHECKPOINT = ROOT / "v2" / "runs" / "orbit_v2_hierarchical_v1" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ORBIT-8 v2 16M with distillation")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--schedule-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=7e-5)
    parser.add_argument("--inherited-learning-rate", type=float, default=2e-5)
    parser.add_argument("--samples-per-epoch", type=int, default=4050)
    parser.add_argument("--freeze-epochs", type=int, default=2)
    parser.add_argument("--run-name", default="orbit_v2_16m_dynamic_v3")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--patience", type=int, default=3)
    return parser.parse_args()


def warm_start(model: OrbitV2RhythmModel16M) -> list[str]:
    source = torch.load(TEACHER_CHECKPOINT, map_location="cpu", weights_only=False)["model"]
    target = model.state_dict()
    compatible = {
        key: value for key, value in source.items()
        if key in target and value.shape == target[key].shape
    }
    model.load_state_dict(compatible, strict=False)
    return sorted(compatible)


def set_inherited_trainable(model: OrbitV2RhythmModel16M, enabled: bool) -> None:
    for name in inherited_module_names():
        for parameter in getattr(model, name).parameters():
            parameter.requires_grad_(enabled)


def run_epoch(model, loader, optimizer, scaler, args, teacher=None) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    if teacher is not None:
        teacher.eval()
    totals = Counter()
    event_metrics = BinaryMetrics()
    onset_metrics = OnsetMetrics()
    max_batches = args.max_train_batches if training else args.max_val_batches
    batch_limit = min(len(loader), max_batches) if max_batches is not None else len(loader)
    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader):
        if batch_index >= batch_limit:
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
                output["count"], count_target, reduction="none",
                weight=torch.tensor([1.0, 8.0, 15.0], device="cuda"),
            )
            count_loss = (count_raw * valid_ticks).sum() / valid_ticks.sum().clamp_min(1)
            onset_raw = F.binary_cross_entropy_with_logits(
                output["onset"][:, 0], onset_target, reduction="none"
            )
            onset_focus = torch.where(onset_target > 0.05, 4.0, 1.0)
            onset_loss = (onset_raw * onset_focus * valid_ticks).sum() / valid_ticks.sum().clamp_min(1)
            subdivision_raw = F.cross_entropy(
                output["subdivision"], subdivision_target, reduction="none",
                weight=torch.tensor(SUBDIVISION_WEIGHTS, device="cuda"),
            )
            subdivision_loss = (subdivision_raw * chart_onset).sum() / chart_onset.sum().clamp_min(1)
            accent_raw = F.binary_cross_entropy_with_logits(
                output["accent"][:, 0], accent_target, reduction="none"
            )
            accent_focus = torch.where(chart_onset, 3.0, 0.35)
            accent_loss = (accent_raw * accent_focus * valid_ticks).sum() / valid_ticks.sum().clamp_min(1)
            supervised_loss = (
                event_loss + 0.35 * count_loss + 0.50 * onset_loss
                + 0.35 * subdivision_loss + 0.20 * accent_loss
            )

            distill_event = torch.zeros((), device="cuda")
            distill_count = torch.zeros((), device="cuda")
            distill_onset = torch.zeros((), device="cuda")
            if training and teacher is not None:
                with torch.no_grad():
                    teacher_output = teacher(audio, controls)
                event_teacher = torch.sigmoid(teacher_output["event"])
                event_distill_raw = F.binary_cross_entropy_with_logits(
                    output["event"], event_teacher, reduction="none"
                )
                distill_event = (event_distill_raw * valid_ticks[:, None]).sum() / (
                    valid_ticks.sum() * 5
                ).clamp_min(1)
                count_distill_raw = F.kl_div(
                    F.log_softmax(output["count"], dim=1),
                    F.softmax(teacher_output["count"], dim=1),
                    reduction="none",
                ).sum(dim=1)
                distill_count = (count_distill_raw * valid_ticks).sum() / valid_ticks.sum().clamp_min(1)
                onset_distill_raw = F.binary_cross_entropy_with_logits(
                    output["onset"][:, 0], torch.sigmoid(teacher_output["onset"][:, 0]), reduction="none"
                )
                distill_onset = (onset_distill_raw * valid_ticks).sum() / valid_ticks.sum().clamp_min(1)
            loss = (
                supervised_loss
                + getattr(args, "distill_event_weight", 0.20) * distill_event
                + getattr(args, "distill_count_weight", 0.10) * distill_count
                + getattr(args, "distill_onset_weight", 0.15) * distill_onset
            )

        if training:
            scaler.scale(loss / args.gradient_accumulation).backward()
            should_step = (
                (batch_index + 1) % args.gradient_accumulation == 0
                or batch_index + 1 == batch_limit
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        predicted_subdivision = output["subdivision"].argmax(1)
        dense_expected = (subdivision_target >= 4) & (subdivision_target <= 6) & chart_onset
        dense_predicted = (predicted_subdivision >= 4) & (predicted_subdivision <= 6) & chart_onset
        for name, value in (
            ("loss", loss), ("event_loss", event_loss), ("count_loss", count_loss),
            ("onset_loss", onset_loss), ("subdivision_loss", subdivision_loss),
            ("accent_loss", accent_loss), ("distill_event", distill_event),
            ("distill_count", distill_count), ("distill_onset", distill_onset),
        ):
            totals[name] += float(value.detach())
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
        name: totals[name] / batches for name in (
            "loss", "event_loss", "count_loss", "onset_loss", "subdivision_loss",
            "accent_loss", "distill_event", "distill_count", "distill_onset",
        )
    }
    result.update({
        "count_accuracy": totals["count_correct"] / max(1, totals["count_total"]),
        "subdivision_accuracy": totals["subdivision_correct"] / max(1, totals["subdivision_total"]),
        "dense_precision": dense_precision,
        "dense_recall": dense_recall,
        "dense_f1": 2 * dense_precision * dense_recall / max(1e-12, dense_precision + dense_recall),
        "batches": totals["batches"],
    })
    result.update(event_metrics.result())
    result.update(onset_metrics.result())
    return result


def selection_score(metrics: dict[str, float]) -> float:
    return (
        metrics["onset_f1_t2"]
        + 0.15 * metrics["subdivision_accuracy"]
        + 0.10 * metrics["dense_f1"]
        + 0.20 * metrics["event_f1"]
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
        "model": "ORBIT-8 v2 16M",
        "architecture": "160-dim transient U-Net + 288-dim 12-layer hybrid core",
        "teacher_checkpoint": str(TEACHER_CHECKPOINT),
        "training_data": str(DYNAMIC),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    train_dataset = DynamicAudioChartDataset(
        PREPARED, AUDIO, DYNAMIC,
        samples_per_epoch=args.samples_per_epoch, augment=True, cache_size=16,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    val_loader = make_loader("validation", args)
    model = OrbitV2RhythmModel16M()
    loaded = warm_start(model)
    teacher = OrbitV2RhythmModel()
    teacher.load_state_dict(torch.load(TEACHER_CHECKPOINT, map_location="cpu", weights_only=False)["model"])
    teacher = teacher.cuda().eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model = model.cuda()

    new_names = {"sequence_core", "core_in", "core_out"}
    new_parameters = []
    inherited_parameters = []
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.schedule_epochs)
    scaler = torch.amp.GradScaler("cuda")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"warm_started_tensors={len(loaded)}")
    print(f"dynamic_train_samples={len(train_dataset)} fixed_val_windows={len(val_loader.dataset)}")

    best_score = -1.0
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        frozen = epoch <= args.freeze_epochs
        set_inherited_trainable(model, not frozen)
        started = time.time()
        train = run_epoch(model, train_loader, optimizer, scaler, args, teacher)
        with torch.no_grad():
            validation = run_epoch(model, val_loader, None, scaler, args)
        score = selection_score(validation)
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 3),
            "inherited_frozen": frozen,
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
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "selection_score": score, "args": config,
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
