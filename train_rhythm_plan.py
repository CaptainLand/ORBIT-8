from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler

from maimai_ai.audio_dataset import AudioChartWindowDataset
from maimai_ai.rhythm_model import RhythmPlanModel


PREPARED = Path(r"D:\trans\maimai_finale_dataset\prepared_v2")
AUDIO = Path(r"D:\trans\maimai_finale_dataset\prepared_audio_orbit_v15")
RUN_ROOT = Path(r"D:\trans\maimai_rhythm\runs")
BASE_CHECKPOINT = Path(r"D:\trans\maimai_rhythm\runs\orbit_v15_rhythm_grouped\best.pt")
POSITIVE_WEIGHTS = [4.0, 12.0, 12.0, 1.5, 10.0]


class SongGroupedSampler(Sampler[int]):
    def __init__(self, rows: list[dict]) -> None:
        groups: dict[str, list[int]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(row["song_id"], []).append(index)
        self.groups = list(groups.values())
        self.length = len(rows)

    def __iter__(self):
        groups = self.groups.copy()
        random.shuffle(groups)
        for group in groups:
            random.shuffle(group)
            yield from group

    def __len__(self) -> int:
        return self.length


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--run-name", default=time.strftime("rhythm_%Y%m%d_%H%M%S"))
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--from-scratch", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(split: str, args: argparse.Namespace) -> DataLoader:
    dataset = AudioChartWindowDataset(
        PREPARED,
        AUDIO,
        split,
        augment=split == "train",
        cache_size=16,
        audio_per_tick=True,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=SongGroupedSampler(dataset.rows) if split == "train" else None,
        num_workers=0,
        pin_memory=True,
        drop_last=split == "train",
    )


def targets(chart: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    grouped = chart.reshape(chart.shape[0], 6, 8, chart.shape[-1])
    event = grouped[:, :5].amax(dim=2)
    identity = grouped[:, 0] + grouped[:, 2] + grouped[:, 4]
    count = identity.sum(dim=1).round().long().clamp(0, 2)
    return event, count


class BinaryMetrics:
    def __init__(self) -> None:
        self.counts = Counter()

    def update(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        predicted = torch.sigmoid(logits) >= 0.5
        expected = target > 0.5
        valid = valid[:, None]
        self.counts["tp"] += int((predicted & expected & valid).sum())
        self.counts["fp"] += int((predicted & ~expected & valid).sum())
        self.counts["fn"] += int((~predicted & expected & valid).sum())

    def result(self) -> dict[str, float]:
        precision = self.counts["tp"] / max(1, self.counts["tp"] + self.counts["fp"])
        recall = self.counts["tp"] / max(1, self.counts["tp"] + self.counts["fn"])
        return {
            "event_precision": precision,
            "event_recall": recall,
            "event_f1": 2 * precision * recall / max(1e-12, precision + recall),
        }


class OnsetMetrics:
    def __init__(self) -> None:
        self.counts = Counter()

    def update(self, logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> None:
        probabilities = torch.sigmoid(logits)
        local_maximum = probabilities >= F.max_pool1d(probabilities[:, None], 5, stride=1, padding=2)[:, 0]
        predicted = (probabilities >= 0.35) & local_maximum & valid
        expected = (target >= 0.6) & valid
        expected_near = F.max_pool1d(expected.float()[:, None], 5, stride=1, padding=2)[:, 0] > 0
        predicted_near = F.max_pool1d(predicted.float()[:, None], 5, stride=1, padding=2)[:, 0] > 0
        self.counts["predicted"] += int(predicted.sum())
        self.counts["expected"] += int(expected.sum())
        self.counts["matched_predicted"] += int((predicted & expected_near).sum())
        self.counts["matched_expected"] += int((expected & predicted_near).sum())

    def result(self) -> dict[str, float]:
        precision = self.counts["matched_predicted"] / max(1, self.counts["predicted"])
        recall = self.counts["matched_expected"] / max(1, self.counts["expected"])
        return {
            "onset_precision_t2": precision,
            "onset_recall_t2": recall,
            "onset_f1_t2": 2 * precision * recall / max(1e-12, precision + recall),
        }


def run_epoch(model, loader, optimizer, scaler, args) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = Counter()
    metrics = BinaryMetrics()
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
        event_target, count_target = targets(chart)
        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.float16):
            output = model(audio, controls)
            positive = torch.tensor(POSITIVE_WEIGHTS, device="cuda").view(1, 5, 1)
            event_loss_raw = F.binary_cross_entropy_with_logits(output["event"], event_target, reduction="none")
            event_loss_raw = torch.where(event_target > 0.5, event_loss_raw * positive, event_loss_raw)
            event_loss = (event_loss_raw * valid_ticks[:, None]).sum() / (valid_ticks.sum() * 5).clamp_min(1)
            count_loss_raw = F.cross_entropy(
                output["count"], count_target, reduction="none", weight=torch.tensor([1.0, 8.0, 15.0], device="cuda")
            )
            count_loss = (count_loss_raw * valid_ticks).sum() / valid_ticks.sum().clamp_min(1)
            onset_loss_raw = F.binary_cross_entropy_with_logits(
                output["onset"][:, 0], onset_target, reduction="none"
            )
            onset_focus = torch.where(onset_target > 0.05, 4.0, 1.0)
            onset_loss = (onset_loss_raw * onset_focus * valid_ticks).sum() / valid_ticks.sum().clamp_min(1)
            loss = event_loss + 0.35 * count_loss + 0.5 * onset_loss
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        totals["loss"] += float(loss.detach())
        totals["event_loss"] += float(event_loss.detach())
        totals["count_loss"] += float(count_loss.detach())
        totals["onset_loss"] += float(onset_loss.detach())
        totals["count_correct"] += int(((output["count"].argmax(1) == count_target) & valid_ticks).sum())
        totals["count_total"] += int(valid_ticks.sum())
        totals["batches"] += 1
        metrics.update(output["event"], event_target, valid_ticks)
        onset_metrics.update(output["onset"][:, 0], batch["onset_peak"].cuda(non_blocking=True), valid_ticks)
    batches = max(1, totals["batches"])
    result = {
        "loss": totals["loss"] / batches,
        "event_loss": totals["event_loss"] / batches,
        "count_loss": totals["count_loss"] / batches,
        "onset_loss": totals["onset_loss"] / batches,
        "count_accuracy": totals["count_correct"] / max(1, totals["count_total"]),
        "batches": totals["batches"],
    }
    result.update(metrics.result())
    result.update(onset_metrics.result())
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    run_dir = RUN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")
    train_loader = make_loader("train", args)
    val_loader = make_loader("validation", args)
    model = RhythmPlanModel().cuda()
    if not args.from_scratch:
        incompatible = model.load_state_dict(
            torch.load(BASE_CHECKPOINT, map_location="cuda", weights_only=False)["model"], strict=False
        )
        if "onset_head.weight" in incompatible.missing_keys:
            with torch.no_grad():
                source_heads = torch.tensor([0, 2, 4], device="cuda")
                model.onset_head.weight.copy_(model.event_head.weight[source_heads].mean(dim=0, keepdim=True))
                model.onset_head.bias.copy_(model.event_head.bias[source_heads].mean().view(1))
        print(f"loaded_base={BASE_CHECKPOINT} missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
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
            "train": train,
            "validation": validation,
            "max_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        }
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(json.dumps(record))
        checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "args": vars(args)}
        torch.save(checkpoint, run_dir / "last.pt")
        if validation["onset_f1_t2"] > best_f1:
            best_f1 = validation["onset_f1_t2"]
            stale_epochs = 0
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"early_stopping epoch={epoch} best_validation_onset_f1={best_f1:.6f}")
            break
    print(f"run_dir={run_dir} best_validation_onset_f1={best_f1:.6f}")


if __name__ == "__main__":
    main()
