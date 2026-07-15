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
from torch import nn
from torch.utils.data import DataLoader

from maimai_ai.dataset import FinaleWindowDataset
from maimai_ai.vae import ChartVAE


PREPARED = Path(r"D:\trans\maimai_finale_dataset\prepared_v1")
RUN_ROOT = Path(r"D:\trans\maimai_vae\runs")
POS_WEIGHTS = [30.0] * 8 + [100.0] * 8 + [100.0] * 8 + [5.0] * 8 + [100.0] * 8 + [1.0] * 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--kl-max", type=float, default=1e-4)
    parser.add_argument("--kl-warmup-steps", type=int, default=4000)
    parser.add_argument("--levels-12-15", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--run-name", default=time.strftime("vae_%Y%m%d_%H%M%S"))
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reconstruction_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    pos_weight = torch.tensor(POS_WEIGHTS, device=target.device, dtype=target.dtype).view(1, -1, 1)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    weights = torch.where(target > 0.5, pos_weight, torch.ones_like(target))
    bce = (bce * weights).mean()

    probability = torch.sigmoid(logits)
    active_channels = target.sum(dim=(0, 2)) > 0
    intersection = (probability * target).sum(dim=(0, 2))
    denominator = probability.sum(dim=(0, 2)) + target.sum(dim=(0, 2))
    dice_per_channel = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    dice = dice_per_channel[active_channels].mean() if active_channels.any() else dice_per_channel.mean()
    kl = -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()
    loss = bce + 0.5 * dice + kl_weight * kl
    return loss, {"loss": float(loss.detach()), "bce": float(bce.detach()), "dice": float(dice.detach()), "kl": float(kl.detach())}


class BinaryMetrics:
    def __init__(self) -> None:
        self.values = {threshold: Counter() for threshold in (0.3, 0.5)}

    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        probability = torch.sigmoid(logits.detach())
        # Event identity channels; modifiers and hold bodies are evaluated separately by loss.
        selected = torch.cat([probability[:, 0:8], probability[:, 16:24], probability[:, 32:40]], dim=1)
        expected = torch.cat([target[:, 0:8], target[:, 16:24], target[:, 32:40]], dim=1) > 0.5
        for threshold, counts in self.values.items():
            predicted = selected >= threshold
            counts["tp"] += int((predicted & expected).sum())
            counts["fp"] += int((predicted & ~expected).sum())
            counts["fn"] += int((~predicted & expected).sum())

    def result(self) -> dict[str, float]:
        result = {}
        for threshold, counts in self.values.items():
            precision = counts["tp"] / max(1, counts["tp"] + counts["fp"])
            recall = counts["tp"] / max(1, counts["tp"] + counts["fn"])
            f1 = 2 * precision * recall / max(1e-12, precision + recall)
            result[f"precision_{threshold}"] = precision
            result[f"recall_{threshold}"] = recall
            result[f"f1_{threshold}"] = f1
        return result


def make_loader(split: str, args: argparse.Namespace) -> DataLoader:
    dataset = FinaleWindowDataset(
        PREPARED,
        split,
        levels_12_15=args.levels_12_15,
        augment=split == "train",
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=split == "train",
        num_workers=0,
        pin_memory=True,
        drop_last=split == "train",
    )


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    global_step: int,
    max_batches: int | None,
) -> tuple[dict[str, float], int]:
    training = optimizer is not None
    model.train(training)
    metrics = BinaryMetrics()
    totals = Counter()
    batches = 0
    if training:
        optimizer.zero_grad(set_to_none=True)

    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        chart = batch["chart"].cuda(non_blocking=True)
        kl_weight = args.kl_max * min(1.0, global_step / max(1, args.kl_warmup_steps))
        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.float16):
            logits, mean, logvar = model(chart)
            loss, parts = reconstruction_loss(logits, chart, mean, logvar, kl_weight)
            scaled_loss = loss / args.grad_accum

        if training:
            scaler.scale(scaled_loss).backward()
            if (batch_index + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        metrics.update(logits, chart)
        for key, value in parts.items():
            totals[key] += value
        batches += 1

    if training and batches % args.grad_accum:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        global_step += 1

    summary = {key: value / max(1, batches) for key, value in totals.items()}
    summary.update(metrics.result())
    summary["batches"] = batches
    return summary, global_step


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    run_dir = RUN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    train_loader = make_loader("train", args)
    val_loader = make_loader("validation", args)
    model = ChartVAE().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={torch.cuda.get_device_name(0)} parameters={parameter_count:,}")
    print(f"train_windows={len(train_loader.dataset)} val_windows={len(val_loader.dataset)}")

    history = []
    best_f1 = -1.0
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_summary, global_step = run_epoch(
            model, train_loader, optimizer, scaler, args, global_step, args.max_train_batches
        )
        with torch.no_grad():
            val_summary, _ = run_epoch(
                model, val_loader, None, scaler, args, global_step, args.max_val_batches
            )
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 3),
            "global_step": global_step,
            "train": train_summary,
            "validation": val_summary,
            "max_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        }
        history.append(record)
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(json.dumps(record))

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "args": vars(args),
        }
        torch.save(checkpoint, run_dir / "last.pt")
        current_f1 = val_summary["f1_0.3"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(checkpoint, run_dir / "best.pt")

    print(f"run_dir={run_dir} best_validation_f1_0.3={best_f1:.6f}")


if __name__ == "__main__":
    main()
