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

from maimai_ai.audio_dataset import AudioChartWindowDataset
from maimai_ai.dataset import FinaleWindowDataset
from maimai_ai.diffusion import cosine_schedule
from maimai_ai.mug_diffusion import MugInspiredAudioChartDiffusion
from maimai_ai.vae import ChartVAE


PREPARED = Path(r"D:\trans\maimai_finale_dataset\prepared_v1")
AUDIO = Path(r"D:\trans\maimai_finale_dataset\prepared_audio_mug_v1")
VAE_CHECKPOINT = Path(r"D:\trans\maimai_vae\runs\finale_vae_v1\best.pt")
RUN_ROOT = Path(r"D:\trans\maimai_audio_diffusion\runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--control-dropout", type=float, default=0.5)
    parser.add_argument("--levels-12-15", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--run-name", default=time.strftime("mug_maimai_%Y%m%d_%H%M%S"))
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_audio_loader(split: str, args: argparse.Namespace, augment: bool | None = None) -> DataLoader:
    dataset = AudioChartWindowDataset(
        PREPARED,
        AUDIO,
        split,
        levels_12_15=args.levels_12_15,
        augment=(split == "train") if augment is None else augment,
        audio_per_tick=True,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=split == "train" and augment is not False,
        num_workers=0,
        pin_memory=True,
        drop_last=split == "train" and augment is not False,
    )


@torch.inference_mode()
def calculate_latent_stats(vae: ChartVAE, args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = FinaleWindowDataset(PREPARED, "train", levels_12_15=args.levels_12_15, augment=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)
    total = torch.zeros(16, device="cuda", dtype=torch.float64)
    square_total = torch.zeros_like(total)
    count = 0
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            latent, _ = vae.encode(batch["chart"].cuda(non_blocking=True))
        latent = latent.double()
        total += latent.sum(dim=(0, 2))
        square_total += latent.square().sum(dim=(0, 2))
        count += latent.shape[0] * latent.shape[2]
    mean = total / count
    std = (square_total / count - mean.square()).clamp_min(1e-6).sqrt()
    return mean.float().view(1, 16, 1), std.float().view(1, 16, 1)


def rhythm_targets(chart: torch.Tensor) -> torch.Tensor:
    grouped = chart.reshape(chart.shape[0], 6, 8, 384, 8)
    return grouped[:, :5].sum(dim=(2, 4)).clamp_max(1.0)


def masked_smooth_l1(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = F.smooth_l1_loss(prediction, target, beta=0.02, reduction="none")
    return (error * mask[:, None]).sum() / (mask.sum() * prediction.shape[1]).clamp_min(1.0)


def masked_rhythm_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positive_weights = torch.tensor([2.0, 5.0, 5.0, 1.5, 4.0], device=logits.device).view(1, 5, 1)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = torch.where(target > 0.5, loss * positive_weights, loss)
    return (loss * mask[:, None]).sum() / (mask.sum() * logits.shape[1]).clamp_min(1.0)


class RhythmMetrics:
    def __init__(self) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def update(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        predicted = torch.sigmoid(logits.detach()) >= 0.5
        expected = target > 0.5
        valid = mask[:, None] > 0.5
        self.tp += int((predicted & expected & valid).sum())
        self.fp += int((predicted & ~expected & valid).sum())
        self.fn += int((~predicted & expected & valid).sum())

    def result(self) -> dict[str, float]:
        precision = self.tp / max(1, self.tp + self.fp)
        recall = self.tp / max(1, self.tp + self.fn)
        return {
            "rhythm_precision": precision,
            "rhythm_recall": recall,
            "rhythm_f1": 2 * precision * recall / max(1e-12, precision + recall),
        }


def run_epoch(
    model: nn.Module,
    vae: ChartVAE,
    loader: DataLoader,
    schedule: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = Counter()
    rhythm_metrics = RhythmMetrics()
    batches = 0
    max_batches = args.max_train_batches if training else args.max_val_batches
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        chart = batch["chart"].cuda(non_blocking=True)
        audio = batch["audio"].cuda(non_blocking=True)
        controls = batch["controls"].cuda(non_blocking=True)
        mask = batch["valid_mask"].cuda(non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            clean, _ = vae.encode(chart)
            clean = (clean - latent_mean) / latent_std

        if training:
            timesteps = torch.randint(0, schedule.shape[0], (chart.shape[0],), device="cuda")
            noise = torch.randn_like(clean)
            drop_mask = torch.rand(chart.shape[0], device="cuda") < args.control_dropout
        else:
            span = schedule.shape[0] // 2
            timesteps = schedule.shape[0] - span + (
                (torch.arange(chart.shape[0], device="cuda") * 83 + batch_index * 137) % span
            )
            generator = torch.Generator(device="cuda").manual_seed(300000 + batch_index)
            noise = torch.randn(clean.shape, device="cuda", dtype=clean.dtype, generator=generator)
            drop_mask = None
        alpha = schedule[timesteps].view(-1, 1, 1)
        noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise

        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.float16):
            prediction, rhythm_logits = model(
                noisy,
                timesteps,
                audio,
                controls,
                control_drop_mask=drop_mask,
                return_aux=True,
            )
            diffusion_loss = masked_smooth_l1(prediction, noise, mask)
            targets = rhythm_targets(chart)
            rhythm_loss = masked_rhythm_loss(rhythm_logits, targets, mask)
            loss = diffusion_loss + 0.25 * rhythm_loss
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        totals["loss"] += float(loss.detach())
        totals["diffusion_loss"] += float(diffusion_loss.detach())
        totals["rhythm_loss"] += float(rhythm_loss.detach())
        rhythm_metrics.update(rhythm_logits, targets, mask)
        batches += 1

    result = {key: value / max(1, batches) for key, value in totals.items()}
    result.update(rhythm_metrics.result())
    result["batches"] = batches
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True
    run_dir = RUN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    vae = ChartVAE().cuda().eval()
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location="cuda", weights_only=False)["model"])
    vae.requires_grad_(False)
    latent_mean, latent_std = calculate_latent_stats(vae, args)
    (run_dir / "latent_stats.json").write_text(
        json.dumps({"mean": latent_mean.flatten().tolist(), "std": latent_std.flatten().tolist()}, indent=2) + "\n",
        encoding="utf-8",
    )

    train_loader = make_audio_loader("train", args)
    val_loader = make_audio_loader("validation", args, augment=False)
    model = MugInspiredAudioChartDiffusion().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    schedule = torch.cumprod(1.0 - cosine_schedule(args.diffusion_steps, torch.device("cuda")), dim=0)
    print(f"device={torch.cuda.get_device_name(0)} parameters={sum(p.numel() for p in model.parameters()):,}")
    print(f"train_windows={len(train_loader.dataset)} val_windows={len(val_loader.dataset)}")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_summary = run_epoch(
            model, vae, train_loader, schedule, latent_mean, latent_std, args, optimizer, scaler
        )
        with torch.no_grad():
            validation_summary = run_epoch(
                model, vae, val_loader, schedule, latent_mean, latent_std, args, None, scaler
            )
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 3),
            "train": train_summary,
            "validation": validation_summary,
            "max_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
        }
        with (run_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")
        print(json.dumps(record))
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "latent_mean": latent_mean.cpu(),
            "latent_std": latent_std.cpu(),
            "vae_checkpoint": str(VAE_CHECKPOINT),
            "architecture": "mug-inspired-maimai-v3",
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if validation_summary["diffusion_loss"] < best_loss:
            best_loss = validation_summary["diffusion_loss"]
            torch.save(checkpoint, run_dir / "best.pt")
    print(f"run_dir={run_dir} best_validation_diffusion_loss={best_loss:.6f}")


if __name__ == "__main__":
    main()
