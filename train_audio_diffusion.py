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
from maimai_ai.diffusion import AudioChartDiffusion, cosine_schedule
from maimai_ai.vae import ChartVAE


PREPARED = Path(r"D:\trans\maimai_finale_dataset\prepared_v1")
AUDIO = Path(r"D:\trans\maimai_finale_dataset\prepared_audio_v1")
VAE_CHECKPOINT = Path(r"D:\trans\maimai_vae\runs\finale_vae_v1\best.pt")
RUN_ROOT = Path(r"D:\trans\maimai_audio_diffusion\runs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--levels-12-15", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--run-name", default=time.strftime("audio_diffusion_%Y%m%d_%H%M%S"))
    parser.add_argument("--seed", type=int, default=20260701)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(split: str, args: argparse.Namespace, augment: bool | None = None) -> DataLoader:
    dataset = AudioChartWindowDataset(
        PREPARED,
        AUDIO,
        split,
        levels_12_15=args.levels_12_15,
        augment=(split == "train") if augment is None else augment,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=split == "train" and (augment is not False),
        num_workers=0,
        pin_memory=True,
        drop_last=split == "train" and (augment is not False),
    )


@torch.inference_mode()
def calculate_latent_stats(vae: ChartVAE, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
    total = torch.zeros(16, device="cuda", dtype=torch.float64)
    square_total = torch.zeros_like(total)
    count = torch.zeros((), device="cuda", dtype=torch.float64)
    for batch in loader:
        chart = batch["chart"].cuda(non_blocking=True)
        mask = batch["valid_mask"].cuda(non_blocking=True)[:, None]
        with torch.autocast("cuda", dtype=torch.float16):
            latent, _ = vae.encode(chart)
        latent = latent.double()
        total += (latent * mask).sum(dim=(0, 2))
        square_total += (latent.square() * mask).sum(dim=(0, 2))
        count += mask.sum()
    mean = total / count
    std = (square_total / count - mean.square()).clamp_min(1e-6).sqrt()
    return mean.float().view(1, 16, 1), std.float().view(1, 16, 1)


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    error = F.mse_loss(prediction, target, reduction="none")
    return (error * mask[:, None]).sum() / (mask.sum() * prediction.shape[1]).clamp_min(1.0)


def rhythm_targets(chart: torch.Tensor) -> torch.Tensor:
    grouped = chart.reshape(chart.shape[0], 6, 8, 384, 8)
    return grouped[:, :5].sum(dim=(2, 4)).clamp_max(1.0)


def rhythm_loss(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    positive_weights = torch.tensor([2.0, 5.0, 5.0, 1.5, 4.0], device=logits.device).view(1, 5, 1)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = torch.where(target > 0.5, loss * positive_weights, loss)
    return (loss * mask[:, None]).sum() / (mask.sum() * logits.shape[1]).clamp_min(1.0)


def run_epoch(
    model: nn.Module,
    vae: ChartVAE,
    loader: DataLoader,
    schedule: torch.Tensor,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    max_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = Counter()
    batches = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        chart = batch["chart"].cuda(non_blocking=True)
        audio = batch["audio"].cuda(non_blocking=True)
        level = batch["level"].cuda(non_blocking=True)
        mask = batch["valid_mask"].cuda(non_blocking=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            clean, _ = vae.encode(chart)
            clean = (clean - latent_mean) / latent_std

        if training:
            timesteps = torch.randint(0, schedule.shape[0], (chart.shape[0],), device="cuda")
            noise = torch.randn_like(clean)
        else:
            # Fixed validation corruption makes epochs directly comparable.
            # At high noise the chart itself carries little information, so this
            # validation slice specifically measures whether audio conditioning helps.
            validation_span = schedule.shape[0] // 2
            timesteps = schedule.shape[0] - validation_span + (
                (torch.arange(chart.shape[0], device="cuda") * 83 + batch_index * 137) % validation_span
            )
            generator = torch.Generator(device="cuda").manual_seed(100000 + batch_index)
            noise = torch.randn(clean.shape, device="cuda", dtype=clean.dtype, generator=generator)
        alpha = schedule[timesteps].view(-1, 1, 1)
        noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise

        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.float16):
            prediction = model(noisy, timesteps, audio, level)
            diffusion_loss = masked_mse(prediction, noise, mask)
            rhythm_logits = model.predict_rhythm(audio, level)
            auxiliary_loss = rhythm_loss(rhythm_logits, rhythm_targets(chart), mask)
            loss = diffusion_loss + 0.25 * auxiliary_loss
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        totals["loss"] += float(loss.detach())
        totals["paired_loss"] += float(diffusion_loss.detach())
        totals["rhythm_loss"] += float(auxiliary_loss.detach())
        if not training and chart.shape[0] > 1:
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                wrong_prediction = model(noisy, timesteps, audio.roll(1, dims=0), level)
                wrong_loss = masked_mse(wrong_prediction, noise, mask)
            totals["mismatched_loss"] += float(wrong_loss)
        batches += 1

    result = {key: value / max(1, batches) for key, value in totals.items()}
    result["batches"] = batches
    if not training and "mismatched_loss" in result:
        result["conditioning_gain"] = result["mismatched_loss"] - result["paired_loss"]
        result["conditioning_ratio"] = result["mismatched_loss"] / max(1e-12, result["paired_loss"])
    return result


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this training script")
    if not (AUDIO / "config.json").exists():
        raise RuntimeError("Run prepare_audio_features.py first")
    seed_everything(args.seed)
    torch.backends.cudnn.benchmark = True

    run_dir = RUN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n", encoding="utf-8")

    vae = ChartVAE().cuda().eval()
    vae_checkpoint = torch.load(VAE_CHECKPOINT, map_location="cuda", weights_only=False)
    vae.load_state_dict(vae_checkpoint["model"])
    vae.requires_grad_(False)
    stats_loader = make_loader("train", args, augment=False)
    latent_mean, latent_std = calculate_latent_stats(vae, stats_loader)
    latent_stats = {"mean": latent_mean.flatten().tolist(), "std": latent_std.flatten().tolist()}
    (run_dir / "latent_stats.json").write_text(json.dumps(latent_stats, indent=2) + "\n", encoding="utf-8")

    train_loader = make_loader("train", args)
    val_loader = make_loader("validation", args, augment=False)
    model = AudioChartDiffusion().cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    betas = cosine_schedule(args.diffusion_steps, device=torch.device("cuda"))
    schedule = torch.cumprod(1.0 - betas, dim=0)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={torch.cuda.get_device_name(0)} parameters={parameter_count:,}")
    print(f"train_windows={len(train_loader.dataset)} val_windows={len(val_loader.dataset)}")

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_summary = run_epoch(
            model, vae, train_loader, schedule, latent_mean, latent_std, optimizer, scaler, args.max_train_batches
        )
        with torch.no_grad():
            validation_summary = run_epoch(
                model, vae, val_loader, schedule, latent_mean, latent_std, None, scaler, args.max_val_batches
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
        }
        torch.save(checkpoint, run_dir / "last.pt")
        if validation_summary["paired_loss"] < best_loss:
            best_loss = validation_summary["paired_loss"]
            torch.save(checkpoint, run_dir / "best.pt")
    print(f"run_dir={run_dir} best_validation_loss={best_loss:.6f}")


if __name__ == "__main__":
    main()
