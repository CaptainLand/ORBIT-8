from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from maimai_ai.audio_dataset import AudioChartWindowDataset
from maimai_ai.diffusion import AudioChartDiffusion, cosine_schedule
from maimai_ai.vae import ChartVAE


PREPARED = Path(r"D:\trans\maimai_finale_dataset\prepared_v1")
AUDIO = Path(r"D:\trans\maimai_finale_dataset\prepared_audio_v1")
VAE_CHECKPOINT = Path(r"D:\trans\maimai_vae\runs\finale_vae_v1\best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def different_song_indices(dataset: AudioChartWindowDataset) -> list[int]:
    result = []
    count = len(dataset)
    offset = max(1, count // 2)
    for index, row in enumerate(dataset.rows):
        wrong = (index + offset) % count
        while dataset.rows[wrong]["song_id"] == row["song_id"]:
            wrong = (wrong + 1) % count
        result.append(wrong)
    return result


def wrong_audio(dataset: AudioChartWindowDataset, index: int) -> torch.Tensor:
    row = dataset.rows[index]
    timing = dataset._load_timing(row["tensor_path"])[row["tensor_row"]]
    return torch.from_numpy(dataset._aligned_audio(row["song_id"], timing))


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    error = (prediction.float() - target.float()).square() * mask[:, None]
    return float(error.sum() / (mask.sum() * prediction.shape[1]).clamp_min(1.0))


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    diffusion_steps = int(checkpoint["args"]["diffusion_steps"])
    vae = ChartVAE().cuda().eval()
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location="cuda", weights_only=False)["model"])
    model = AudioChartDiffusion().cuda().eval()
    model.load_state_dict(checkpoint["model"])
    latent_mean = checkpoint["latent_mean"].cuda()
    latent_std = checkpoint["latent_std"].cuda()
    schedule = torch.cumprod(1.0 - cosine_schedule(diffusion_steps, torch.device("cuda")), dim=0)

    dataset = AudioChartWindowDataset(PREPARED, AUDIO, args.split, augment=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    wrong_indices = different_song_indices(dataset)
    timesteps_to_test = [250, 500, 750, 900]
    totals = {
        step: {"paired_epsilon": 0.0, "wrong_epsilon": 0.0, "paired_x0": 0.0, "wrong_x0": 0.0, "batches": 0}
        for step in timesteps_to_test
    }
    cursor = 0
    for batch_index, batch in enumerate(loader):
        size = batch["chart"].shape[0]
        chart = batch["chart"].cuda(non_blocking=True)
        audio = batch["audio"].cuda(non_blocking=True)
        level = batch["level"].cuda(non_blocking=True)
        mask = batch["valid_mask"].cuda(non_blocking=True)
        wrong = torch.stack([wrong_audio(dataset, i) for i in wrong_indices[cursor : cursor + size]]).cuda()
        cursor += size
        with torch.autocast("cuda", dtype=torch.float16):
            clean, _ = vae.encode(chart)
            clean = (clean - latent_mean) / latent_std
        generator = torch.Generator(device="cuda").manual_seed(900000 + batch_index)
        noise = torch.randn(clean.shape, device="cuda", dtype=clean.dtype, generator=generator)

        for step in timesteps_to_test:
            timestep = torch.full((size,), step, device="cuda", dtype=torch.long)
            alpha = schedule[step].view(1, 1, 1)
            noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise
            with torch.autocast("cuda", dtype=torch.float16):
                paired_prediction = model(noisy, timestep, audio, level)
                wrong_prediction = model(noisy, timestep, wrong, level)
            paired_x0 = (noisy - (1.0 - alpha).sqrt() * paired_prediction) / alpha.sqrt()
            wrong_x0 = (noisy - (1.0 - alpha).sqrt() * wrong_prediction) / alpha.sqrt()
            values = totals[step]
            values["paired_epsilon"] += masked_mse(paired_prediction, noise, mask)
            values["wrong_epsilon"] += masked_mse(wrong_prediction, noise, mask)
            values["paired_x0"] += masked_mse(paired_x0, clean, mask)
            values["wrong_x0"] += masked_mse(wrong_x0, clean, mask)
            values["batches"] += 1

    report = {"checkpoint": str(args.checkpoint), "split": args.split, "windows": len(dataset), "different_song_only": True, "timesteps": {}}
    for step, values in totals.items():
        count = values.pop("batches")
        averaged = {key: value / count for key, value in values.items()}
        averaged["epsilon_wrong_over_paired"] = averaged["wrong_epsilon"] / averaged["paired_epsilon"]
        averaged["x0_wrong_over_paired"] = averaged["wrong_x0"] / averaged["paired_x0"]
        report["timesteps"][str(step)] = averaged
    output = args.checkpoint.parent / f"{args.split}_conditioning.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    print(f"output={output}")


if __name__ == "__main__":
    main()
