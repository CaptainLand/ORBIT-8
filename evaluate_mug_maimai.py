from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from maimai_ai.audio_dataset import AudioChartWindowDataset
from maimai_ai.diffusion import cosine_schedule
from maimai_ai.mug_diffusion import MugInspiredAudioChartDiffusion
from maimai_ai.vae import ChartVAE


PREPARED = Path(r"D:\trans\maimai_finale_dataset\prepared_v1")
AUDIO = Path(r"D:\trans\maimai_finale_dataset\prepared_audio_mug_v1")
VAE_CHECKPOINT = Path(r"D:\trans\maimai_vae\runs\finale_vae_v1\best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--split", default="test", choices=("validation", "test"))
    parser.add_argument("--batch-size", type=int, default=16)
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


def audio_for_row(dataset: AudioChartWindowDataset, index: int) -> torch.Tensor:
    row = dataset.rows[index]
    timing = dataset._load_timing(row["tensor_path"])[row["tensor_row"]]
    return torch.from_numpy(dataset._aligned_audio(row["song_id"], timing))


def masked_mean(error: torch.Tensor, mask: torch.Tensor) -> float:
    return float((error * mask[:, None]).sum() / (mask.sum() * error.shape[1]).clamp_min(1.0))


def rhythm_counts(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[int, int, int]:
    predicted = torch.sigmoid(logits) >= 0.5
    expected = target > 0.5
    valid = mask[:, None] > 0.5
    return (
        int((predicted & expected & valid).sum()),
        int((predicted & ~expected & valid).sum()),
        int((~predicted & expected & valid).sum()),
    )


def rhythm_target(chart: torch.Tensor) -> torch.Tensor:
    return chart.reshape(chart.shape[0], 6, 8, 384, 8)[:, :5].sum(dim=(2, 4)).clamp_max(1.0)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    vae = ChartVAE().cuda().eval()
    vae.load_state_dict(torch.load(VAE_CHECKPOINT, map_location="cuda", weights_only=False)["model"])
    model = MugInspiredAudioChartDiffusion().cuda().eval()
    model.load_state_dict(checkpoint["model"])
    latent_mean = checkpoint["latent_mean"].cuda()
    latent_std = checkpoint["latent_std"].cuda()
    steps = int(checkpoint["args"]["diffusion_steps"])
    schedule = torch.cumprod(1.0 - cosine_schedule(steps, torch.device("cuda")), dim=0)

    dataset = AudioChartWindowDataset(PREPARED, AUDIO, args.split, augment=False, audio_per_tick=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    wrong_indices = different_song_indices(dataset)
    test_steps = [250, 500, 750, 900]
    totals = {
        step: {"paired_epsilon": 0.0, "wrong_epsilon": 0.0, "paired_x0": 0.0, "wrong_x0": 0.0, "batches": 0}
        for step in test_steps
    }
    rhythm = {"paired": [0, 0, 0], "wrong": [0, 0, 0]}
    cursor = 0
    for batch_index, batch in enumerate(loader):
        size = batch["chart"].shape[0]
        chart = batch["chart"].cuda(non_blocking=True)
        audio = batch["audio"].cuda(non_blocking=True)
        controls = batch["controls"].cuda(non_blocking=True)
        mask = batch["valid_mask"].cuda(non_blocking=True)
        wrong_audio = torch.stack([audio_for_row(dataset, i) for i in wrong_indices[cursor : cursor + size]]).cuda()
        cursor += size
        with torch.autocast("cuda", dtype=torch.float16):
            clean, _ = vae.encode(chart)
            clean = (clean - latent_mean) / latent_std
        generator = torch.Generator(device="cuda").manual_seed(700000 + batch_index)
        noise = torch.randn(clean.shape, device="cuda", dtype=clean.dtype, generator=generator)

        for step in test_steps:
            timestep = torch.full((size,), step, device="cuda", dtype=torch.long)
            alpha = schedule[step].view(1, 1, 1)
            noisy = alpha.sqrt() * clean + (1.0 - alpha).sqrt() * noise
            with torch.autocast("cuda", dtype=torch.float16):
                paired = model(noisy, timestep, audio, controls)
                wrong = model(noisy, timestep, wrong_audio, controls)
            paired_x0 = (noisy - (1.0 - alpha).sqrt() * paired) / alpha.sqrt()
            wrong_x0 = (noisy - (1.0 - alpha).sqrt() * wrong) / alpha.sqrt()
            values = totals[step]
            values["paired_epsilon"] += masked_mean(F.smooth_l1_loss(paired, noise, beta=0.02, reduction="none"), mask)
            values["wrong_epsilon"] += masked_mean(F.smooth_l1_loss(wrong, noise, beta=0.02, reduction="none"), mask)
            values["paired_x0"] += masked_mean((paired_x0.float() - clean.float()).square(), mask)
            values["wrong_x0"] += masked_mean((wrong_x0.float() - clean.float()).square(), mask)
            values["batches"] += 1

        timestep = torch.full((size,), 750, device="cuda", dtype=torch.long)
        noisy = schedule[750].sqrt() * clean + (1.0 - schedule[750]).sqrt() * noise
        with torch.autocast("cuda", dtype=torch.float16):
            _, paired_rhythm = model(noisy, timestep, audio, controls, return_aux=True)
            _, wrong_rhythm = model(noisy, timestep, wrong_audio, controls, return_aux=True)
        target = rhythm_target(chart)
        for key, logits in (("paired", paired_rhythm), ("wrong", wrong_rhythm)):
            counts = rhythm_counts(logits, target, mask)
            rhythm[key] = [rhythm[key][i] + counts[i] for i in range(3)]

    report = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "windows": len(dataset),
        "mismatch": "audio always taken from a different song; chart controls unchanged",
        "timesteps": {},
        "rhythm": {},
    }
    for step, values in totals.items():
        count = values.pop("batches")
        average = {key: value / count for key, value in values.items()}
        average["epsilon_wrong_over_paired"] = average["wrong_epsilon"] / average["paired_epsilon"]
        average["x0_wrong_over_paired"] = average["wrong_x0"] / average["paired_x0"]
        report["timesteps"][str(step)] = average
    for key, (tp, fp, fn) in rhythm.items():
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        report["rhythm"][key] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
        }
    output = args.checkpoint.parent / f"{args.split}_conditioning.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    print(f"output={output}")


if __name__ == "__main__":
    main()
