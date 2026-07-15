from __future__ import annotations

import argparse
import json
from pathlib import Path

import av
import numpy as np
import torch
import torchaudio


DATASET_ROOT = Path(r"D:\trans\maimai_finale_dataset")
TARGET_SAMPLE_RATE = 22050


def decode_mono(path: Path) -> tuple[torch.Tensor, int]:
    container = av.open(str(path))
    stream = container.streams.audio[0]
    chunks = [frame.to_ndarray() for frame in container.decode(stream)]
    sample_rate = int(stream.codec_context.sample_rate)
    container.close()
    if not chunks:
        raise RuntimeError(f"No audio frames decoded: {path}")
    waveform = np.concatenate(chunks, axis=1).astype(np.float32, copy=False)
    mono = torch.from_numpy(waveform).mean(dim=0, keepdim=True)
    if sample_rate != TARGET_SAMPLE_RATE:
        mono = torchaudio.functional.resample(mono, sample_rate, TARGET_SAMPLE_RATE)
    return mono, sample_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-name", default="prepared_audio_v1")
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--n-mels", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = DATASET_ROOT / args.output_name
    output_root.mkdir(parents=True, exist_ok=True)
    feature_dir = output_root / "songs"
    feature_dir.mkdir(exist_ok=True)
    songs = [json.loads(line) for line in (DATASET_ROOT / "songs.jsonl").read_text(encoding="utf-8").splitlines()]
    if args.limit:
        songs = songs[: args.limit]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SAMPLE_RATE,
        n_fft=args.n_fft,
        win_length=args.n_fft,
        hop_length=args.hop_length,
        f_min=30.0,
        f_max=TARGET_SAMPLE_RATE / 2,
        n_mels=args.n_mels,
        power=2.0,
        center=True,
    ).to(device)

    rows = []
    train_sum = np.zeros(args.n_mels, dtype=np.float64)
    train_square_sum = np.zeros(args.n_mels, dtype=np.float64)
    train_frames = 0
    max_duration_error_ms = 0.0
    for number, song in enumerate(songs, 1):
        output_path = feature_dir / f"{song['song_id']}.npz"
        audio_path = DATASET_ROOT / song["directory"] / "track.mp3"
        if output_path.exists() and not args.force:
            with np.load(output_path) as saved:
                log_mel = saved["log_mel"].astype(np.float32)
                sample_count = int(saved["sample_count"])
                source_sample_rate = int(saved["source_sample_rate"])
        else:
            waveform, source_sample_rate = decode_mono(audio_path)
            sample_count = waveform.shape[-1]
            with torch.inference_mode():
                mel = transform(waveform.to(device)).squeeze(0)
                log_mel = torch.log(mel.clamp_min(1e-5)).cpu().numpy()
            np.savez_compressed(
                output_path,
                log_mel=log_mel.astype(np.float16),
                sample_count=np.asarray(sample_count, dtype=np.int64),
                source_sample_rate=np.asarray(source_sample_rate, dtype=np.int32),
            )

        decoded_seconds = sample_count / TARGET_SAMPLE_RATE
        duration_error_ms = abs(decoded_seconds - float(song["audio_seconds"])) * 1000.0
        max_duration_error_ms = max(max_duration_error_ms, duration_error_ms)
        if song["split"] == "train":
            train_sum += log_mel.sum(axis=1, dtype=np.float64)
            train_square_sum += np.square(log_mel, dtype=np.float64).sum(axis=1)
            train_frames += log_mel.shape[1]
        rows.append(
            {
                "song_id": song["song_id"],
                "split": song["split"],
                "feature_path": f"songs/{song['song_id']}.npz",
                "frames": int(log_mel.shape[1]),
                "decoded_seconds": decoded_seconds,
                "duration_error_ms": duration_error_ms,
            }
        )
        print(f"[{number:02d}/{len(songs):02d}] {song['song_id']} frames={log_mel.shape[1]} error_ms={duration_error_ms:.2f}")

    mean = train_sum / max(1, train_frames)
    variance = train_square_sum / max(1, train_frames) - np.square(mean)
    std = np.sqrt(np.maximum(variance, 1e-6))
    config = {
        "version": 1,
        "sample_rate": TARGET_SAMPLE_RATE,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "n_mels": args.n_mels,
        "frame_seconds": args.hop_length / TARGET_SAMPLE_RATE,
        "train_frames": train_frames,
        "train_mean": mean.tolist(),
        "train_std": std.tolist(),
        "max_duration_error_ms": max_duration_error_ms,
        "alignment": "full-song log-mel cache; window alignment is performed from exact tick_time_ms",
    }
    (output_root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    with (output_root / "index.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"output={output_root} songs={len(rows)} max_duration_error_ms={max_duration_error_ms:.3f}")


if __name__ == "__main__":
    main()
