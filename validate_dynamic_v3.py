from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from maimai_ai.dynamic_audio_dataset import DynamicAudioChartDataset


ROOT = Path(r"D:\trans")
PREPARED = ROOT / "maimai_finale_dataset" / "prepared_v2"
AUDIO = ROOT / "maimai_finale_dataset" / "prepared_audio_orbit_v15"
DYNAMIC = ROOT / "maimai_finale_dataset" / "prepared_v3"


def main() -> None:
    np.random.seed(20260703)
    dataset = DynamicAudioChartDataset(
        PREPARED, AUDIO, DYNAMIC, samples_per_epoch=512, augment=True
    )
    categories = Counter()
    crop_lengths = Counter()
    song_ids = set()
    window_ids = set()
    note_counts = []
    onset_counts = []
    failures = []
    for index in range(len(dataset)):
        sample = dataset[index]
        categories[sample["sample_kind"]] += 1
        crop_lengths[str(int(sample["crop_measures"]))] += 1
        song_ids.add(sample["song_id"])
        window_ids.add(sample["window_id"])
        chart = sample["chart"].numpy()
        audio = sample["audio"].numpy()
        valid_ticks = int(sample["valid_mask"].sum().item() * 8)
        note_counts.append(float(chart[:, :valid_ticks].sum()))
        onset_counts.append(int((sample["onset_peak"][:valid_ticks] > 0).sum()))
        if chart.shape != (48, 3072) or audio.shape != (132, 3072):
            failures.append(f"shape:{sample['window_id']}")
        if not np.isfinite(audio).all():
            failures.append(f"nonfinite:{sample['window_id']}")
        if valid_ticks not in (1536, 2304, 3072):
            failures.append(f"valid_ticks:{sample['window_id']}:{valid_ticks}")
    result = {
        "samples": len(dataset),
        "unique_windows": len(window_ids),
        "unique_songs": len(song_ids),
        "sample_categories": dict(categories),
        "crop_measures": dict(crop_lengths),
        "mean_chart_activity": float(np.mean(note_counts)),
        "mean_consensus_onsets": float(np.mean(onset_counts)),
        "failures": failures,
    }
    (DYNAMIC / "qa_summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    if failures:
        raise RuntimeError(f"dynamic dataset QA failed with {len(failures)} errors")


if __name__ == "__main__":
    main()
