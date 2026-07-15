from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GroupKFold

from maimai_ai.audio_analysis import FEATURE_NAMES, MODEL_PATH, extract_candidates


DATASET = Path(r"D:\trans\maimai_finale_dataset")
CACHE = DATASET / "bpm_training_features.json"


def load_samples() -> list[dict]:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    songs = [json.loads(line) for line in (DATASET / "songs.jsonl").read_text(encoding="utf-8").splitlines()]
    samples = []
    for number, song in enumerate(songs, 1):
        candidates, metadata = extract_candidates(DATASET / song["directory"] / "track.mp3")
        truth = float(song["whole_bpm"])
        best = min(range(len(candidates)), key=lambda index: abs(candidates[index]["bpm"] - truth) / truth)
        for index, candidate in enumerate(candidates):
            samples.append(
                {
                    "song_id": song["song_id"],
                    "truth": truth,
                    "bpm": candidate["bpm"],
                    "relative_error": abs(candidate["bpm"] - truth) / truth,
                    "label": int(index == best),
                    "feature": candidate["feature"],
                    "metadata": metadata,
                }
            )
        print(f"[{number:02d}/{len(songs):02d}] {song['song_id']} truth={truth:g} candidates={len(candidates)} best={candidates[best]['bpm']:.3f}")
    CACHE.write_text(json.dumps(samples, separators=(",", ":")) + "\n", encoding="utf-8")
    return samples


def main() -> None:
    samples = load_samples()
    x = np.asarray([row["feature"] for row in samples], dtype=np.float32)
    y = np.asarray([row["label"] for row in samples], dtype=np.int64)
    groups = np.asarray([row["song_id"] for row in samples])
    candidates = {
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=20260701,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=120,
            learning_rate=0.04,
            max_depth=2,
            random_state=20260701,
        ),
    }
    best_name = ""
    best_accuracy = -1.0
    splitter = GroupKFold(n_splits=7)
    for name, model in candidates.items():
        selected = []
        for train, test in splitter.split(x, y, groups):
            model.fit(x[train], y[train])
            probability = model.predict_proba(x[test])[:, 1]
            for song_id in np.unique(groups[test]):
                indices = test[groups[test] == song_id]
                selected.append(samples[int(indices[int(np.argmax(probability[groups[test] == song_id]))])])
        within_3 = sum(row["relative_error"] <= 0.03 for row in selected) / len(selected)
        within_5 = sum(row["relative_error"] <= 0.05 for row in selected) / len(selected)
        print(f"{name} group_cv songs={len(selected)} within_3pct={within_3:.4f} within_5pct={within_5:.4f}")
        if within_3 > best_accuracy:
            best_name = name
            best_accuracy = within_3

    model = candidates[best_name]
    model.fit(x, y)
    package = {
        "model": model,
        "model_name": best_name,
        "feature_names": FEATURE_NAMES,
        "training_songs": len(np.unique(groups)),
        "group_cv_within_3pct": best_accuracy,
    }
    joblib.dump(package, MODEL_PATH)
    print(f"output={MODEL_PATH} model={best_name} group_cv_within_3pct={best_accuracy:.4f}")


if __name__ == "__main__":
    main()
