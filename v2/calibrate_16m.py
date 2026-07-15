from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from train_rhythm_plan import make_loader, seed_everything
from v2.rhythm_model import OrbitV2RhythmModel
from v2.rhythm_model_16m import OrbitV2RhythmModel16M
from v2.train_16m import run_epoch, selection_score


ROOT = Path(r"D:\trans")
SOURCE = ROOT / "v2" / "runs" / "orbit_v2_16m_dynamic_v3" / "best.pt"
TEACHER = ROOT / "v2" / "runs" / "orbit_v2_hierarchical_v1" / "best.pt"
RUN_ROOT = ROOT / "v2" / "runs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate ORBIT-8 v2 16M onset timing")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--gradient-accumulation", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1.2e-5)
    parser.add_argument("--inherited-learning-rate", type=float, default=8e-6)
    parser.add_argument("--distill-event-weight", type=float, default=0.25)
    parser.add_argument("--distill-count-weight", type=float, default=0.10)
    parser.add_argument("--distill-onset-weight", type=float, default=0.50)
    parser.add_argument("--run-name", default="orbit_v2_16m_calibrated")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int, default=20260703)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed_everything(args.seed)
    run_dir = RUN_ROOT / args.run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        **vars(args),
        "model": "ORBIT-8 v2 16M calibrated",
        "source_checkpoint": str(SOURCE),
        "teacher_checkpoint": str(TEACHER),
        "training_data": "prepared_v2 fixed windows",
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    train_loader = make_loader("train", args)
    val_loader = make_loader("validation", args)
    source = torch.load(SOURCE, map_location="cpu", weights_only=False)
    model = OrbitV2RhythmModel16M()
    model.load_state_dict(source["model"])
    teacher = OrbitV2RhythmModel()
    teacher.load_state_dict(torch.load(TEACHER, map_location="cpu", weights_only=False)["model"])
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
    scaler = torch.amp.GradScaler("cuda")
    best_score = float(source.get("selection_score", -1.0))
    torch.save({**source, "args": config}, run_dir / "best.pt")

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train = run_epoch(model, train_loader, optimizer, scaler, args, teacher)
        with torch.no_grad():
            validation = run_epoch(model, val_loader, None, scaler, args)
        score = selection_score(validation)
        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 3),
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
            torch.save(checkpoint, run_dir / "best.pt")
    print(f"run_dir={run_dir} best_selection_score={best_score:.6f}")


if __name__ == "__main__":
    main()
