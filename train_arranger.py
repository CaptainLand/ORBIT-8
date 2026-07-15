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
from torch.utils.data import DataLoader

from maimai_ai.arranger import OPERATORS, OfficialPatternArranger, OraclePlanDataset
from maimai_ai.patterns import PATTERN_NAMES, PATTERN_TRAINING_WEIGHTS


PREPARED = Path(r"D:\trans\maimai_finale_dataset\prepared_v2")
RUN_ROOT = Path(r"D:\trans\maimai_arranger\runs")
BASE_CHECKPOINT = Path(r"D:\trans\maimai_arranger\runs\orbit_v16_arranger\best.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--run-name", default=time.strftime("orbit_v171_arranger_%Y%m%d_%H%M%S"))
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--seed", type=int, default=20260701)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--from-scratch", action="store_true")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_loader(split: str, args: argparse.Namespace) -> DataLoader:
    dataset = OraclePlanDataset(
        PREPARED,
        split,
        corrupt_previous=0.15 if split == "train" else 0.0,
        mirror_augment=split == "train",
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=split == "train",
        num_workers=0,
        pin_memory=True,
        drop_last=split == "train",
    )


def operator_weights() -> torch.Tensor:
    catalog = json.loads((PREPARED / "star_catalog.json").read_text(encoding="utf-8"))
    counts = Counter()
    for template in catalog:
        for operator in template["operators"]:
            counts[operator] += template["count"] / len(template["operators"])
    total = sum(counts.values())
    weights = [min(4.0, np.sqrt(total / (len(OPERATORS) * max(1.0, counts[operator])))) for operator in OPERATORS]
    return torch.tensor(weights, device="cuda", dtype=torch.float32)


def move_batch(batch: dict) -> dict:
    return {key: value.cuda(non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def run_epoch(model, loader, optimizer, scaler, args, weights) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = Counter()
    max_batches = args.max_train_batches if training else args.max_val_batches
    for batch_index, raw_batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = move_batch(raw_batch)
        with torch.set_grad_enabled(training), torch.autocast("cuda", dtype=torch.float16):
            output = model(batch)
            lane_mask = batch["lane_loss_mask"] > 0.5
            slide_mask = (batch["event_type"] == 2) & (batch["mask"] > 0.5)
            lane_loss = F.cross_entropy(output["delta"][lane_mask], batch["target_delta"][lane_mask])
            if slide_mask.any():
                operator_loss = F.cross_entropy(
                    output["operator"][slide_mask], batch["target_operator"][slide_mask], weight=weights
                )
                endpoint_loss = F.cross_entropy(output["endpoint"][slide_mask], batch["target_endpoint"][slide_mask])
                branch_loss = F.cross_entropy(output["branch"][slide_mask], batch["target_branch"][slide_mask])
            else:
                operator_loss = endpoint_loss = branch_loss = lane_loss * 0.0
            pattern_mask = batch["mask"] > 0.5
            pattern_weights = torch.tensor(PATTERN_TRAINING_WEIGHTS, device="cuda", dtype=torch.float32)
            pattern_loss = F.cross_entropy(
                output["pattern"][pattern_mask], batch["target_pattern"][pattern_mask], weight=pattern_weights
            )
            loss = (
                lane_loss
                + 0.35 * pattern_loss
                + 0.45 * operator_loss
                + 0.25 * endpoint_loss
                + 0.1 * branch_loss
            )
        if training:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        totals["loss"] += float(loss.detach())
        totals["lane_loss"] += float(lane_loss.detach())
        totals["operator_loss"] += float(operator_loss.detach())
        totals["endpoint_loss"] += float(endpoint_loss.detach())
        totals["branch_loss"] += float(branch_loss.detach())
        totals["pattern_loss"] += float(pattern_loss.detach())
        totals["lane_correct"] += int((output["delta"][lane_mask].argmax(-1) == batch["target_delta"][lane_mask]).sum())
        totals["lane_count"] += int(lane_mask.sum())
        predicted_pattern = output["pattern"][pattern_mask].argmax(-1)
        target_pattern = batch["target_pattern"][pattern_mask]
        totals["pattern_correct"] += int((predicted_pattern == target_pattern).sum())
        totals["pattern_count"] += int(pattern_mask.sum())
        for pattern_id in predicted_pattern.detach().cpu().tolist():
            totals[f"predicted_pattern_{PATTERN_NAMES[pattern_id]}"] += 1
        for pattern_id in target_pattern.detach().cpu().tolist():
            totals[f"target_pattern_{PATTERN_NAMES[pattern_id]}"] += 1
        for pattern_id, pattern_name in enumerate(PATTERN_NAMES):
            target_class = target_pattern == pattern_id
            totals[f"pattern_target_{pattern_name}"] += int(target_class.sum())
            totals[f"pattern_correct_{pattern_name}"] += int(
                ((predicted_pattern == pattern_id) & target_class).sum()
            )
        if slide_mask.any():
            predicted_operator = output["operator"][slide_mask].argmax(-1)
            totals["operator_correct"] += int((predicted_operator == batch["target_operator"][slide_mask]).sum())
            totals["operator_count"] += int(slide_mask.sum())
            for operator_id in predicted_operator.detach().cpu().tolist():
                totals[f"predicted_operator_{OPERATORS[operator_id]}"] += 1
        totals["batches"] += 1

    batches = max(1, totals["batches"])
    result = {
        "loss": totals["loss"] / batches,
        "lane_loss": totals["lane_loss"] / batches,
        "operator_loss": totals["operator_loss"] / batches,
        "endpoint_loss": totals["endpoint_loss"] / batches,
        "branch_loss": totals["branch_loss"] / batches,
        "pattern_loss": totals["pattern_loss"] / batches,
        "lane_accuracy": totals["lane_correct"] / max(1, totals["lane_count"]),
        "operator_accuracy": totals["operator_correct"] / max(1, totals["operator_count"]),
        "pattern_accuracy": totals["pattern_correct"] / max(1, totals["pattern_count"]),
        "batches": totals["batches"],
    }
    operator_total = max(1, totals["operator_count"])
    result["predicted_operator_distribution"] = {
        operator: totals[f"predicted_operator_{operator}"] / operator_total for operator in OPERATORS
    }
    pattern_total = max(1, totals["pattern_count"])
    result["predicted_pattern_distribution"] = {
        pattern: totals[f"predicted_pattern_{pattern}"] / pattern_total for pattern in PATTERN_NAMES
    }
    result["target_pattern_distribution"] = {
        pattern: totals[f"target_pattern_{pattern}"] / pattern_total for pattern in PATTERN_NAMES
    }
    result["pattern_recall"] = {
        pattern: totals[f"pattern_correct_{pattern}"] / max(1, totals[f"pattern_target_{pattern}"])
        for pattern in PATTERN_NAMES
    }
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
    model = OfficialPatternArranger().cuda()
    if not args.from_scratch:
        incompatible = model.load_state_dict(
            torch.load(BASE_CHECKPOINT, map_location="cuda", weights_only=False)["model"], strict=False
        )
        print(f"loaded_base={BASE_CHECKPOINT} missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    weights = operator_weights()
    print(f"parameters={sum(parameter.numel() for parameter in model.parameters()):,}")
    print(f"train_windows={len(train_loader.dataset)} val_windows={len(val_loader.dataset)}")

    best_loss = float("inf")
    stale_epochs = 0
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train = run_epoch(model, train_loader, optimizer, scaler, args, weights)
        with torch.no_grad():
            validation = run_epoch(model, val_loader, None, scaler, args, weights)
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
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            stale_epochs = 0
            torch.save(checkpoint, run_dir / "best.pt")
        else:
            stale_epochs += 1
        if stale_epochs >= args.patience:
            print(f"early_stopping epoch={epoch} best_validation_loss={best_loss:.6f}")
            break
    print(f"run_dir={run_dir} best_validation_loss={best_loss:.6f}")


if __name__ == "__main__":
    main()
