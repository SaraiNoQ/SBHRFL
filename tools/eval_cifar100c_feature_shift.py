"""
Evaluate corruption-specific accuracy on CIFAR-100-C to mirror feature-shift tables.

Features:
- Uses the same dataset loader and model builder as the main pipeline.
- Supports corruption-specific evaluation and shard-to-corruption assignment to check heterogeneity.
- Outputs a CSV-friendly summary line per corruption plus mean.
"""
import argparse
import csv
import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from sbhfrl.data_utils import CIFARCTransformDataset, _extract_labels, dirichlet_partition
from sbhfrl.models import build_model
from sbhfrl.utils import evaluate, load_config, set_seed, get_device


def _build_dataset(root: str, corruption: str, severities: List[int], train: bool) -> CIFARCTransformDataset:
    return CIFARCTransformDataset(
        root=root,
        corruption_dir="cifar100-c",
        corruptions=[corruption],
        severities=severities if severities else None,
        train=train,
        train_ratio=0.8,
        augment=train,
    )


def _count_partition(dataset, num_clients: int, alpha: float, num_classes: int) -> Dict[int, Dict[int, int]]:
    subsets = dirichlet_partition(dataset, num_clients, alpha)
    labels = _extract_labels(dataset)
    stats: Dict[int, Dict[int, int]] = {}
    for cid, subset in enumerate(subsets):
        idxs = subset.indices
        client_labels = labels[idxs]
        hist = torch.bincount(torch.tensor(client_labels, dtype=torch.long), minlength=num_classes)
        stats[cid] = {int(i): int(v.item()) for i, v in enumerate(hist) if v.item() > 0}
    return stats


def evaluate_corruptions(config: Dict, checkpoint: str, corruptions: List[str], severities: List[int], out_csv: str):
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    model = build_model(config).to(device)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        # Accept plain state_dict or wrapped dict
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state)
    model.eval()

    results = []
    for corr in corruptions:
        test_ds = _build_dataset(config.get("data_root", "./data"), corr, severities, train=False)
        test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=config.get("data_num_workers", 0))
        acc = evaluate(model, test_loader, device)
        results.append((corr, acc))
        print(f"[Eval] Corruption={corr} Acc={acc * 100:.2f}%")

    mean_acc = sum(acc for _, acc in results) / max(len(results), 1)
    print(f"[Eval] Mean Acc={mean_acc * 100:.2f}%")

    if out_csv:
        header = ["corruption", "acc"]
        os.makedirs(os.path.dirname(out_csv), exist_ok=True) if os.path.dirname(out_csv) else None
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for corr, acc in results:
                writer.writerow([corr, f"{acc * 100:.2f}"])
            writer.writerow(["Mean", f"{mean_acc * 100:.2f}"])
        print(f"Saved CSV summary to {out_csv}")


def visualize_partition_summary(config: Dict, corruptions: List[str], severities: List[int]) -> None:
    """Print label-count summaries for one shard per corruption (round-robin)."""
    num_clients = config["num_shards"] * config["clients_per_shard"]
    alpha = config.get("alpha_dirichlet", 0.5)
    num_classes = config.get("num_classes", 100)
    for idx, corr in enumerate(corruptions):
        train_ds = _build_dataset(config.get("data_root", "./data"), corr, severities, train=True)
        stats = _count_partition(train_ds, num_clients, alpha, num_classes)
        total = len(train_ds)
        print(f"[Partition] Corruption={corr}, train_size={total}, alpha={alpha}")
        # Show a compact summary: total per shard (not full histogram to avoid clutter)
        for cid, hist in stats.items():
            shard_total = sum(hist.values())
            print(f"  shard {cid}: {shard_total} samples")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-100-C corruption-specific accuracy.")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config JSON.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (state_dict).")
    parser.add_argument(
        "--corruptions",
        type=str,
        default="fog,snow,frost,brightness,contrast",
        help="Comma-separated corruption types to evaluate.",
    )
    parser.add_argument(
        "--severities",
        type=str,
        default="",
        help="Comma-separated severities (1-5); empty for all.",
    )
    parser.add_argument("--out-csv", type=str, default="eval_cifar100c_corruptions.csv", help="Where to save CSV summary.")
    parser.add_argument(
        "--no-partition-summary",
        action="store_true",
        help="Skip printing partition counts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    severities = [int(s) for s in args.severities.split(",") if s.strip()]

    if not args.no_partition_summary:
        visualize_partition_summary(config, corruptions, severities)
    evaluate_corruptions(config, args.checkpoint, corruptions, severities, args.out_csv)


if __name__ == "__main__":
    main()
