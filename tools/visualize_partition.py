"""
Visualize client-wise label distributions after dataset partitioning.

- Uses the same dataset loader and Dirichlet split logic as the main code.
- Outputs a bubble scatter plot: x=client, y=label, size/color = sample count.
"""
import argparse
import os
from typing import Dict

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

from sbhfrl.data_utils import _extract_labels, dirichlet_partition, get_dataset
from sbhfrl.utils import load_config, set_seed


def _count_per_client(dataset, subsets, num_classes: int) -> np.ndarray:
    labels = _extract_labels(dataset)
    counts = np.zeros((len(subsets), num_classes), dtype=int)
    for client_id, subset in enumerate(subsets):
        idxs = np.array(subset.indices)
        client_labels = labels[idxs]
        hist = np.bincount(client_labels, minlength=num_classes)
        counts[client_id, :] = hist
    return counts


def _plot_bubble(counts: np.ndarray, title: str, out_path: str) -> None:
    clients = np.arange(counts.shape[0])
    classes = np.arange(counts.shape[1])
    xs, ys, sizes, colors = [], [], [], []
    max_count = counts.max() if counts.size > 0 else 1
    for cid in clients:
        for cls in classes:
            cnt = counts[cid, cls]
            if cnt <= 0:
                continue
            xs.append(cid)
            ys.append(cls)
            colors.append(cnt)
            # Scale bubble area; add a small floor so tiny counts remain visible.
            sizes.append(200.0 * cnt / max_count + 10.0)

    plt.figure(figsize=(7, 5), dpi=150)
    sc = plt.scatter(xs, ys, c=colors, s=sizes, cmap="Blues", alpha=0.7, edgecolors="k", linewidths=0.3)
    cbar = plt.colorbar(sc)
    cbar.set_label("Number of Samples")
    plt.xlabel("Client")
    plt.ylabel("Label")
    plt.title(title)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")


def visualize_partition(config: Dict, out_path: str, alpha_override: float = None) -> None:
    set_seed(config.get("seed", 42))
    train_dataset, _ = get_dataset(config)
    num_clients = config["num_shards"] * config["clients_per_shard"]
    alpha = alpha_override if alpha_override is not None else config.get("alpha_dirichlet", 0.5)
    subsets = dirichlet_partition(train_dataset, num_clients, alpha)
    counts = _count_per_client(train_dataset, subsets, config.get("num_classes", 10))
    title = f"{config.get('dataset', 'dataset')} | Dirichlet alpha={alpha}"
    _plot_bubble(counts, title, out_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize label distribution across clients after partition.")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config JSON.")
    parser.add_argument("--out", type=str, default="partition_vis.png", help="Output image path.")
    parser.add_argument("--alpha", type=float, default=None, help="Override Dirichlet alpha (uses config if None).")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    visualize_partition(config, args.out, alpha_override=args.alpha)


if __name__ == "__main__":
    main()
