"""
Visualize feature-shift robustness on CIFAR-100-C with t-SNE.

The script:
- Builds a mixed Vis_Dataset with Domain A (clean), Domain B (fog/low-contrast), Domain C (snow/spatter).
- Samples 500-1000 images over 3-5 vehicle classes (default: pickup_truck, bus, train).
- Extracts embeddings for FedProto/FedMPS (penultimate layer) and SB-HFRL (purified fingerprint after SFP/attention).
- Plots a 2x3 grid: columns = methods, row 1 color by class, row 2 color by domain.
"""
import argparse
import os
from typing import Dict, Iterable, List, Sequence, Tuple
import sys

# Ensure repository root is on sys.path when running as a script.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import ConcatDataset, DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from sbhfrl.data_utils import CIFARCTransformDataset
from sbhfrl.models import build_model
from sbhfrl.utils import get_device, load_config, set_seed

# Fine-label names from torchvision.datasets.CIFAR100 (kept inline to avoid meta dependency at runtime).
CIFAR100_FINE_LABELS: List[str] = [
    "apple", "aquarium_fish", "baby", "bear", "beaver", "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly", "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach", "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox", "girl", "hamster", "house", "kangaroo", "computer_keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard", "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid", "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum", "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew", "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table", "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe", "whale", "willow_tree", "wolf", "woman", "worm",
]


def _resolve_class_ids(names: Sequence[str]) -> List[int]:
    name_to_idx = {name: idx for idx, name in enumerate(CIFAR100_FINE_LABELS)}
    ids = []
    for name in names:
        key = name.strip()
        if key not in name_to_idx:
            raise ValueError(f"Unknown CIFAR-100 class name '{key}'.")
        ids.append(name_to_idx[key])
    return ids


def _parse_aliases(alias_str: str) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for item in alias_str.split(","):
        if ":" not in item:
            continue
        key, val = item.split(":", 1)
        key = key.strip()
        val = val.strip()
        if key and val:
            aliases[key] = val
    return aliases


def _sample_indices(labels: Iterable[int], target_ids: Sequence[int], max_count: int, rng: np.random.Generator) -> List[int]:
    arr = np.array(list(labels))
    mask = np.isin(arr, target_ids)
    candidates = np.where(mask)[0].tolist()
    if not candidates:
        return []
    rng.shuffle(candidates)
    if max_count and max_count > 0:
        candidates = candidates[:max_count]
    return candidates


class DomainSubset(Dataset):
    """Wrap a dataset with fixed domain id/name."""

    def __init__(self, dataset: Dataset, indices: List[int], domain_id: int, domain_name: str):
        self.dataset = dataset
        self.indices = indices
        self.domain_id = domain_id
        self.domain_name = domain_name

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        img, label = self.dataset[self.indices[idx]]
        return img, label, self.domain_id


def build_vis_dataset(
    data_root: str,
    target_names: Sequence[str],
    fog_corrs: Sequence[str],
    snow_corrs: Sequence[str],
    severities: Sequence[int],
    max_total: int,
    seed: int,
    download_clean: bool,
) -> Tuple[Dataset, Dict[int, str], Dict[int, str]]:
    target_ids = _resolve_class_ids(target_names)
    per_domain = max(1, max_total // 3)
    rng = np.random.default_rng(seed)
    normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
    clean_transform = transforms.Compose([transforms.ToTensor(), normalize])

    try:
        clean_ds = CIFAR100(
            root=data_root,
            train=False,
            download=download_clean,
            transform=clean_transform,
        )
    except Exception as e:
        raise FileNotFoundError(
            "CIFAR-100 clean test set not found. "
            "Place it under data_root/cifar-100-python or re-run with --download-clean."
        ) from e

    clean_idx = _sample_indices(clean_ds.targets, target_ids, per_domain, rng)
    fog_ds = CIFARCTransformDataset(
        root=data_root,
        corruption_dir="cifar100-c",
        corruptions=list(fog_corrs),
        severities=list(severities) if severities else None,
        train=False,
        train_ratio=0.0,  # use full corruption pool for the test split
        augment=False,
    )
    fog_idx = _sample_indices(getattr(fog_ds, "targets", []), target_ids, per_domain, rng)
    snow_ds = CIFARCTransformDataset(
        root=data_root,
        corruption_dir="cifar100-c",
        corruptions=list(snow_corrs),
        severities=list(severities) if severities else None,
        train=False,
        train_ratio=0.0,
        augment=False,
    )
    snow_idx = _sample_indices(getattr(snow_ds, "targets", []), target_ids, per_domain, rng)

    if min(len(clean_idx), len(fog_idx), len(snow_idx)) == 0:
        raise RuntimeError("No samples were collected for at least one domain. Check class names and corruption files.")

    datasets = [
        DomainSubset(clean_ds, clean_idx, 0, "Clean/Sunny"),
        DomainSubset(fog_ds, fog_idx, 1, "Fog/Low-Contrast"),
        DomainSubset(snow_ds, snow_idx, 2, "Snow/Spatter"),
    ]
    class_id_to_name = {idx: CIFAR100_FINE_LABELS[idx] for idx in target_ids}
    domain_id_to_name = {ds.domain_id: ds.domain_name for ds in datasets}
    combined = ConcatDataset(datasets)
    print(
        f"[VisDataset] Clean={len(clean_idx)}, Fog/Low-Contrast={len(fog_idx)}, "
        f"Snow/Spatter={len(snow_idx)} | total={len(combined)}"
    )
    return combined, class_id_to_name, domain_id_to_name


def load_model_from_ckpt(config_path: str, ckpt_path: str, device: torch.device) -> Tuple[torch.nn.Module, Dict]:
    config = load_config(config_path)
    model = build_model(config).to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, config


@torch.no_grad()
def collect_embeddings(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_purified: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feats: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    domains: List[torch.Tensor] = []
    model.eval()
    for images, cls, dom in loader:
        images = images.to(device)
        logits, emb = model(images)
        if use_purified:
            aux = getattr(model, "aux", {})
            emb = aux.get("embeddings", emb)
        feats.append(emb.cpu())
        labels.append(cls)
        domains.append(dom)
    feats_all = torch.cat(feats, dim=0).numpy()
    labels_all = torch.cat(labels, dim=0).numpy()
    domains_all = torch.cat(domains, dim=0).numpy()
    return feats_all, labels_all, domains_all


def run_tsne(embeddings: np.ndarray, perplexity: int, seed: int) -> np.ndarray:
    n_samples = embeddings.shape[0]
    max_allowed = max(2, n_samples - 1)
    effective_perplexity = min(perplexity, max_allowed, 50)
    effective_perplexity = max(2, effective_perplexity)
    tsne = TSNE(
        n_components=2,
        perplexity=effective_perplexity,
        init="pca",
        random_state=seed,
        learning_rate="auto",
    )
    return tsne.fit_transform(embeddings)


def _scatter(ax, coords: np.ndarray, labels: np.ndarray, palette: Dict[int, str], title: str, legend_labels: Dict[int, str]):
    for val in np.unique(labels):
        mask = labels == val
        ax.scatter(coords[mask, 0], coords[mask, 1], s=9, color=palette.get(int(val), "#7f7f7f"), label=legend_labels.get(int(val), str(val)), alpha=0.8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def plot_grid(
    tsne_coords: Dict[str, np.ndarray],
    labels: np.ndarray,
    domains: np.ndarray,
    class_names: Dict[int, str],
    domain_names: Dict[int, str],
    out_path: str,
):
    methods = list(tsne_coords.keys())
    fig, axes = plt.subplots(2, len(methods), figsize=(4 * len(methods), 8))
    class_palette_base = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]  # red/blue first for Car/Truck mapping
    domain_palette = {0: "#d62728", 1: "#1f77b4", 2: "#2ca02c"}  # red: sunny, blue: fog, green: snow
    class_ids_sorted = sorted(np.unique(labels).tolist())
    class_palette = {cid: class_palette_base[i % len(class_palette_base)] for i, cid in enumerate(class_ids_sorted)}

    for col, method in enumerate(methods):
        coords = tsne_coords[method]
        ax_class = axes[0, col]
        _scatter(ax_class, coords, labels, class_palette, f"{method} | Class view", class_names)
        ax_dom = axes[1, col]
        _scatter(ax_dom, coords, domains, domain_palette, f"{method} | Domain view", domain_names)
        if col == len(methods) - 1:
            ax_class.legend(loc="upper right", fontsize=8)
            ax_dom.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    plt.savefig(out_path, dpi=320, bbox_inches="tight")
    print(f"[Viz] Saved t-SNE grid to {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="t-SNE visualization for CIFAR-100-C feature-shift baselines.")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Default config path used when method-specific configs are omitted.")
    parser.add_argument("--fedproto-config", type=str, default=None, help="Config for FedProto backbone (falls back to --config).")
    parser.add_argument("--fedmps-config", type=str, default=None, help="Config for FedMPS backbone (falls back to --config).")
    parser.add_argument("--ours-config", type=str, default=None, help="Config for SB-HFRL backbone (falls back to --config).")
    parser.add_argument("--fedproto-ckpt", type=str, required=True, help="Checkpoint/state_dict for FedProto.")
    parser.add_argument("--fedmps-ckpt", type=str, required=True, help="Checkpoint/state_dict for FedMPS.")
    parser.add_argument("--ours-ckpt", type=str, required=True, help="Checkpoint/state_dict for SB-HFRL.")
    parser.add_argument("--classes", type=str, default="pickup_truck,bus,train", help="Comma-separated CIFAR-100 fine-grained class names (order sets class colors; red then blue).")
    parser.add_argument("--class-aliases", type=str, default="pickup_truck:Truck,bus:Bus,train:Train", help="Optional display aliases, e.g., pickup_truck:Truck.")
    parser.add_argument("--fog-corruptions", type=str, default="fog,contrast", help="Corruptions for Domain B (fog / low contrast).")
    parser.add_argument("--snow-corruptions", type=str, default="snow,frost,spatter", help="Corruptions for Domain C (snow / rain-like).")
    parser.add_argument("--severities", type=str, default="3,4,5", help="Comma-separated severities (1-5); empty for all severities.")
    parser.add_argument("--max-total", type=int, default=900, help="Total samples across domains (2x3 grid looks clean around 500-1000).")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for feature extraction.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--perplexity", type=int, default=35, help="Perplexity for t-SNE.")
    parser.add_argument("--out", type=str, default="tsne_feature_shift_grid.png", help="Where to save the 2x3 grid.")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root; defaults to config['data_root'].")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and t-SNE.")
    parser.add_argument("--device", type=str, default="auto", help="Computation device.")
    parser.add_argument("--download-clean", action="store_true", help="Allow downloading CIFAR-100 clean test set if missing.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    base_config = load_config(args.config)
    data_root = args.data_root or base_config.get("data_root", "./data")

    target_classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    class_alias = _parse_aliases(args.class_aliases)
    fog_corrs = [c.strip() for c in args.fog_corruptions.split(",") if c.strip()]
    snow_corrs = [c.strip() for c in args.snow_corruptions.split(",") if c.strip()]
    severities = [int(s) for s in args.severities.split(",") if s.strip()]

    vis_dataset, class_id_to_name, domain_id_to_name = build_vis_dataset(
        data_root=data_root,
        target_names=target_classes,
        fog_corrs=fog_corrs,
        snow_corrs=snow_corrs,
        severities=severities,
        max_total=args.max_total,
        seed=args.seed,
        download_clean=args.download_clean,
    )
    # Apply user-friendly aliases for legends where provided.
    for cid, name in class_id_to_name.items():
        if name in class_alias:
            class_id_to_name[cid] = class_alias[name]

    loader = DataLoader(
        vis_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    methods = [
        ("FedProto", args.fedproto_config or args.config, args.fedproto_ckpt, False),
        ("FedMPS", args.fedmps_config or args.config, args.fedmps_ckpt, False),
        ("SB-HFRL", args.ours_config or args.config, args.ours_ckpt, True),
    ]

    tsne_outputs: Dict[str, np.ndarray] = {}
    labels_ref: np.ndarray = None  # type: ignore
    domains_ref: np.ndarray = None  # type: ignore

    for name, cfg_path, ckpt_path, purified in methods:
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint for {name} not found: {ckpt_path}")
        print(f"[Load] {name} | config={cfg_path} | ckpt={ckpt_path} | purified={purified}")
        model, cfg = load_model_from_ckpt(cfg_path, ckpt_path, device)
        feats, labels, domains = collect_embeddings(model, loader, device, use_purified=purified)
        if labels_ref is None:
            labels_ref, domains_ref = labels, domains
        tsne = run_tsne(feats, args.perplexity, args.seed)
        tsne_outputs[name] = tsne

    plot_grid(
        tsne_coords=tsne_outputs,
        labels=labels_ref,
        domains=domains_ref,
        class_names=class_id_to_name,
        domain_names=domain_id_to_name,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()
