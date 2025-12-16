import argparse
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from sbhfrl.data_utils import get_dataset
from sbhfrl.losses import BaseProtoLoss, HDIBLoss
from sbhfrl.models import build_model
from sbhfrl.optim import Muon
from sbhfrl.utils import evaluate, get_device, load_config, save_checkpoint, set_seed


def _build_loader(dataset: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    pin_memory = num_workers > 0
    persistent = num_workers > 0
    prefetch = 2 if num_workers > 0 else None
    kwargs = {}
    if prefetch is not None:
        kwargs["prefetch_factor"] = prefetch
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=shuffle and len(dataset) >= batch_size,
        pin_memory=pin_memory,
        persistent_workers=persistent,
        **kwargs,
    )


def _compute_epochs(config: Dict) -> int:
    epochs = config.get("epochs")
    if epochs is not None:
        return int(epochs)
    rounds = int(config.get("rounds", 1))
    local_epochs = int(config.get("local_epochs", 1))
    return max(1, rounds * local_epochs)


def _train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    config: Dict,
    use_hdib: bool,
) -> Tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    max_grad_norm = float(config.get("max_grad_norm", 5.0))
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits, embeddings = model(images)
        if use_hdib:
            aux = getattr(model, "aux", {})
            attn_weights = aux.get("attn_weights", aux.get("weights", None))
            loss = criterion(
                logits,
                labels,
                mus=aux.get("mus", []),
                logvars=aux.get("logvars", []),
                sampled_feats=aux.get("sampled_feats", []),
                fused_repr=aux.get("embeddings", embeddings),
                attn_weights=attn_weights,
            )
        else:
            loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        batch_size = int(labels.size(0))
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
    return total_loss / max(total_samples, 1), total_samples


def run_centralized_training(config: Dict, device: torch.device) -> None:
    train_dataset, test_dataset = get_dataset(config)
    train_loader = _build_loader(
        train_dataset,
        batch_size=int(config.get("batch_size", 64)),
        shuffle=True,
        num_workers=int(config.get("data_num_workers", 0)),
    )
    test_loader = _build_loader(
        test_dataset,
        batch_size=int(config.get("eval_batch_size", 256)),
        shuffle=False,
        num_workers=int(config.get("data_num_workers", 0)),
    )

    model = build_model(config).to(device)
    use_hdib = config.get("model", "base") == "hdib"
    criterion = HDIBLoss(config) if use_hdib else BaseProtoLoss()

    betas = config.get("muon_betas", [0.9, 0.99])
    optimizer = Muon(
        model.parameters(),
        lr=float(config.get("lr", 0.01)),
        betas=(float(betas[0]), float(betas[1])),
        eps=float(config.get("muon_eps", 1e-8)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    epochs = _compute_epochs(config)
    best_acc = -1.0
    best_state = None
    for epoch in range(epochs):
        loss, seen = _train_one_epoch(model, train_loader, optimizer, criterion, device, config, use_hdib)
        acc = evaluate(model, test_loader, device)
        print(f"[Epoch {epoch + 1:03d}/{epochs}] loss={loss:.4f} samples={seen} acc={acc * 100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    save_path = config.get("save_checkpoint")
    if save_path:
        to_save = best_state or {k: v.detach().cpu() for k, v in model.state_dict().items()}
        save_checkpoint(to_save, save_path, meta={"method": "centralized", "best_acc": best_acc})


def _parse_args():
    parser = argparse.ArgumentParser(description="Centralized training baseline using SB-HFRL models.")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config JSON file.")
    parser.add_argument("--save-ckpt", type=str, default=None, help="Optional path to save the best checkpoint.")
    parser.add_argument(
        "--force-resnet18",
        action="store_true",
        help="Override config['model'] to 'resnet18' (useful for centralized baseline).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)
    if args.force_resnet18:
        config["model"] = "resnet18"
    if args.save_ckpt:
        config["save_checkpoint"] = args.save_ckpt
    set_seed(int(config.get("seed", 42)))
    device = get_device(config.get("device", "auto"))
    run_centralized_training(config, device)


if __name__ == "__main__":
    main()
