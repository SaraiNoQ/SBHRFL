import argparse
import random
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from sbhfrl.data_utils import build_loaders, dirichlet_partition, get_dataset
from sbhfrl.federated.aggregation import _avg_state_dicts
from sbhfrl.losses import BaseProtoLoss, HDIBLoss
from sbhfrl.models import build_model
from sbhfrl.optim import Muon
from sbhfrl.utils import evaluate, get_device, load_config, save_checkpoint, set_seed


class FedAvgClient:
    """Minimal FedAvg client that mirrors the local training setup of SB-HFRL."""

    def __init__(self, client_id: int, loader: DataLoader, config: Dict, malicious: bool = False):
        self.client_id = client_id
        self.loader = loader
        self.num_samples = len(loader.dataset)
        self.config = config
        self.malicious = malicious
        self.use_hdib = config.get("model", "base") == "hdib"
        self.criterion = HDIBLoss(config) if self.use_hdib else BaseProtoLoss()

    def _fake_state(self, reference_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Craft a noisy payload to simulate a malicious participant."""
        fake = {}
        for name, tensor in reference_state.items():
            if tensor.is_floating_point():
                fake[name] = torch.randn_like(tensor).cpu()
            elif tensor.is_complex():
                real = torch.randn_like(tensor.real)
                imag = torch.randn_like(tensor.real)
                fake[name] = (real + 1j * imag).cpu()
            elif tensor.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
                fake[name] = torch.randint(low=0, high=10, size=tensor.shape, dtype=tensor.dtype).cpu()
            else:
                fake[name] = torch.zeros_like(tensor).cpu()
        return fake

    def train(self, global_state: Dict[str, torch.Tensor], device: torch.device) -> Dict:
        if self.malicious:
            return {
                "client_id": self.client_id,
                "state_dict": self._fake_state(global_state),
                "num_samples": self.num_samples,
                "malicious": True,
            }

        model = build_model(self.config).to(device)
        model.load_state_dict(global_state)

        betas = self.config.get("muon_betas", [0.9, 0.99])
        optimizer = Muon(
            model.parameters(),
            lr=self.config.get("lr", 0.01),
            betas=(betas[0], betas[1]),
            eps=self.config.get("muon_eps", 1e-8),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )

        for _ in range(self.config.get("local_epochs", 1)):
            for images, labels in self.loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits, embeddings = model(images)
                if self.use_hdib:
                    aux = getattr(model, "aux", {})
                    loss = self.criterion(
                        logits,
                        labels,
                        mus=aux.get("mus", []),
                        logvars=aux.get("logvars", []),
                        sampled_feats=aux.get("sampled_feats", []),
                        fused_repr=aux.get("embeddings", embeddings),
                    )
                else:
                    loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get("max_grad_norm", 5.0))
                optimizer.step()

        return {
            "client_id": self.client_id,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "num_samples": self.num_samples,
            "malicious": False,
        }


def _aggregate(payloads: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights = torch.tensor([p["num_samples"] for p in payloads], dtype=torch.float32)
    state_dicts = [{k: v.float() for k, v in p["state_dict"].items()} for p in payloads]
    return _avg_state_dicts(state_dicts, weights)


def run_fedavg(config: Dict, device: torch.device) -> None:
    train_dataset, test_dataset = get_dataset(config)
    num_clients = config["num_shards"] * config["clients_per_shard"]
    subsets = dirichlet_partition(train_dataset, num_clients, config.get("alpha_dirichlet", 0.5))
    loaders = build_loaders(subsets, config.get("batch_size", 64), num_workers=config.get("data_num_workers", 0))
    mal_ratio = max(0.0, min(1.0, config.get("mal_ratio", 0.0)))
    num_malicious = int(num_clients * mal_ratio)
    malicious_ids = set(random.sample(range(num_clients), num_malicious)) if num_malicious > 0 else set()
    clients = [
        FedAvgClient(idx, loader, config, malicious=idx in malicious_ids) for idx, loader in enumerate(loaders)
    ]
    if malicious_ids:
        print(f"[FedAvg] Injected {len(malicious_ids)}/{num_clients} malicious clients (ratio={mal_ratio:.2f}).")

    global_model = build_model(config).to(device)
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=config.get("data_num_workers", 0),
    )

    best_acc = -1.0
    best_state = None
    for round_idx in range(config.get("rounds", 1)):
        participate = min(config.get("clients_per_round", len(clients)), len(clients))
        selected = random.sample(clients, participate)
        payloads = [client.train(global_state, device) for client in selected]
        global_state = _aggregate(payloads)
        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, test_loader, device)
        print(f"[Round {round_idx + 1}] FedAvg Accuracy: {acc * 100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in global_state.items()}

    save_path = config.get("save_checkpoint")
    if save_path:
        to_save = best_state or {k: v.cpu() for k, v in global_state.items()}
        save_checkpoint(to_save, save_path, meta={"method": "fedavg", "best_acc": best_acc})


def _parse_args():
    parser = argparse.ArgumentParser(description="FedAvg baseline aligned with SB-HFRL configs.")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config JSON file.")
    parser.add_argument("--save-ckpt", type=str, default=None, help="Optional path to save the best checkpoint.")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)
    if args.save_ckpt:
        config["save_checkpoint"] = args.save_ckpt
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    run_fedavg(config, device)


if __name__ == "__main__":
    main()
