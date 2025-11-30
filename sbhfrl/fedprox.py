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
from sbhfrl.utils import evaluate, get_device, load_config, set_seed


class FedProxClient:
    """FedProx client with proximal regularization to the global model."""

    def __init__(self, client_id: int, loader: DataLoader, config: Dict):
        self.client_id = client_id
        self.loader = loader
        self.num_samples = len(loader.dataset)
        self.config = config
        self.use_hdib = config.get("model", "base") == "hdib"
        self.criterion = HDIBLoss(config) if self.use_hdib else BaseProtoLoss()
        self.mu = float(config.get("lambda_prox", 0.01))

    def _build_optimizer(self, model: torch.nn.Module) -> Muon:
        betas = self.config.get("muon_betas", [0.9, 0.99])
        return Muon(
            model.parameters(),
            lr=self.config.get("lr", 0.01),
            betas=(betas[0], betas[1]),
            eps=self.config.get("muon_eps", 1e-8),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )

    def _prox_term(self, model: torch.nn.Module, global_state: Dict[str, torch.Tensor], device: torch.device) -> torch.Tensor:
        if self.mu <= 0:
            return torch.tensor(0.0, device=device)
        loss = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if not param.requires_grad or not param.is_floating_point():
                continue
            if name not in global_state:
                continue
            ref = global_state[name].to(device)
            loss = loss + (param - ref).pow(2).sum()
        return 0.5 * self.mu * loss

    def train(self, global_state: Dict[str, torch.Tensor], device: torch.device) -> Dict:
        model = build_model(self.config).to(device)
        model.load_state_dict(global_state)
        optimizer = self._build_optimizer(model)

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

                loss = loss + self._prox_term(model, global_state, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get("max_grad_norm", 5.0))
                optimizer.step()

        return {
            "client_id": self.client_id,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "num_samples": self.num_samples,
        }


def _aggregate(payloads: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    weights = torch.tensor([p["num_samples"] for p in payloads], dtype=torch.float32)
    state_dicts = [{k: v.float() for k, v in p["state_dict"].items()} for p in payloads]
    return _avg_state_dicts(state_dicts, weights)


def run_fedprox(config: Dict, device: torch.device) -> None:
    train_dataset, test_dataset = get_dataset(config)
    num_clients = config["num_shards"] * config["clients_per_shard"]
    subsets = dirichlet_partition(train_dataset, num_clients, config.get("alpha_dirichlet", 0.5))
    loaders = build_loaders(subsets, config.get("batch_size", 64), num_workers=config.get("data_num_workers", 0))
    clients = [FedProxClient(idx, loader, config) for idx, loader in enumerate(loaders)]

    global_model = build_model(config).to(device)
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=config.get("data_num_workers", 0),
    )

    for round_idx in range(config.get("rounds", 1)):
        participate = min(config.get("clients_per_round", len(clients)), len(clients))
        selected = random.sample(clients, participate)
        payloads = [client.train(global_state, device) for client in selected]
        global_state = _aggregate(payloads)
        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, test_loader, device)
        print(f"[Round {round_idx + 1}] FedProx Accuracy: {acc * 100:.2f}%")


def _parse_args():
    parser = argparse.ArgumentParser(description="FedProx baseline aligned with SB-HFRL configs.")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config JSON file.")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    run_fedprox(config, device)


if __name__ == "__main__":
    main()
