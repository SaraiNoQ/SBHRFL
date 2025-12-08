import argparse
import random
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sbhfrl.data_utils import build_loaders, dirichlet_partition, get_dataset
from sbhfrl.federated.aggregation import _avg_state_dicts
from sbhfrl.losses import BaseProtoLoss, HDIBLoss
from sbhfrl.models import build_model
from sbhfrl.optim import Muon
from sbhfrl.utils import evaluate, get_device, load_config, save_checkpoint, set_seed


class FedSRClient:
    """
    FedSR client: lightweight reimplementation of FedSR (NeurIPS'22) using the existing
    SB-HFRL pipeline. Adds two regularizers:
      - L2R: penalize representation norm to control magnitude drift.
      - CMI: class-conditional mutual information via learnable Gaussian priors (r_mu, r_sigma)
             scaled by a learnable C following the paper's formulation.
    """

    def __init__(self, client_id: int, loader: DataLoader, config: Dict):
        self.client_id = client_id
        self.loader = loader
        self.num_samples = len(loader.dataset)
        self.config = config
        self.use_hdib = config.get("model", "base") == "hdib"
        self.base_loss = HDIBLoss(config) if self.use_hdib else BaseProtoLoss()

        self.lambda_l2r = float(config.get("lambda_l2r", 0.1))
        self.lambda_cmi = float(config.get("lambda_cmi", 0.01))

        self.num_classes = config.get("num_classes", 10)
        self.embedding_dim = config.get("embedding_dim", 128)

        # Class-wise Gaussian priors and a global scale parameter C.
        self.r_mu = torch.nn.Parameter(torch.zeros(self.num_classes, self.embedding_dim))
        self.r_sigma = torch.nn.Parameter(torch.ones(self.num_classes, self.embedding_dim))
        self.C = torch.nn.Parameter(torch.ones([]))

    def _build_optimizer(self, model: torch.nn.Module) -> Muon:
        betas = self.config.get("muon_betas", [0.9, 0.99])
        opt = Muon(
            list(model.parameters()) + [self.r_mu, self.r_sigma, self.C],
            lr=self.config.get("lr", 0.01),
            betas=(betas[0], betas[1]),
            eps=self.config.get("muon_eps", 1e-8),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )
        return opt

    def _cmi_regularizer(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        r_sigma_softplus = F.softplus(self.r_sigma)
        r_mu = self.r_mu[labels]
        r_sigma = r_sigma_softplus[labels]

        # Treat embeddings as z_mu with unit variance; scale by learnable C as in the paper.
        z_mu_scaled = embeddings * self.C
        z_sigma_scaled = torch.ones_like(embeddings) * self.C

        reg_cmi = torch.log(r_sigma) - torch.log(z_sigma_scaled) + (
            z_sigma_scaled.pow(2) + (z_mu_scaled - r_mu).pow(2)
        ) / (2 * r_sigma.pow(2)) - 0.5
        reg_cmi = reg_cmi.sum(dim=1).mean()
        return reg_cmi

    def train(self, global_state: Dict[str, torch.Tensor], device: torch.device) -> Dict:
        model = build_model(self.config).to(device)
        model.load_state_dict(global_state)

        # Move client-specific priors to the current device before constructing the optimizer.
        self.r_mu.data = self.r_mu.data.to(device)
        self.r_sigma.data = self.r_sigma.data.to(device)
        self.C.data = self.C.data.to(device)
        optimizer = self._build_optimizer(model)

        for _ in range(self.config.get("local_epochs", 1)):
            for images, labels in self.loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits, embeddings = model(images)

                if self.use_hdib:
                    aux = getattr(model, "aux", {})
                    loss = self.base_loss(
                        logits,
                        labels,
                        mus=aux.get("mus", []),
                        logvars=aux.get("logvars", []),
                        sampled_feats=aux.get("sampled_feats", []),
                        fused_repr=aux.get("embeddings", embeddings),
                    )
                else:
                    loss = self.base_loss(logits, labels)

                if self.lambda_l2r > 0:
                    reg_l2r = embeddings.norm(dim=1).mean()
                    loss = loss + self.lambda_l2r * reg_l2r

                if self.lambda_cmi > 0:
                    reg_cmi = self._cmi_regularizer(embeddings, labels)
                    loss = loss + self.lambda_cmi * reg_cmi

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


def run_fedsr(config: Dict, device: torch.device) -> None:
    train_dataset, test_dataset = get_dataset(config)
    num_clients = config["num_shards"] * config["clients_per_shard"]
    subsets = dirichlet_partition(train_dataset, num_clients, config.get("alpha_dirichlet", 0.5))
    loaders = build_loaders(subsets, config.get("batch_size", 64), num_workers=config.get("data_num_workers", 0))
    clients = [FedSRClient(idx, loader, config) for idx, loader in enumerate(loaders)]

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
        print(f"[Round {round_idx + 1}] FedSR Accuracy: {acc * 100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in global_state.items()}

    save_path = config.get("save_checkpoint")
    if save_path:
        to_save = best_state or {k: v.cpu() for k, v in global_state.items()}
        save_checkpoint(to_save, save_path, meta={"method": "fedsr", "best_acc": best_acc})


def _parse_args():
    parser = argparse.ArgumentParser(description="FedSR baseline aligned with SB-HFRL configs.")
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
    run_fedsr(config, device)


if __name__ == "__main__":
    main()
