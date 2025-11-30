import argparse
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sbhfrl.data_utils import build_loaders, compute_prototypes, dirichlet_partition, get_dataset
from sbhfrl.federated.aggregation import _avg_prototypes, _avg_state_dicts
from sbhfrl.losses import BaseProtoLoss, HDIBLoss, supervised_contrastive
from sbhfrl.models import build_model
from sbhfrl.optim import Muon
from sbhfrl.utils import evaluate, get_device, load_config, set_seed


def _temperatured_kl(
    student_logits: torch.Tensor, teacher_probs: torch.Tensor, temperature: float
) -> torch.Tensor:
    """KL divergence with temperature scaling."""
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


class FedMPSClient:
    """FedMPS client: multi-level prototypes + soft label guidance."""

    def __init__(self, client_id: int, loader: DataLoader, config: Dict):
        self.client_id = client_id
        self.loader = loader
        self.num_samples = len(loader.dataset)
        self.config = config
        self.use_hdib = config.get("model", "base") == "hdib"
        self.ce_loss = HDIBLoss(config) if self.use_hdib else BaseProtoLoss()
        # Paper-inspired defaults: stronger soft-label guidance and prototype alignment.
        self.lambda_contrast = float(config.get("lambda_proto_contrast", 0.2))
        self.lambda_soft = float(config.get("lambda_soft_label", 0.6))
        self.lambda_align = float(config.get("lambda_proto_align", 0.3))
        self.proto_temp = float(config.get("proto_temperature", 0.7))

    def _build_optimizer(self, model: torch.nn.Module) -> Muon:
        betas = self.config.get("muon_betas", [0.9, 0.99])
        return Muon(
            model.parameters(),
            lr=self.config.get("lr", 0.01),
            betas=(betas[0], betas[1]),
            eps=self.config.get("muon_eps", 1e-8),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )

    def _soft_labels(self, embeddings: torch.Tensor, global_proto: torch.Tensor) -> torch.Tensor:
        """Generate soft labels via prototype similarity."""
        sim = torch.matmul(F.normalize(embeddings, dim=1), global_proto.t()) / self.proto_temp
        return F.softmax(sim, dim=1)

    def train(
        self,
        global_state: Dict[str, torch.Tensor],
        global_proto: Optional[torch.Tensor],
        device: torch.device,
    ) -> Dict:
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
                    base_loss = self.ce_loss(
                        logits,
                        labels,
                        mus=aux.get("mus", []),
                        logvars=aux.get("logvars", []),
                        sampled_feats=aux.get("sampled_feats", []),
                        fused_repr=aux.get("embeddings", embeddings),
                    )
                else:
                    base_loss = self.ce_loss(logits, labels)

                loss = base_loss
                loss = loss + self.lambda_contrast * supervised_contrastive(embeddings, labels)

                if global_proto is not None:
                    proto_device = global_proto.to(device)
                    soft_target = self._soft_labels(embeddings, proto_device)
                    loss = loss + self.lambda_soft * _temperatured_kl(logits, soft_target, self.proto_temp)
                    proto_targets = proto_device[labels]
                    loss = loss + self.lambda_align * F.mse_loss(F.normalize(embeddings, dim=1), proto_targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get("max_grad_norm", 5.0))
                optimizer.step()

        prototypes = compute_prototypes(model, DataLoader(self.loader.dataset, batch_size=self.config.get("proto_batch_size", 128), shuffle=False), self.config.get("num_classes", 10), device)
        return {
            "client_id": self.client_id,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "prototypes": prototypes.cpu(),
            "num_samples": self.num_samples,
        }


def _aggregate_state(payloads: List[Dict]) -> Dict[str, torch.Tensor]:
    weights = torch.tensor([p["num_samples"] for p in payloads], dtype=torch.float32)
    state_dicts = [{k: v.float() for k, v in p["state_dict"].items()} for p in payloads]
    return _avg_state_dicts(state_dicts, weights)


def _aggregate_proto(payloads: List[Dict]) -> torch.Tensor:
    weights = torch.tensor([p["num_samples"] for p in payloads], dtype=torch.float32)
    protos = [p["prototypes"].float() for p in payloads]
    return _avg_prototypes(protos, weights)


def run_fedmps(config: Dict, device: torch.device) -> None:
    train_dataset, test_dataset = get_dataset(config)
    num_clients = config["num_shards"] * config["clients_per_shard"]
    subsets = dirichlet_partition(train_dataset, num_clients, config.get("alpha_dirichlet", 0.5))
    loaders = build_loaders(subsets, config.get("batch_size", 64), num_workers=config.get("data_num_workers", 0))
    clients = [FedMPSClient(idx, loader, config) for idx, loader in enumerate(loaders)]

    global_model = build_model(config).to(device)
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
    global_proto: Optional[torch.Tensor] = None

    test_loader = DataLoader(
        test_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=config.get("data_num_workers", 0),
    )

    for round_idx in range(config.get("rounds", 1)):
        participate = min(config.get("clients_per_round", len(clients)), len(clients))
        selected = random.sample(clients, participate)
        payloads = [client.train(global_state, global_proto, device) for client in selected]
        global_state = _aggregate_state(payloads)
        global_proto = _aggregate_proto(payloads)
        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, test_loader, device)
        print(f"[Round {round_idx + 1}] FedMPS Accuracy: {acc * 100:.2f}%")


def _parse_args():
    parser = argparse.ArgumentParser(description="FedMPS baseline aligned with SB-HFRL configs.")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config JSON file.")
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    run_fedmps(config, device)


if __name__ == "__main__":
    main()
