import argparse
import random
from typing import Dict, List
import copy  # Added for safe operations if needed

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
    """Original FedAvg Aggregation (Weighted Average)."""
    weights = torch.tensor([p["num_samples"] for p in payloads], dtype=torch.float32)
    state_dicts = [{k: v.float() for k, v in p["state_dict"].items()} for p in payloads]
    return _avg_state_dicts(state_dicts, weights)


def _trim_mean_aggregate(payloads: List[Dict[str, torch.Tensor]], beta: float = 0.1) -> Dict[str, torch.Tensor]:
    """
    [Added] Trimmed Mean Aggregation.
    Sorts parameters coordinate-wise and removes the largest and smallest beta fraction.
    """
    if not payloads:
        return {}
    
    state_dicts = [p["state_dict"] for p in payloads]
    num_clients = len(state_dicts)
    k = int(num_clients * beta)
    
    # Safety check: ensure we don't trim all clients
    if 2 * k >= num_clients:
        k = max(0, (num_clients - 1) // 2)

    aggregated_state = {}
    ref_keys = state_dicts[0].keys()

    for key in ref_keys:
        # Stack tensors: (num_clients, *shape)
        stacked = torch.stack([sd[key].float() for sd in state_dicts], dim=0)
        
        if stacked.dtype.is_floating_point:
            # Sort along client dimension
            sorted_tensor, _ = torch.sort(stacked, dim=0)
            # Trim the top k and bottom k
            # If k=0, slice [0:num_clients], effectively no trimming
            trimmed = sorted_tensor[k : num_clients - k]
            # Average the remaining
            aggregated_state[key] = torch.mean(trimmed, dim=0)
        else:
            # Fallback for non-float types (e.g. integer buffers)
            aggregated_state[key] = torch.mean(stacked.float(), dim=0).to(stacked.dtype)

    return aggregated_state


def _krum_aggregate(payloads: List[Dict[str, torch.Tensor]], malicious_ratio: float = 0.0) -> Dict[str, torch.Tensor]:
    """
    [Added] Krum Aggregation.
    Selects one local model update that minimizes the sum of squared Euclidean distances 
    to its (n - f - 2) nearest neighbors.
    """
    if not payloads:
        return {}

    state_dicts = [p["state_dict"] for p in payloads]
    num_clients = len(state_dicts)
    
    # f is the estimated number of malicious clients
    f = int(num_clients * malicious_ratio)
    
    # Krum constraint: we sum distances to k = n - f - 2 neighbors
    # Ensure k >= 1
    k = num_clients - f - 2
    if k < 1:
        # If too few clients or f is too high, fallback to simple majority or min neighbors
        k = max(1, num_clients // 2)

    # Flatten parameters into a single vector per client for distance calculation
    flat_params_list = []
    for sd in state_dicts:
        flat = []
        for key in sorted(sd.keys()):
            # Only use floating point params for distance to save compute
            if sd[key].is_floating_point():
                flat.append(sd[key].view(-1).float())
        flat_params_list.append(torch.cat(flat))
    
    updates = torch.stack(flat_params_list) # (num_clients, total_params)
    
    # Compute pairwise Euclidean distances: dists[i, j] = ||w_i - w_j||
    # utilizing torch.cdist for efficiency
    dists = torch.cdist(updates, updates, p=2) 
    
    scores = []
    for i in range(num_clients):
        d_i = dists[i]
        # Sort distances for client i
        sorted_dists, _ = torch.sort(d_i)
        
        # Sum the smallest k distances
        # Note: sorted_dists[0] is distance to self (0.0), so we take [1 : k+1]
        score = torch.sum(sorted_dists[1 : k + 1])
        scores.append(score.item())
    
    # Select the client index with the minimum score
    best_client_idx = scores.index(min(scores))
    
    # Return the raw state_dict of the selected client (Krum does not average)
    return state_dicts[best_client_idx]


def run_fedavg(config: Dict, device: torch.device, aggregator_name: str = "avg") -> None:
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
    
    print(f"[FedAvg] Aggregation Method: {aggregator_name}")

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
        
        if aggregator_name == "trim_mean":
            # Use mal_ratio as beta if provided, else default to small trim
            beta = mal_ratio if mal_ratio > 0 else 0.1
            global_state = _trim_mean_aggregate(payloads, beta=beta)
        elif aggregator_name == "krum":
            global_state = _krum_aggregate(payloads, malicious_ratio=mal_ratio)
        else:
            global_state = _aggregate(payloads)
            
        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, test_loader, device)
        print(f"[Round {round_idx + 1}] FedAvg ({aggregator_name}) Accuracy: {acc * 100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in global_state.items()}

    save_path = config.get("save_checkpoint")
    if save_path:
        to_save = best_state or {k: v.cpu() for k, v in global_state.items()}
        # Save aggregator info in metadata
        save_checkpoint(to_save, save_path, meta={"method": "fedavg", "aggregator": aggregator_name, "best_acc": best_acc})


def _parse_args():
    parser = argparse.ArgumentParser(description="FedAvg baseline aligned with SB-HFRL configs.")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config JSON file.")
    parser.add_argument("--save-ckpt", type=str, default=None, help="Optional path to save the best checkpoint.")
    parser.add_argument(
        "--aggregator", 
        type=str, 
        default="avg", 
        choices=["avg", "trim_mean", "krum"], 
        help="Aggregation method defense against attacks: 'avg' (default), 'trim_mean', or 'krum'."
    )
    
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)
    if args.save_ckpt:
        config["save_checkpoint"] = args.save_ckpt
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    
    # Pass the aggregator choice to the runner
    run_fedavg(config, device, aggregator_name=args.aggregator)


if __name__ == "__main__":
    main()