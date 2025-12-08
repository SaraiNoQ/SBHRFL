"""
Run feature-shift experiments on CIFAR-100-C across multiple baselines.

The script:
- Assigns corruption types to shards (round-robin) and partitions each shard via Dirichlet.
- Trains the requested method (fedavg, fedmps, feddp, fedprox, sbhfrl) under the existing config.
- Reports per-shard (per-corruption) accuracy and overall global accuracy.
"""
import argparse
import os
import random
import sys
from typing import Dict, List, Tuple

import torch
from torch.utils.data import ConcatDataset, DataLoader

# Ensure repository root is on sys.path when running as a script.
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from sbhfrl.data_utils import CIFARCTransformDataset, dirichlet_partition
from sbhfrl.federated.aggregation import _avg_prototypes, _avg_state_dicts
from sbhfrl.federated.aggregation import QualityAwareAggregator, SimpleAggregator
from sbhfrl.federated.client import ClientNode
from sbhfrl.federated.consensus import ReputationConsensus, SimpleConsensus
from sbhfrl.losses import BaseProtoLoss, HDIBLoss
from sbhfrl.models import build_model
from sbhfrl.optim import Muon
from sbhfrl.utils import evaluate, get_device, load_config, save_checkpoint, set_seed

# Baseline clients
from sbhfrl.fedavg import FedAvgClient
from sbhfrl.fedmps import FedMPSClient
from sbhfrl.feddp import FedDPClient
from sbhfrl.fedprox import FedProxClient


def build_shard_datasets(config: Dict, corruptions: List[str], severities: List[int]) -> List[Tuple[str, CIFARCTransformDataset, CIFARCTransformDataset]]:
    shards = []
    data_root = config.get("data_root", "./data")
    for shard_id in range(config["num_shards"]):
        corr = corruptions[shard_id % len(corruptions)]
        train_ds = CIFARCTransformDataset(
            root=data_root,
            corruption_dir="cifar100-c",
            corruptions=[corr],
            severities=severities if severities else None,
            train=True,
            train_ratio=0.8,
            augment=True,
        )
        test_ds = CIFARCTransformDataset(
            root=data_root,
            corruption_dir="cifar100-c",
            corruptions=[corr],
            severities=severities if severities else None,
            train=False,
            train_ratio=0.8,
            augment=False,
        )
        shards.append((corr, train_ds, test_ds))
    return shards


def build_clients_for_shard(train_ds, shard_id: int, config: Dict, client_cls):
    subsets = dirichlet_partition(train_ds, config["clients_per_shard"], config.get("alpha_dirichlet", 0.5))
    loaders = [
        DataLoader(subset, batch_size=config.get("batch_size", 64), shuffle=True, num_workers=config.get("data_num_workers", 0))
        for subset in subsets
    ]
    clients = []
    for local_id, loader in enumerate(loaders):
        global_id = shard_id * config["clients_per_shard"] + local_id
        # Baseline clients do not track shard id; SB-HFRL client does.
        if client_cls is ClientNode:
            clients.append(ClientNode(global_id, shard_id, loader, config))
        else:
            clients.append(client_cls(global_id, loader, config))
    return clients


def eval_on_loaders(model: torch.nn.Module, loaders: List[Tuple[str, DataLoader]], device: torch.device) -> Dict[str, float]:
    accs = {}
    for name, loader in loaders:
        accs[name] = evaluate(model, loader, device)
    return accs


def aggregate_fedavg(payloads):
    weights = torch.tensor([p["num_samples"] for p in payloads], dtype=torch.float32)
    state_dicts = [{k: v.float() for k, v in p["state_dict"].items()} for p in payloads]
    return _avg_state_dicts(state_dicts, weights)


def aggregate_proto(payloads):
    weights = torch.tensor([p["num_samples"] for p in payloads], dtype=torch.float32)
    protos = [p["prototypes"].float() for p in payloads]
    return _avg_prototypes(protos, weights)


def run_method(method: str, config: Dict, shards: List[Tuple[str, CIFARCTransformDataset, CIFARCTransformDataset]], device: torch.device):
    # Build clients grouped by shard
    shard_clients: Dict[int, List] = {}
    if method == "fedavg":
        client_cls = FedAvgClient
    elif method == "fedmps":
        client_cls = FedMPSClient
    elif method == "feddp":
        client_cls = FedDPClient
    elif method == "fedprox":
        client_cls = FedProxClient
    else:
        client_cls = ClientNode  # SB-HFRL

    for shard_id, (_, train_ds, _) in enumerate(shards):
        shard_clients[shard_id] = build_clients_for_shard(train_ds, shard_id, config, client_cls)

    global_model = build_model(config).to(device)
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
    global_proto = None

    # Prepare test loaders per shard and combined
    test_loaders = []
    for corr, _, test_ds in shards:
        loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=config.get("data_num_workers", 0))
        test_loaders.append((corr, loader))
    combined = DataLoader(
        ConcatDataset([ds for _, _, ds in shards]),
        batch_size=256,
        shuffle=False,
        num_workers=config.get("data_num_workers", 0),
    )

    rounds = config.get("rounds", 1)
    for _ in range(rounds):
        if method in {"fedavg", "fedprox"}:
            payloads = []
            for shard_id, clients in shard_clients.items():
                participate = min(config.get("clients_per_round", len(clients)), len(clients))
                selected = random.sample(clients, participate)
                for client in selected:
                    payloads.append(client.train(global_state, device))
            global_state = aggregate_fedavg(payloads)
        elif method == "fedmps":
            payloads = []
            for shard_id, clients in shard_clients.items():
                participate = min(config.get("clients_per_round", len(clients)), len(clients))
                selected = random.sample(clients, participate)
                for client in selected:
                    payloads.append(client.train(global_state, global_proto, device))
            global_state = aggregate_fedavg(payloads)
            global_proto = aggregate_proto(payloads)
        elif method == "feddp":
            payloads = []
            for shard_id, clients in shard_clients.items():
                participate = min(config.get("clients_per_round", len(clients)), len(clients))
                selected = random.sample(clients, participate)
                for client in selected:
                    payloads.append(client.train(global_state, global_proto, device))
            global_state = aggregate_fedavg(payloads)
            global_proto = aggregate_proto(payloads)
        else:  # sbhfrl
            use_cluster = config.get("use_cluster_prototypes", False)
            cluster_threshold = config.get("cluster_threshold", 0.8)
            aggregator = (
                QualityAwareAggregator(
                    config.get("alpha_quality", 0.6),
                    config.get("beta_quality", 0.2),
                    config.get("gamma_quality", 0.2),
                    use_cluster=use_cluster,
                    cluster_threshold=cluster_threshold,
                )
                if config.get("use_quality_fusion", False)
                else SimpleAggregator(use_cluster=use_cluster, cluster_threshold=cluster_threshold)
            )
            consensus = (
                ReputationConsensus(config.get("wasserstein_threshold", 0.8), config.get("init_reputation", 0.8))
                if config.get("use_reputation_consensus", False)
                else SimpleConsensus()
            )
            shard_summaries = []
            for shard_id, clients in shard_clients.items():
                participate = min(config.get("clients_per_round", len(clients)), len(clients))
                selected = random.sample(clients, participate)
                payloads = []
                for client in selected:
                    payloads.append(client.run_round(global_state, global_proto, device))
                shard_summary = aggregator.fuse(shard_id, payloads)
                shard_summaries.append(shard_summary)
            global_state, global_proto = consensus.aggregate(shard_summaries)

    global_model.load_state_dict(global_state)
    shard_accs = eval_on_loaders(global_model, test_loaders, device)
    global_acc = evaluate(global_model, combined, device)
    return shard_accs, global_acc, {k: v.cpu() for k, v in global_model.state_dict().items()}


def parse_args():
    parser = argparse.ArgumentParser(description="Run CIFAR-100-C feature-shift experiments across baselines.")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config JSON.")
    parser.add_argument(
        "--method",
        type=str,
        default="fedavg",
        choices=["fedavg", "fedmps", "feddp", "fedprox", "sbhfrl"],
        help="Which baseline to run.",
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        default="fog,snow,frost,brightness,contrast",
        help="Comma-separated corruption types to assign to shards (round-robin).",
    )
    parser.add_argument(
        "--severities",
        type=str,
        default="",
        help="Comma-separated severities (1-5); empty for all severities.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="cifar100c_feature_shift_results.txt",
        help="Where to save a human-readable summary.",
    )
    parser.add_argument(
        "--save-ckpt",
        type=str,
        default="",
        help="Optional path to save the final state_dict for the selected method.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))

    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    severities = [int(s) for s in args.severities.split(",") if s.strip()]

    shards = build_shard_datasets(config, corruptions, severities)
    shard_accs, global_acc, final_state = run_method(args.method, config, shards, device)

    lines = []
    lines.append(f"Method: {args.method}")
    lines.append(f"Global Acc: {global_acc * 100:.2f}%")
    for corr, acc in shard_accs.items():
        lines.append(f"{corr}: {acc * 100:.2f}%")

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        for line in lines:
            f.write(line + "\n")
    print("\n".join(lines))
    print(f"Saved summary to {args.out}")

    if args.save_ckpt:
        save_checkpoint(final_state, args.save_ckpt, meta={"method": args.method, "global_acc": global_acc})


if __name__ == "__main__":
    main()
