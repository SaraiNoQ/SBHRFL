import random
from collections import defaultdict
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .data_utils import build_loaders, dirichlet_partition, get_dataset
from .federated.aggregation import QualityAwareAggregator, SimpleAggregator
from .federated.blockchain import BlockchainMemory
from .federated.client import ClientNode
from .federated.consensus import ReputationConsensus, SimpleConsensus
from .models import build_model
from .utils import evaluate, save_checkpoint


def _create_clients(loaders: List[DataLoader], config: Dict, malicious_ids=None) -> List[ClientNode]:
    clients = []
    shard_size = config["clients_per_shard"]
    for idx, loader in enumerate(loaders):
        shard_id = idx // shard_size
        is_malicious = malicious_ids is not None and idx in malicious_ids
        clients.append(ClientNode(idx, shard_id, loader, config, malicious=is_malicious))
    return clients


def run_federated_training(config: Dict, device: torch.device) -> None:
    train_dataset, test_dataset = get_dataset(config)
    num_clients = config["num_shards"] * config["clients_per_shard"]
    num_malicious = int(num_clients * config.get("mal_ratio", 0.0))
    malicious_ids = set(random.sample(range(num_clients), num_malicious)) if num_malicious > 0 else set()
    subsets = dirichlet_partition(train_dataset, num_clients, config.get("alpha_dirichlet", 0.5))
    loaders = build_loaders(subsets, config.get("batch_size", 64), num_workers=config.get("data_num_workers", 0))
    clients = _create_clients(loaders, config, malicious_ids)
    shard_groups: Dict[int, List[ClientNode]] = defaultdict(list)
    for client in clients:
        shard_groups[client.shard_id].append(client)

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
    blockchain = (
        BlockchainMemory(config.get("history", 4), config.get("ema_decay", 0.7))
        if config.get("use_blockchain_memory", False)
        else None
    )

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
        shard_summaries = []
        teacher_proto = blockchain.teacher() if blockchain is not None else None
        for shard_id, shard_clients in shard_groups.items():
            participate = min(config.get("clients_per_round", 2), len(shard_clients))
            selected = random.sample(shard_clients, participate)
            payloads = []
            for client in selected:
                payload = client.run_round(global_state, teacher_proto, device)
                payloads.append(payload)
            shard_summary = aggregator.fuse(shard_id, payloads)
            shard_summaries.append(shard_summary)
        global_state, global_proto = consensus.aggregate(shard_summaries)
        if blockchain is not None:
            blockchain.update(global_proto)
        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, test_loader, device)
        enabled_components = []
        if config.get("model") == "hdib":
            enabled_components.append("HD-IB")
        if config.get("use_quality_fusion"):
            enabled_components.append("QualityFusion")
        if config.get("use_reputation_consensus"):
            enabled_components.append("LedgerConsensus")
        if config.get("use_blockchain_memory"):
            enabled_components.append("MemoryDistill")
        if use_cluster:
            enabled_components.append("Clustering")
        tag = "+".join(enabled_components) if enabled_components else "FedProto"
        print(f"[Round {round_idx + 1}] {tag} Accuracy: {acc * 100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in global_state.items()}

    save_path = config.get("save_checkpoint")
    if save_path:
        to_save = best_state or {k: v.cpu() for k, v in global_state.items()}
        tag = config.get("model", "base")
        save_checkpoint(to_save, save_path, meta={"method": tag, "best_acc": best_acc})
