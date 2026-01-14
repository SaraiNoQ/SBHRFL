import argparse
import random
import copy
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 保持你的原始引用
from sbhfrl.data_utils import build_loaders, compute_prototypes, dirichlet_partition, get_dataset
from sbhfrl.federated.aggregation import _avg_prototypes, _avg_state_dicts
from sbhfrl.losses import BaseProtoLoss, HDIBLoss, supervised_contrastive
from sbhfrl.models import build_model
from sbhfrl.optim import Muon
from sbhfrl.utils import evaluate, get_device, load_config, save_checkpoint, set_seed


# --- [辅助函数] 防御算法工具 ---

def flatten_state_dict(state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """将 state_dict 扁平化为一个一维向量 (只处理浮点参数，忽略整数buffer以防报错)"""
    # 过滤掉非浮点类型的参数（如 BatchNorm 的 num_batches_tracked），因为它们不参与欧氏距离计算
    return torch.cat([param.view(-1).float() for param in state_dict.values() if param.is_floating_point()])

def unflatten_state_dict(flat_vector: torch.Tensor, ref_state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """将一维向量还原为 state_dict 结构"""
    restored = {}
    offset = 0
    for k, v in ref_state_dict.items():
        # 只还原浮点参数
        if v.is_floating_point():
            numel = v.numel()
            restored[k] = flat_vector[offset : offset + numel].view(v.shape).to(v.device).type(v.dtype)
            offset += numel
        else:
            # 对于非浮点参数（如整数计数器），直接保留原始值，不更新（或者可以采用众数，这里简化处理）
            restored[k] = v
    return restored

def _aggregate_krum(payloads: List[Dict], byzantine_ratio: float = 0.2, multi_k: bool = True) -> Dict[str, torch.Tensor]:
    n = len(payloads)
    f = int(n * byzantine_ratio)
    # Krum 要求的最小客户端数量限制
    if n <= 2 * f + 2: 
        f = max(0, n - 3)

    ref_dict = payloads[0]["state_dict"]
    
    # 1. 扁平化
    client_vectors = [flatten_state_dict(p["state_dict"]) for p in payloads]
    client_stack = torch.stack(client_vectors) # (n, d)

    # 2. 计算距离矩阵
    dists = torch.cdist(client_stack, client_stack, p=2)

    # 3. Krum Score 计算
    num_neighbors = max(1, n - f - 2)
    scores = []
    for i in range(n):
        # 取最近的 num_neighbors 个邻居（排除自己）
        sorted_dists, _ = torch.topk(dists[i], k=num_neighbors + 1, largest=False) 
        scores.append(torch.sum(sorted_dists[1:]))
    scores = torch.tensor(scores)

    # 4. 选择
    if multi_k:
        m = max(1, n - f) # 选择前 n-f 个最好的
        _, indices = torch.topk(scores, k=m, largest=False)
        aggregated_vec = torch.mean(client_stack[indices], dim=0)
    else:
        idx = torch.argmin(scores)
        aggregated_vec = client_stack[idx]

    # 5. 还原
    return unflatten_state_dict(aggregated_vec, ref_dict)


def _aggregate_trimmed_mean(payloads: List[Dict], byzantine_ratio: float = 0.2) -> Dict[str, torch.Tensor]:
    n = len(payloads)
    beta = int(n * byzantine_ratio)
    if 2 * beta >= n: beta = int((n - 1) / 2)
    
    ref_dict = payloads[0]["state_dict"]
    agg_dict = {}

    for key, val in ref_dict.items():
        # 如果是 LongTensor (如 BN 的计数器)，不适合做 Trimmed Mean，直接取平均或众数
        if not val.is_floating_point():
             # 简单的处理：取第一个客户端的值，或者取所有客户端的 float 平均再取整
             # 这里为了稳定，直接取 float 平均再取整
             stacked = torch.stack([p["state_dict"][key].float() for p in payloads])
             agg_dict[key] = torch.mean(stacked, dim=0).type(val.dtype)
             continue

        # 正常浮点参数处理
        stacked = torch.stack([p["state_dict"][key] for p in payloads])
        sorted_params, _ = torch.sort(stacked, dim=0)
        
        if beta > 0:
            trimmed = sorted_params[beta : n - beta]
        else:
            trimmed = sorted_params
            
        agg_dict[key] = torch.mean(trimmed, dim=0)

    return agg_dict


# --- Client 类 ---

def _temperatured_kl(student_logits: torch.Tensor, teacher_probs: torch.Tensor, temperature: float) -> torch.Tensor:
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature ** 2)


class FedMPSClient:
    def __init__(self, client_id: int, loader: DataLoader, config: Dict, is_attacker: bool = False):
        self.client_id = client_id
        self.loader = loader
        self.num_samples = len(loader.dataset)
        self.config = config
        self.use_hdib = config.get("model", "base") == "hdib"
        self.ce_loss = HDIBLoss(config) if self.use_hdib else BaseProtoLoss()
        
        # Hyperparameters
        self.lambda_contrast = float(config.get("lambda_proto_contrast", 0.2))
        self.lambda_soft = float(config.get("lambda_soft_label", 0.6))
        self.lambda_align = float(config.get("lambda_proto_align", 0.3))
        self.proto_temp = float(config.get("proto_temperature", 0.7))
        
        # Attack flag
        self.is_attacker = is_attacker

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
        sim = torch.matmul(F.normalize(embeddings, dim=1), global_proto.t()) / self.proto_temp
        return F.softmax(sim, dim=1)

    def _malicious_payload(self, global_state: Dict[str, torch.Tensor], device: torch.device) -> Dict:
        """生成恶意数据：随机噪声"""
        num_classes = self.config.get("num_classes", 10)
        
        # 尝试推断特征维度
        temp_model = build_model(self.config)
        if hasattr(temp_model, "fc"):
            feat_dim = temp_model.fc.in_features
        elif hasattr(temp_model, "classifier"):
             feat_dim = temp_model.classifier.in_features
        elif hasattr(temp_model, "linear"): # MobileNet etc
             feat_dim = temp_model.linear.in_features
        else:
            feat_dim = 512 # Fallback
            
        # 1. 攻击 Prototypes: 巨大的随机噪声
        malicious_protos = torch.randn((num_classes, feat_dim)) * 1.0
        
        # 2. 攻击 Weights: 巨大的随机噪声
        malicious_state = {}
        for k, v in global_state.items():
            if v.is_floating_point():
                malicious_state[k] = torch.randn_like(v) * 1.0
            else:
                malicious_state[k] = v # 整数参数不改动，避免报错

        return {
            "client_id": self.client_id,
            "state_dict": malicious_state,
            "prototypes": malicious_protos,
            "num_samples": self.num_samples,
        }

    def train(
        self,
        global_state: Dict[str, torch.Tensor],
        global_proto: Optional[torch.Tensor],
        device: torch.device,
    ) -> Dict:
        # 如果是攻击者，直接返回恶意载荷
        if self.is_attacker:
            return self._malicious_payload(global_state, device)

        # 正常训练
        model = build_model(self.config).to(device)
        model.load_state_dict(global_state)
        
        # [关键修复] 显式开启训练模式，这对 BatchNorm 和 Dropout 至关重要
        model.train() 
        
        optimizer = self._build_optimizer(model)

        for _ in range(self.config.get("local_epochs", 1)):
            for images, labels in self.loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                logits, embeddings = model(images)

                # 计算损失
                if self.use_hdib:
                    aux = getattr(model, "aux", {})
                    base_loss = self.ce_loss(
                        logits, labels, mus=aux.get("mus", []),
                        logvars=aux.get("logvars", []),
                        sampled_feats=aux.get("sampled_feats", []),
                        fused_repr=aux.get("embeddings", embeddings),
                    )
                else:
                    base_loss = self.ce_loss(logits, labels)

                loss = base_loss + self.lambda_contrast * supervised_contrastive(embeddings, labels)

                if global_proto is not None:
                    proto_device = global_proto.to(device)
                    # 只有当 proto 包含正常数值时才计算，防止 NaN 传播
                    if not torch.isnan(proto_device).any():
                        soft_target = self._soft_labels(embeddings, proto_device)
                        loss = loss + self.lambda_soft * _temperatured_kl(logits, soft_target, self.proto_temp)
                        proto_targets = proto_device[labels]
                        loss = loss + self.lambda_align * F.mse_loss(F.normalize(embeddings, dim=1), proto_targets)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get("max_grad_norm", 5.0))
                optimizer.step()

        # 计算原型 (原型计算通常在 eval 模式下或保持当前模式，这里用 eval 更稳妥)
        model.eval()
        prototypes = compute_prototypes(
            model, 
            DataLoader(self.loader.dataset, batch_size=self.config.get("proto_batch_size", 128), shuffle=False), 
            self.config.get("num_classes", 10), 
            device
        )
        
        return {
            "client_id": self.client_id,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "prototypes": prototypes.cpu(),
            "num_samples": self.num_samples,
        }


# --- 聚合逻辑 ---

def _aggregate_state(payloads: List[Dict], config: Dict) -> Dict[str, torch.Tensor]:
    method = config.get("defense", "none").lower()
    ratio = config.get("byzantine_ratio", 0.0)

    # 如果无防御，严格按照原有逻辑执行
    if method == "none" or ratio <= 0:
        weights = torch.tensor([p["num_samples"] for p in payloads], dtype=torch.float32)
        # 保持原来的 float() 转换，因为 sbhfrl._avg_state_dicts 需要它
        state_dicts = [{k: v.float() for k, v in p["state_dict"].items()} for p in payloads]
        return _avg_state_dicts(state_dicts, weights)
        
    elif method == "krum":
        return _aggregate_krum(payloads, byzantine_ratio=ratio, multi_k=True)
    elif method == "trim":
        return _aggregate_trimmed_mean(payloads, byzantine_ratio=ratio)
    else:
        raise ValueError(f"Unknown defense method: {method}")


def _aggregate_proto(payloads: List[Dict], config: Dict) -> torch.Tensor:
    method = config.get("defense", "none").lower()
    ratio = config.get("byzantine_ratio", 0.0)

    if method == "none" or ratio <= 0:
        weights = torch.tensor([p["num_samples"] for p in payloads], dtype=torch.float32)
        protos = [p["prototypes"].float() for p in payloads]
        return _avg_prototypes(protos, weights)
    else:
        # 防御模式：对 Prototype 统一使用 Trimmed Mean
        protos_stack = torch.stack([p["prototypes"].float() for p in payloads])
        n = len(payloads)
        beta = int(n * ratio)
        if 2 * beta >= n: beta = int((n - 1) / 2)
        
        sorted_protos, _ = torch.sort(protos_stack, dim=0)
        if beta > 0:
            trimmed = sorted_protos[beta : n - beta]
        else:
            trimmed = sorted_protos
        return torch.mean(trimmed, dim=0)


# --- 主程序 ---

def run_fedmps(config: Dict, device: torch.device) -> None:
    train_dataset, test_dataset = get_dataset(config)
    num_clients = config["num_shards"] * config["clients_per_shard"]
    subsets = dirichlet_partition(train_dataset, num_clients, config.get("alpha_dirichlet", 0.5))
    loaders = build_loaders(subsets, config.get("batch_size", 64), num_workers=config.get("data_num_workers", 0))
    
    # [关键修复] 确定性攻击者选择，不破坏全局随机种子
    ratio = config.get("byzantine_ratio", 0.0)
    num_attackers = int(num_clients * ratio)
    
    # 简单的取前 N 个作为攻击者。因为数据已经是随机划分的，所以这等同于随机攻击者。
    # 不要调用 random.shuffle(all_indices)！这会改变后续 random.sample 的结果。
    attacker_indices = set(range(num_attackers))
    
    if num_attackers > 0:
        print(f"Warning: Byzantine Attack Enabled! {num_attackers} clients are malicious.")

    clients = []
    for idx, loader in enumerate(loaders):
        is_bad = idx in attacker_indices
        clients.append(FedMPSClient(idx, loader, config, is_attacker=is_bad))

    global_model = build_model(config).to(device)
    global_state = {k: v.cpu() for k, v in global_model.state_dict().items()}
    global_proto: Optional[torch.Tensor] = None

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
        
        # 这里的采样序列现在应该和原始代码完全一致
        selected = random.sample(clients, participate)
        
        payloads = [client.train(global_state, global_proto, device) for client in selected]
        
        # 聚合
        global_state = _aggregate_state(payloads, config)
        global_proto = _aggregate_proto(payloads, config)
        
        # 评估
        global_model.load_state_dict(global_state)
        acc = evaluate(global_model, test_loader, device)
        
        defense_status = config.get('defense', 'none')
        att_ratio = config.get('byzantine_ratio', 0.0)
        print(f"[Round {round_idx + 1}] FedMPS (Def={defense_status}, Att={att_ratio}) Acc: {acc * 100:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu() for k, v in global_state.items()}

    save_path = config.get("save_checkpoint")
    if save_path:
        to_save = best_state or {k: v.cpu() for k, v in global_state.items()}
        save_checkpoint(to_save, save_path, meta={"method": "fedmps", "best_acc": best_acc})


def _parse_args():
    parser = argparse.ArgumentParser(description="FedMPS with Defense and Attacks.")
    parser.add_argument("--config", type=str, default="configs/default.json")
    parser.add_argument("--save-ckpt", type=str, default=None)
    parser.add_argument("--defense", type=str, default="none", choices=["none", "krum", "trim"])
    parser.add_argument("--byzantine", type=float, default=0.0)
    return parser.parse_args()


def main():
    args = _parse_args()
    config = load_config(args.config)
    
    if args.save_ckpt: config["save_checkpoint"] = args.save_ckpt
    config["defense"] = args.defense
    config["byzantine_ratio"] = args.byzantine

    # 设置种子
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    
    run_fedmps(config, device)


if __name__ == "__main__":
    main()