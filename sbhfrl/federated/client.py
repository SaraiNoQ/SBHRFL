from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data_utils import compute_prototypes
from ..losses import BaseProtoLoss, HDIBLoss
from ..models import build_model
from ..optim import Muon


class ClientNode:
    def __init__(self, client_id: int, shard_id: int, loader: DataLoader, config: Dict, malicious: bool = False):
        self.client_id = client_id
        self.shard_id = shard_id
        self.loader = loader
        self.num_samples = len(loader.dataset)
        self.config = config
        self.proto_loader = DataLoader(loader.dataset, batch_size=config.get("proto_batch_size", 128), shuffle=False)
        self.reputation = config.get("init_reputation", 0.8)
        self.malicious = malicious
        self.use_hdib = config.get("model", "base") == "hdib"
        self.criterion = HDIBLoss(config) if self.use_hdib else BaseProtoLoss()

    def run_round(
        self,
        global_state: Dict[str, torch.Tensor],
        teacher_proto: Optional[torch.Tensor],
        device: torch.device,
    ) -> Dict:
        model = build_model(self.config).to(device)
        model.load_state_dict(global_state)
        if self.malicious:
            # 恶意客户端：跳过正常训练，上传随机状态和原型
            fake_state = {k: torch.randn_like(v).cpu() for k, v in global_state.items()}
            proto = torch.randn(self.config.get("num_classes", 10), model.embedding_dim, device=device)
            proto = torch.nn.functional.normalize(proto, dim=1)
            payload = {
                "client_id": self.client_id,
                "shard_id": self.shard_id,
                "state_dict": fake_state,
                "prototypes": proto.cpu(),
                "num_samples": self.num_samples,
                "metrics": {"density": 0.0, "channel": 0.0, "reputation": 0.0},
                "malicious": True,
            }
            return payload

        betas = self.config.get("muon_betas", [0.9, 0.99])
        optimizer = Muon(
            model.parameters(),
            lr=self.config.get("lr", 0.01),
            betas=(betas[0], betas[1]),
            eps=self.config.get("muon_eps", 1e-8),
            weight_decay=self.config.get("weight_decay", 1e-4),
        )
        # if self.use_hdib:
        #     betas = self.config.get("muon_betas", [0.9, 0.99])
        #     optimizer = Muon(
        #         model.parameters(),
        #         lr=self.config.get("lr", 0.01),
        #         betas=(betas[0], betas[1]),
        #         eps=self.config.get("muon_eps", 1e-8),
        #         weight_decay=self.config.get("weight_decay", 1e-4),
        #     )
        # else:
        #     optimizer = torch.optim.SGD(
        #         model.parameters(),
        #         lr=self.config.get("lr", 0.01),
        #         momentum=self.config.get("momentum", 0.9),
        #         weight_decay=self.config.get("weight_decay", 1e-4),
        #     )
        for _ in range(self.config.get("local_epochs", 1)):
            for images, labels in self.loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits, embeddings = model(images)
                if self.use_hdib:
                    aux = getattr(model, "aux", {})
                    teacher = teacher_proto.to(device) if teacher_proto is not None else None
                    loss = self.criterion(
                        logits,
                        labels,
                        mus=aux.get("mus", []),
                        logvars=aux.get("logvars", []),
                        sampled_feats=aux.get("sampled_feats", []),
                        fused_repr=aux.get("embeddings", embeddings),
                        teacher_proto=teacher,
                    )
                else:
                    loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.get("max_grad_norm", 5.0))
                optimizer.step()
        prototypes = compute_prototypes(model, self.proto_loader, self.config.get("num_classes", 10), device)
        payload = {
            "client_id": self.client_id,
            "shard_id": self.shard_id,
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "prototypes": prototypes.cpu(),
            "num_samples": self.num_samples,
            "metrics": {
                "density": float(np.clip(torch.mean(torch.norm(prototypes, dim=1)).item(), 0.0, 10.0)),
                "channel": float(np.random.beta(5, 2)),
                "reputation": self.reputation,
            },
        }
        return payload
