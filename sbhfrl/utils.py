import json
import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    """
    返回设备：
    - "auto": 优先 CUDA，其次 MPS，最后 CPU
    - 其它值直接传给 torch.device()
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    # 如果用户手动指定设备
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS not available on this machine.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA not available on this machine.")

    return torch.device(requested)


def load_config(path: str) -> Dict:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {path} not found.")
    with config_path.open() as f:
        data = json.load(f)
    return data


def save_checkpoint(state_dict: Dict[str, torch.Tensor], path: str, meta: Optional[Dict] = None) -> None:
    """
    Persist a model state_dict with optional metadata.
    """
    payload: Dict = {"state_dict": state_dict}
    if meta:
        payload.update(meta)
    path_obj = Path(path)
    if path_obj.parent:
        os.makedirs(path_obj.parent, exist_ok=True)
    torch.save(payload, path_obj)
    print(f"[Checkpoint] Saved to {path_obj}")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        logits, _ = model(images)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / max(total, 1)
