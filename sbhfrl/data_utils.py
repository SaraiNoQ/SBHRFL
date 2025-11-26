import os
from bisect import bisect_right
from typing import Dict, List, Tuple, Sequence, Optional

import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Dataset, Subset
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class CIFARCTransformDataset(Dataset):
    """支持多 corruption 文件、可选 train/test 划分、惰性增广的 CIFAR-C Dataset。"""
    def __init__(
        self,
        root: str,
        corruption_dir: str,
        corruptions: Optional[Sequence[str]] = None,  # e.g. ["gaussian_noise", "brightness"]
        severities: Optional[Sequence[int]] = None,    # e.g. [1,2,3,4,5]
        train: bool = True,
        train_ratio: float = 0.8,
        augment: bool = True,
    ):
        full_dir = os.path.join(root, corruption_dir)
        if not os.path.isdir(full_dir):
            raise FileNotFoundError(f"CIFAR-C folder not found: {full_dir}")

        files = [f for f in os.listdir(full_dir) if f.endswith(".npy") and f != "labels.npy"]
        if corruptions:
            files = [f for f in files if any(corr in f for corr in corruptions)]
        if severities:
            files = [f for f in files if any(f"s{sev}" in f for sev in severities)]
        files.sort()
        if not files:
            raise FileNotFoundError(f"No corruption .npy files matched under {full_dir}")

        self.images_list = []
        self.cum_lengths = []
        total = 0
        for f in files:
            arr = np.load(os.path.join(full_dir, f), mmap_mode="r")
            self.images_list.append(arr)
            total += len(arr)
            self.cum_lengths.append(total)
        self.total_len = total

        self.labels = np.load(os.path.join(full_dir, "labels.npy"), mmap_mode="r")
        indices = np.arange(self.total_len)
        rng = np.random.default_rng(42)
        rng.shuffle(indices)
        split = int(self.total_len * train_ratio)
        self.indices = indices[:split] if train else indices[split:]

        self.normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        self.aug = None
        if train and augment:
            self.aug = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4),
            ])

    def __len__(self):
        return len(self.indices)

    def _locate(self, global_idx: int):
        file_idx = bisect_right(self.cum_lengths, global_idx)
        prev = 0 if file_idx == 0 else self.cum_lengths[file_idx - 1]
        inner_idx = global_idx - prev
        return file_idx, inner_idx

    def __getitem__(self, idx):
        global_idx = int(self.indices[idx])
        file_idx, inner_idx = self._locate(global_idx)
        img = self.images_list[file_idx][inner_idx]  # HWC uint8
        if not img.flags["WRITEABLE"]:
            img = np.array(img, copy=True)  # ensure torch sees a writable buffer, avoids warning
        img = torch.from_numpy(img).permute(2, 0, 1).float().div_(255.0)
        if self.aug:
            img = self.aug(img)
        img = self.normalize(img)
        label = int(self.labels[inner_idx % len(self.labels)])
        return img, label

# class CIFARCTransformDataset(torch.utils.data.Dataset):
#     """支持多 corruption 文件、可选 train/test 划分、训练增广的 CIFAR-C Dataset。"""
#     def __init__(
#         self,
#         root: str,
#         corruption_dir: str,
#         corruptions: Optional[Sequence[str]] = None,  # e.g. ["gaussian_noise", "brightness"]
#         severities: Optional[Sequence[int]] = None,    # e.g. [1,2,3,4,5]
#         train: bool = True,
#         train_ratio: float = 0.8,
#         augment: bool = True,
#     ):
#         full_dir = os.path.join(root, corruption_dir)
#         if not os.path.isdir(full_dir):
#             raise FileNotFoundError(f"CIFAR-C folder not found: {full_dir}")

#         # 收集候选 corruption 文件
#         files = [f for f in os.listdir(full_dir) if f.endswith(".npy") and f != "labels.npy"]
#         if corruptions:
#             files = [f for f in files if any(corr in f for corr in corruptions)]
#         if severities:
#             files = [f for f in files if any(f"s{sev}" in f for sev in severities)]
#         files.sort()
#         if not files:
#             raise FileNotFoundError(f"No corruption .npy files matched under {full_dir}")

#         # 读取并拼接多 corruption
#         imgs_list = []
#         for f in files:
#             imgs_list.append(np.load(os.path.join(full_dir, f)))
#         images = np.concatenate(imgs_list, axis=0)  # [N, 32, 32, 3]
#         labels = np.load(os.path.join(full_dir, "labels.npy"))
#         # 若 labels 与单个 corruption 对应，需要重复以匹配拼接后的大小
#         if images.shape[0] != labels.shape[0]:
#             reps = images.shape[0] // labels.shape[0]
#             labels = np.tile(labels, reps)

#         # 划分 train/test
#         num_total = images.shape[0]
#         indices = np.arange(num_total)
#         rng = np.random.default_rng(42)  # 固定随机种子，便于复现
#         rng.shuffle(indices)
#         split = int(num_total * train_ratio)
#         chosen = indices[:split] if train else indices[split:]
#         images = images[chosen]
#         labels = labels[chosen]

#         images = torch.tensor(images).permute(0, 3, 1, 2).float() / 255.0
#         normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
#         if train and augment:
#             aug = transforms.Compose([
#                 transforms.RandomHorizontalFlip(),
#                 transforms.RandomCrop(32, padding=4),
#             ])
#             images = aug(images)
#         images = normalize(images)
#         self.images = images
#         self.labels = torch.tensor(labels, dtype=torch.long)

#     def __len__(self):
#         return len(self.labels)

#     def __getitem__(self, idx):
#         return self.images[idx], self.labels[idx]


def get_cifar_c(root: str, corruption_dir: str) -> Dataset:
    """
    默认返回训练集；测试集请使用 get_cifar_c_test。
    若只需要一个 Dataset（当前代码路径），可仍然用此函数，但需注意 train_ratio。
    """
    return CIFARCTransformDataset(root, corruption_dir, train=True)


def get_cifar_c_test(root: str, corruption_dir: str) -> Dataset:
    """与训练集分开取的测试集。"""
    return CIFARCTransformDataset(root, corruption_dir, train=False, augment=False)

def get_dataset(config: Dict) -> Tuple[Dataset, Dataset]:
    name = config.get("dataset", "cifar10").lower()
    root = config.get("data_root", "./data")
    if name == "cifar10":
        return get_cifar10(root)
    if name in {"cifar100c", "cifar100-c"}:
        train_ds = get_cifar_c(root, corruption_dir="cifar100-c")
        test_ds = get_cifar_c_test(root, corruption_dir="cifar100-c")
        return train_ds, test_ds
    if name in {"office-caltech", "office_caltech"}:
        return get_office_caltech(root)
    if name in {"domainnet-car", "domainnet_car"}:
        return get_domainnet_car(root)
    if name == "vlcs":
        return get_vlcs(root)
    raise ValueError("Unsupported dataset. Choose 'cifar10', 'cifar100c', 'office-caltech', or 'domainnet-car'.")


def get_cifar10(root: str) -> Tuple[Dataset, Dataset]:
    transform_train = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    train_dataset = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset


def get_office_caltech(root: str) -> Tuple[Dataset, Dataset]:
    """Office-Caltech loader expects ImageFolder structure under root/office_caltech/{train,test}."""
    import os

    train_dir = os.path.join(root, "office_caltech", "train")
    test_dir = os.path.join(root, "office_caltech", "test")
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        raise FileNotFoundError(
            "Office-Caltech folders not found. Expected train/ and test/ under data_root/office_caltech/."
        )
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    train_dataset = ImageFolder(train_dir, transform=transform_train)
    test_dataset = ImageFolder(test_dir, transform=transform_test)
    return train_dataset, test_dataset


def get_domainnet_car(root: str) -> Tuple[Dataset, Dataset]:
    """DomainNet-Car subset prepared as ImageFolder under root/DomainNet-Car/train|test."""
    train_dir = os.path.join(root, "DomainNet-Car", "train")
    test_dir = os.path.join(root, "DomainNet-Car", "test")
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        raise FileNotFoundError("Expected DomainNet-Car/train and DomainNet-Car/test under data_root.")

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = ImageFolder(train_dir, transform=transform_train)
    test_dataset = ImageFolder(test_dir, transform=transform_test)
    return train_dataset, test_dataset

def get_vlcs(root: str) -> Tuple[Dataset, Dataset]:
    """VLCS prepared as ImageFolder under root/VLCS/train|test."""
    train_dir = os.path.join(root, "VLCS", "train")
    test_dir = os.path.join(root, "VLCS", "test")
    if not (os.path.isdir(train_dir) and os.path.isdir(test_dir)):
        raise FileNotFoundError("Expected VLCS/train and VLCS/test under data_root.")
    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform_train = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = ImageFolder(train_dir, transform=transform_train)
    test_dataset = ImageFolder(test_dir, transform=transform_test)
    return train_dataset, test_dataset


def _extract_labels(dataset: Dataset) -> np.ndarray:
    """Best-effort label extraction supporting TensorDataset and torchvision datasets."""
    if hasattr(dataset, "targets"):
        labels = dataset.targets
    elif hasattr(dataset, "labels"):  # Some datasets use .labels instead of .targets
        labels = dataset.labels
    elif hasattr(dataset, "tensors") and len(dataset.tensors) >= 2:
        labels = dataset.tensors[1]
    else:  # Fallback to indexing; may be slower but keeps function robust.
        labels = [dataset[idx][1] for idx in range(len(dataset))]

    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()
    else:
        labels = np.array(labels)
    return labels


def dirichlet_partition(dataset: Dataset, num_clients: int, alpha: float) -> List[Subset]:
    labels = _extract_labels(dataset)
    num_classes = len(np.unique(labels))
    idx_by_class = [np.where(labels == cls)[0] for cls in range(num_classes)]
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for class_indices in idx_by_class:
        np.random.shuffle(class_indices)
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)
        start = 0
        for client_id, end in enumerate(proportions):
            client_indices[client_id].extend(class_indices[start:end])
            start = end
        client_indices[-1].extend(class_indices[start:])
    return [Subset(dataset, indices) for indices in client_indices]


def build_loaders(subsets: List[Subset], batch_size: int, num_workers: int = 0) -> List[DataLoader]:
    loaders = []
    for subset in subsets:
        loaders.append(
            DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=len(subset) >= batch_size,
            )
        )
    return loaders


def compute_prototypes(model: torch.nn.Module, loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    model.eval()
    proto = torch.zeros(num_classes, model.embedding_dim, device=device)
    counts = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).long()
            _, embeddings = model(images)
            proto.index_add_(0, labels, embeddings)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    counts = counts.clamp(min=1.0).unsqueeze(1)
    proto = proto / counts
    proto = torch.nn.functional.normalize(proto, dim=1)
    return proto
