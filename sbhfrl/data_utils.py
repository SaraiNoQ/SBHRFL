from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision
import torchvision.transforms as transforms


def get_cifar10_datasets(root: str = "./data") -> Tuple[Dataset, Dataset]:
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


def dirichlet_partition(dataset: Dataset, num_clients: int, alpha: float) -> List[Subset]:
    labels = np.array(dataset.targets)
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


def build_loaders(subsets: List[Subset], batch_size: int) -> List[DataLoader]:
    loaders = []
    for subset in subsets:
        loaders.append(DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=len(subset) >= batch_size))
    return loaders


def compute_prototypes(model: torch.nn.Module, loader: DataLoader, num_classes: int, device: torch.device) -> torch.Tensor:
    model.eval()
    proto = torch.zeros(num_classes, model.embedding_dim, device=device)
    counts = torch.zeros(num_classes, device=device)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            _, embeddings = model(images)
            proto.index_add_(0, labels, embeddings)
            counts.index_add_(0, labels, torch.ones_like(labels, dtype=torch.float))
    counts = counts.clamp(min=1.0).unsqueeze(1)
    proto = proto / counts
    proto = torch.nn.functional.normalize(proto, dim=1)
    return proto
