import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time

# -------------------------
# Simple CNN model
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Training function
# -------------------------
def train_one_epoch(model, loader, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    start = time.time()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Sync for accurate timing
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    return end - start


# -------------------------
# Main benchmark
# -------------------------
def benchmark(device, batch_size=128, num_samples=5000):
    print(f"\n=== Testing device: {device} ===")

    # Fake CIFAR-10 style dataset
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 10, (num_samples,))

    dataset = TensorDataset(images, labels)

    # num_workers=0 → 避免 macOS 多进程问题
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SimpleCNN().to(device)

    time_cost = train_one_epoch(model, loader, device)
    print(f"Time on {device}: {time_cost:.3f} s")

    return time_cost


if __name__ == "__main__":
    # CPU benchmark
    cpu_time = benchmark(torch.device("cpu"))

    # MPS benchmark
    if torch.backends.mps.is_available():
        mps_time = benchmark(torch.device("mps"))
        print("\n🔥 Speedup (CPU / MPS): {:.2f}x".format(cpu_time / mps_time))
    else:
        print("❌ MPS not available")
