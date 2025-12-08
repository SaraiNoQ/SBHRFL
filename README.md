## SB-HFRL Simulation

### environment

```
pip install numpy pillow tqdm matplotlib
```

For RTX 50 series GPU, run:
```
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```
and use this to verify avalibility:
```
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_arch_list(), torch.cuda.get_device_capability())"
```

This repository now separates the SB-HFRL simulation into modular components:

- `sbhfrl/models`: prototype backbones (`base`, `hdib`, `resnet10`)
- `sbhfrl/federated`: clients, shard aggregation, consensus, blockchain memory
- `sbhfrl/data_utils.py`: CIFAR-10 loading, non-IID partitioning, prototype extraction
- `sbhfrl/trainer.py`: orchestrates local updates → shard fusion → global consensus
- `configs/default.json`: central place for hyper-parameters and feature toggles
- `run.py`: entry-point that reads a config and executes the experiment

### Usage

1. Install dependencies (`torch`, `torchvision` etc.) inside your Python environment.
2. Adjust `configs/default.json`:
   - Keep `model: "base"` with all `use_*` flags set to `false` to run plain FedProto.
   - Switch to `"hdib"` (Muon optimizer + HD-IB heads, configurable `hdib_backbone`) or `"resnet10"` (standard ResNet backbone) and toggle `use_quality_fusion`, `use_reputation_consensus`, `use_blockchain_memory`, `use_cluster_prototypes` to incrementally enable SB-HFRL components.
3. Launch training:

```bash
python run.py --config configs/default.json
```

```
python -m sbhfrl.fedavg --config configs/default.json
```

```
python run.py --config configs/default.json --save-ckpt checkpoints/sbhfrl_best.pth
```

```
python tools/visualize_partition.py --config configs/default.json --out partition_vis.png
```

```
python tools/eval_cifar100c_feature_shift.py \
  --config configs/default.json \
  --checkpoint path/to/ckpt.pth \
  --corruptions fog,snow,frost,brightness,contrast \
  --severities 1,2,3,4,5 \
  --out-csv eval_cifar100c_corruptions.csv
```

```
python tools/run_cifar100c_feature_shift.py \
  --config configs/default.json \
  --method sbhfrl \
  --corruptions fog,snow,frost,brightness,contrast \
  --severities 1,2,3,4,5 \
  --out results_fedmps.txt
```

The console output shows round accuracy and which components are currently enabled (e.g., `FedProto`, `HD-IB+QualityFusion`, etc.).

### Configuration Guide

- **General**
  - `seed`: random seed for reproducibility; change when you want different non-IID splits.
  - `device`: `"auto"` picks CUDA if available; set `"cpu"` explicitly on machines without GPUs.
  - `data_root`: directory to cache CIFAR-10; use an absolute path if the repo is read-only.
  - `num_classes`: number of classes; keep `10` for CIFAR-10 unless you swap datasets.
  - `model`: choose `"base"`, `"hdib"`, or `"resnet10"` depending on the backbone you want to study.
  - `hdib_backbone`: when `model: "hdib"`, pick `"custom"` (original CNN) or `"resnet10"`; future ResNet variants can be added similarly.
- **Training Horizon**
  - `rounds`: total FL rounds; start with `10-30` for quick runs, increase for convergence studies.
  - `local_epochs`: client-side epochs per round; `1` keeps runtime low, but you can raise to improve on-device fitting.
  - `batch_size`: mini-batch size for local SGD; reduce if VRAM is limited.
  - `proto_batch_size`: batch size used when re-encoding data to compute prototypes; keep larger than `batch_size` for stable centroids.
- **Federated Topology**
  - `num_shards`: number of geographic shards/RSUs; set according to how many groups you want.
  - `clients_per_shard`: clients hosted under each shard; total clients = `num_shards * clients_per_shard`.
  - `clients_per_round`: how many clients per shard participate each round; lower values simulate asynchronous participation.
  - `alpha_dirichlet`: controls non-IID level; smaller values (e.g., `0.3`) make client distributions more skewed.
- **Optimization**
  - `lr`, `momentum`, `weight_decay`: standard SGD hyper-parameters; tune if accuracy stagnates.
  - `max_grad_norm`: gradient clipping threshold to stabilize HD-IB training; decrease when gradients explode.
  - `muon_betas`, `muon_eps`: Muon-specific hyper-parameters used when `model: "hdib"`; defaults mirror the paper’s settings.
- **Ledger/Blockchain Controls**
  - `init_reputation`: initial trust score for each shard; increase if you expect few adversaries.
  - `use_quality_fusion`: toggles PoQ aggregation; requires `alpha_quality`, `beta_quality`, `gamma_quality` weights (density/channel/reputation importance) to be set.
  - `use_reputation_consensus`: enables ledger-based anomaly auditing; relies on `wasserstein_threshold` to reject outliers (higher threshold = more tolerant).
  - `use_blockchain_memory`: activates blockchain EMA teacher; governed by `history` (number of rounds stored) and `ema_decay` (closer to `1.0` keeps older prototypes relevant).
  - `use_cluster_prototypes`: enables lightweight clustering over shard prototypes before they are published; pairs with `cluster_threshold` (cosine similarity cutoff, higher = stricter grouping).
  - `cluster_threshold`: cosine-similarity bar for the clustering stage; values around `0.8-0.9` keep clusters tight.
- **HD-IB Loss Terms (only when `model: "hdib"`):**
  - `lambda_ib`: scales KL regularization; start with `1.0`, reduce if embeddings collapse.
  - `lambda_contrast`: weight of supervised contrastive term; bump to emphasize inter-class separation.
  - `lambda_consistency`: enforces cross-layer alignment; lower it if shallow layers diverge due to noise.
  - `lambda_distill`: strength of EMA teacher guidance; keep small (≤0.2) to avoid over-regularization.

When crafting new experiments, begin with the baseline (`model: "base"`, all `use_*` flags false). Once accuracy improves above the random 10% mark, enable one idea at a time (HD-IB, clustering, PoQ, ledger consensus, blockchain memory, etc.) so you can attribute any gain or regression to a particular component.
