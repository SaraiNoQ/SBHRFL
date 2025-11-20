## SB-HFRL Simulation

This repository now separates the SB-HFRL simulation into modular components:

- `sbhfrl/models`: prototype backbones (`base`, `hdib`)
- `sbhfrl/federated`: clients, shard aggregation, consensus, blockchain memory
- `sbhfrl/data_utils.py`: CIFAR-10 loading, non-IID partitioning, prototype extraction
- `sbhfrl/trainer.py`: orchestrates local updates → shard fusion → global consensus
- `configs/default.json`: central place for hyper-parameters and feature toggles
- `run.py`: entry-point that reads a config and executes the experiment

### Usage

1. Install dependencies (`torch`, `torchvision` etc.) inside your Python environment.
2. Adjust `configs/default.json`:
   - Keep `model: "base"` with all `use_*` flags set to `false` to run plain FedProto.
   - Switch to `"hdib"` and toggle `use_quality_fusion`, `use_reputation_consensus`, `use_blockchain_memory` to incrementally enable SB-HFRL components.
3. Launch training:

```bash
python run.py --config configs/default.json
```

The console output shows round accuracy and which components are currently enabled (e.g., `FedProto`, `HD-IB+QualityFusion`, etc.).
