import argparse

from sbhfrl.trainer import run_federated_training
from sbhfrl.utils import get_device, load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description="SB-HFRL Simulator")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config JSON file.")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    run_federated_training(config, device)


if __name__ == "__main__":
    main()
