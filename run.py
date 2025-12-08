import argparse

from sbhfrl.trainer import run_federated_training
from sbhfrl.utils import get_device, load_config, set_seed


def main():
    parser = argparse.ArgumentParser(description="SB-HFRL Simulator")
    parser.add_argument("--config", type=str, default="configs/default.json", help="Path to config JSON file.")
    parser.add_argument("--save-ckpt", type=str, default=None, help="Optional path to save the best global checkpoint.")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.save_ckpt:
        config["save_checkpoint"] = args.save_ckpt
    set_seed(config.get("seed", 42))
    device = get_device(config.get("device", "auto"))
    run_federated_training(config, device)


if __name__ == "__main__":
    main()
