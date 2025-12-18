from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_rvae import build_argparser, run_training


def load_best_config(config_path: str) -> dict:
    """Load best hyperparameter configuration from Ray Tune.

    Parameters
    ----------
    config_path : str
        Path to best_config.json file

    Returns
    -------
    dict
        Best hyperparameter configuration
    """
    with open(config_path, "r") as f:
        return json.load(f)


def main() -> None:
    """Train with best Ray Tune configuration."""
    parser = argparse.ArgumentParser(
        description="Train RVAE with best Ray Tune hyperparameters",
        parents=[build_argparser()],
        add_help=False,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="checkpoints/best_config.json",
        help="Path to best config JSON from Ray Tune",
    )
    parser.add_argument(
        "--override-epochs",
        type=int,
        default=None,
        help="Override epochs from config (useful for longer final training)",
    )

    args = parser.parse_args()

    # Load best config
    if Path(args.config).exists():
        print(f"Loading best hyperparameters from: {args.config}")
        best_config = load_best_config(args.config)

        # Override with best hyperparameters
        args.lr = best_config.get("lr", args.lr)
        args.latent_dim = best_config.get("latent_dim", args.latent_dim)
        args.beta = best_config.get("beta", args.beta)
        args.weight_decay = best_config.get("weight_decay", args.weight_decay)
        args.batch_size = best_config.get("batch_size", args.batch_size)

        print("\nUsing best hyperparameters:")
        print(f"  Learning Rate:  {args.lr:.2e}")
        print(f"  Latent Dim:     {args.latent_dim}")
        print(f"  Beta:           {args.beta:.3f}")
        print(f"  Weight Decay:   {args.weight_decay:.2e}")
        print(f"  Batch Size:     {args.batch_size}")
    else:
        print(
            f"Warning: Config file not found at {args.config}, using command-line args"
        )

    # Override epochs if specified
    if args.override_epochs is not None:
        args.epochs = args.override_epochs
        print(f"\nTraining for {args.epochs} epochs (overridden)")

    print("\nStarting training with best configuration...\n")
    run_training(args)


if __name__ == "__main__":
    main()
