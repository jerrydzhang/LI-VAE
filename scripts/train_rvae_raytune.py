from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any

import torch
from ray import train, tune
from ray.train import Checkpoint, ScalingConfig
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.search.hyperopt import HyperOptSearch
from torch.utils.data import DataLoader, random_split

from livae.data import AdaptiveLatticeDataset, default_transform
from livae.loss import VAELoss
from livae.model import RVAE
from livae.train import MetricLogger, evaluate, train_one_epoch
from livae.utils import load_image_from_h5


def make_dataloaders(
    h5_paths: list[str],
    patch_size: int = 128,
    padding: int = 32,
    batch_size: int = 512,
    num_workers: int = 2,
    prefetch_factor: int = 1,
    val_split: float = 0.1,
    dataset_name: str | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders from H5 files.

    Parameters
    ----------
    h5_paths : list[str]
        Paths to H5 data files
    patch_size : int
        Size of extracted patches
    padding : int
        Padding around patches for augmentation
    batch_size : int
        Batch size for training
    num_workers : int
        Number of dataloader workers
    prefetch_factor : int
        Number of batches to prefetch per worker
    val_split : float
        Validation split fraction
    dataset_name : str | None
        Dataset path inside H5 file

    Returns
    -------
    train_loader : DataLoader
        Training dataloader
    val_loader : DataLoader
        Validation dataloader
    """
    images = [load_image_from_h5(p, dataset_name=dataset_name) for p in h5_paths]
    dataset = AdaptiveLatticeDataset(
        images, patch_size=patch_size, padding=padding, transform=default_transform
    )

    val_len = max(1, int(len(dataset) * val_split))
    train_len = max(1, len(dataset) - val_len)
    train_ds, val_ds = random_split(
        dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return train_loader, val_loader


def train_rvae_tune(config: dict[str, Any]) -> None:
    """Training function for Ray Tune trials.

    This function is called by Ray Tune for each hyperparameter configuration.
    It trains the model and reports metrics back to Tune for optimization.

    Parameters
    ----------
    config : dict[str, Any]
        Hyperparameter configuration from Ray Tune containing:
        - lr: Learning rate
        - latent_dim: Latent space dimensionality
        - beta: KL divergence weight
        - weight_decay: L2 regularization
        - batch_size: Batch size
        - Other training hyperparameters
    """
    # Get static configuration from Ray Tune's session
    static_config = train.get_context().get_trial_resources().required_resources

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data with hyperparameter-dependent batch size
    train_loader, val_loader = make_dataloaders(
        h5_paths=config["h5_paths"],
        patch_size=config["patch_size"],
        padding=config["padding"],
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        prefetch_factor=config["prefetch_factor"],
        val_split=config["val_split"],
        dataset_name=config.get("dataset_name"),
    )

    # Initialize model with hyperparameters
    model = RVAE(
        latent_dim=config["latent_dim"],
        in_channels=1,
        patch_size=config["patch_size"],
    ).to(device)

    # Optimizer with tunable learning rate and weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"], eta_min=config["lr"] * 0.01
    )

    # Loss function with tunable beta
    criterion = VAELoss(
        beta=0.0 if config.get("beta_annealing", False) else config["beta"],
    )

    # Mixed precision training
    scaler = (
        torch.amp.GradScaler()
        if device.type == "cuda" and not config.get("no_amp", False)
        else None
    )

    train_logger = MetricLogger()
    val_logger = MetricLogger()

    # Training loop
    for epoch in range(1, config["epochs"] + 1):
        # Beta annealing
        if config.get("beta_annealing", False):
            if epoch <= config.get("beta_annealing_epochs", 10):
                current_beta = config["beta"] * (
                    epoch / config.get("beta_annealing_epochs", 10)
                )
            else:
                current_beta = config["beta"]
            criterion.beta = current_beta

        # Train one epoch
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            train_logger,
            device,
            scaler=scaler,
            grad_max_norm=config.get("grad_max_norm"),
        )

        # Evaluate
        evaluate(model, val_loader, criterion, val_logger, device)

        # Get metrics
        train_metrics = train_logger.get_averages()
        val_metrics = val_logger.get_averages()

        # Step scheduler
        scheduler.step()

        # Report metrics to Ray Tune
        metrics = {
            "loss": val_metrics.get("val_loss", float("inf")),
            "train_loss": train_metrics.get("train_loss", 0.0),
            "val_loss": val_metrics.get("val_loss", 0.0),
            "train_psnr": train_metrics.get("train_psnr", 0.0),
            "val_psnr": val_metrics.get("val_psnr", 0.0),
            "train_recon_loss": train_metrics.get("train_recon_loss", 0.0),
            "val_recon_loss": val_metrics.get("val_recon_loss", 0.0),
            "train_kl_loss": train_metrics.get("train_kl_loss", 0.0),
            "val_kl_loss": val_metrics.get("val_kl_loss", 0.0),
            "epoch": epoch,
        }

        # Create checkpoint for this trial
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        # Report to Ray Tune with checkpoint
        train.report(metrics, checkpoint=checkpoint)

        # Reset loggers
        train_logger.reset()
        val_logger.reset()


def run_hyperparameter_search(args: argparse.Namespace) -> None:
    """Run Ray Tune hyperparameter search.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments containing search space and configuration
    """
    # Prepare data paths
    h5_paths = args.data or glob.glob(str(Path("data") / "*.h5"))
    if not h5_paths:
        raise FileNotFoundError(
            "No H5 data files found. Provide --data paths or place H5 files in ./data"
        )

    print(f"Data files: {h5_paths}")
    print(f"Starting Ray Tune hyperparameter search...")
    print(f"Number of trials: {args.num_samples}")
    print(f"Max concurrent trials: {args.max_concurrent}")

    # Define search space
    config = {
        # Hyperparameters to tune
        "lr": tune.loguniform(args.lr_min, args.lr_max),
        "latent_dim": tune.choice(args.latent_dims),
        "beta": tune.loguniform(args.beta_min, args.beta_max),
        "weight_decay": tune.loguniform(args.weight_decay_min, args.weight_decay_max),
        "batch_size": tune.choice(args.batch_sizes),
        # Fixed parameters
        "h5_paths": h5_paths,
        "patch_size": args.patch_size,
        "padding": args.padding,
        "num_workers": args.num_workers,
        "prefetch_factor": args.prefetch_factor,
        "val_split": args.val_split,
        "dataset_name": args.dataset_name,
        "epochs": args.epochs,
        "beta_annealing": args.beta_annealing,
        "beta_annealing_epochs": args.beta_annealing_epochs,
        "grad_max_norm": args.grad_max_norm,
        "no_amp": args.no_amp,
    }

    # Choose scheduler
    if args.scheduler == "asha":
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=args.epochs,
            grace_period=args.grace_period,
            reduction_factor=args.reduction_factor,
        )
    elif args.scheduler == "pbt":
        scheduler = PopulationBasedTraining(
            time_attr="epoch",
            metric="loss",
            mode="min",
            perturbation_interval=args.perturbation_interval,
            hyperparam_mutations={
                "lr": tune.loguniform(args.lr_min, args.lr_max),
                "beta": tune.loguniform(args.beta_min, args.beta_max),
            },
        )
    else:
        scheduler = None

    # Choose search algorithm
    if args.search_alg == "hyperopt":
        search_alg = HyperOptSearch(metric="loss", mode="min")
    else:
        search_alg = None

    # Configure Ray Tune
    tuner = tune.Tuner(
        tune.with_resources(
            train_rvae_tune,
            resources={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
        ),
        param_space=config,
        tune_config=tune.TuneConfig(
            scheduler=scheduler,
            search_alg=search_alg,
            num_samples=args.num_samples,
            max_concurrent_trials=args.max_concurrent,
        ),
        run_config=train.RunConfig(
            name=args.experiment_name,
            storage_path=args.ray_results_dir,
            stop={"epoch": args.epochs},
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="loss",
                checkpoint_score_order="min",
            ),
        ),
    )

    # Run the search
    results = tuner.fit()

    # Get best trial
    best_result = results.get_best_result(metric="loss", mode="min")

    print("\n" + "=" * 80)
    print("HYPERPARAMETER SEARCH COMPLETE")
    print("=" * 80)
    print(f"\nBest trial config:")
    for key, value in best_result.config.items():
        if key not in ["h5_paths", "dataset_name"]:  # Skip long/non-essential params
            print(f"  {key}: {value}")

    print(f"\nBest trial metrics:")
    print(f"  val_loss: {best_result.metrics['val_loss']:.4f}")
    print(f"  val_psnr: {best_result.metrics.get('val_psnr', 0):.2f}")
    print(f"  train_loss: {best_result.metrics['train_loss']:.4f}")

    print(f"\nBest checkpoint: {best_result.checkpoint}")

    # Save best config to file
    if args.save_best_config:
        config_path = Path(args.save_best_config)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        import json

        with open(config_path, "w") as f:
            # Filter out non-serializable items
            save_config = {
                k: v
                for k, v in best_result.config.items()
                if k not in ["h5_paths"] and not callable(v)
            }
            json.dump(save_config, f, indent=2)
        print(f"\nBest config saved to: {config_path}")


def build_argparser() -> argparse.ArgumentParser:
    """Build argument parser for Ray Tune hyperparameter search."""
    parser = argparse.ArgumentParser(
        description="Ray Tune hyperparameter search for RVAE training"
    )

    # Data arguments
    parser.add_argument(
        "--data", nargs="*", help="Paths to H5 files (default: data/*.h5)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Dataset path inside H5 file (auto-detected if omitted)",
    )
    parser.add_argument(
        "--patch-size", type=int, default=128, help="Size of extracted patches"
    )
    parser.add_argument(
        "--padding", type=int, default=32, help="Padding around patches"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split fraction"
    )

    # Hyperparameter search space
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-5,
        help="Minimum learning rate for search",
    )
    parser.add_argument(
        "--lr-max",
        type=float,
        default=2e-3,
        help="Maximum learning rate for search",
    )
    parser.add_argument(
        "--latent-dims",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="Latent dimensions to try",
    )
    parser.add_argument(
        "--beta-min",
        type=float,
        default=0.1,
        help="Minimum beta (KL weight) for search",
    )
    parser.add_argument(
        "--beta-max",
        type=float,
        default=2.0,
        help="Maximum beta (KL weight) for search",
    )
    parser.add_argument(
        "--weight-decay-min",
        type=float,
        default=1e-6,
        help="Minimum weight decay for search",
    )
    parser.add_argument(
        "--weight-decay-max",
        type=float,
        default=1e-3,
        help="Maximum weight decay for search",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[512],
        help="Batch sizes to try",
    )

    # Training configuration
    parser.add_argument(
        "--epochs",
        type=int,
        default=310,
        help="Max epochs per trial (consider reducing for faster search)",
    )
    parser.add_argument(
        "--beta-annealing",
        action="store_true",
        help="Enable beta annealing",
    )
    parser.add_argument(
        "--beta-annealing-epochs",
        type=int,
        default=10,
        help="Epochs for beta warmup",
    )
    parser.add_argument(
        "--grad-max-norm",
        type=float,
        default=None,
        help="Max gradient norm for clipping",
    )

    # Ray Tune configuration
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of hyperparameter configurations to try",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=4,
        help="Maximum number of concurrent trials",
    )
    parser.add_argument(
        "--cpus-per-trial",
        type=float,
        default=2,
        help="CPUs allocated per trial",
    )
    parser.add_argument(
        "--gpus-per-trial",
        type=float,
        default=0.25,
        help="GPUs allocated per trial (fractional allowed for sharing)",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        choices=["asha", "pbt", "none"],
        default="asha",
        help="Trial scheduler (ASHA=early stopping, PBT=population-based)",
    )
    parser.add_argument(
        "--search-alg",
        type=str,
        choices=["hyperopt", "none"],
        default="hyperopt",
        help="Search algorithm (HyperOpt=Bayesian optimization)",
    )

    # ASHA scheduler parameters
    parser.add_argument(
        "--grace-period",
        type=int,
        default=30,
        help="Minimum epochs before early stopping (ASHA)",
    )
    parser.add_argument(
        "--reduction-factor",
        type=int,
        default=3,
        help="Fraction of trials to keep at each rung (ASHA)",
    )

    # PBT scheduler parameters
    parser.add_argument(
        "--perturbation-interval",
        type=int,
        default=10,
        help="Epochs between perturbations (PBT)",
    )

    # Dataloader configuration
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Dataloader workers per trial",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="Batches to prefetch per worker",
    )

    # Output configuration
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="rvae_tune",
        help="Name for this experiment",
    )
    parser.add_argument(
        "--ray-results-dir",
        type=str,
        default="~/ray_results",
        help="Directory for Ray Tune results",
    )
    parser.add_argument(
        "--save-best-config",
        type=str,
        default="checkpoints/best_config.json",
        help="Path to save best hyperparameter configuration",
    )

    # Other options
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision",
    )

    return parser


if __name__ == "__main__":
    print("Starting RVAE Ray Tune hyperparameter search...")
    args = build_argparser().parse_args()
    run_hyperparameter_search(args)
