from __future__ import annotations
import time

import argparse
import glob
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from livae.data import AdaptiveLatticeDataset, default_transform
from livae.loss import VAELoss
from livae.model import RVAE
from livae.train import (
    MetricLogger,
    evaluate,
    log_reconstructions_tensorboard,
    log_scalar_metrics_tensorboard,
    train_one_epoch,
)
from livae.utils import load_image_from_h5


def make_dataloaders(
    h5_paths: Sequence[str],
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
    h5_paths : Sequence[str]
        Paths to H5 data files
    patch_size : int
        Size of extracted patches (default: 128)
    padding : int
        Padding around patches for augmentation (default: 32)
    batch_size : int
        Batch size for training (default: 512)
    num_workers : int
        Number of dataloader workers (default: 2)
    prefetch_factor : int
        Number of batches to prefetch per worker (default: 1)
    val_split : float
        Validation split fraction (default: 0.1)
    dataset_name : str | None
        Dataset path inside H5 file

    Returns
    -------
    train_loader : DataLoader
        Training dataloader
    val_loader : DataLoader
        Validation dataloader
    """
    print(f"Loading data from: {h5_paths}")
    images = [load_image_from_h5(p, dataset_name=dataset_name) for p in h5_paths]  # type: ignore
    dataset = AdaptiveLatticeDataset(
        images, patch_size=patch_size, padding=padding, transform=default_transform
    )

    val_len = max(1, int(len(dataset) * val_split))
    train_len = max(1, len(dataset) - val_len)
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    print(f"Dataset size: {len(dataset)} patches ({train_len} train, {val_len} val)")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    return train_loader, val_loader


def run_training(args: argparse.Namespace) -> None:
    """Main training loop for plain RVAE.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    print(f"Using device: {device}")

    h5_paths = args.data or glob.glob(str(Path("data") / "*.h5"))
    if not h5_paths:
        raise FileNotFoundError(
            "No H5 data files found. Provide --data paths or place H5 files in ./data"
        )

    train_loader, val_loader = make_dataloaders(
        h5_paths,
        patch_size=args.patch_size,
        padding=args.padding,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        val_split=args.val_split,
        dataset_name=args.dataset_name,
    )

    model = RVAE(
        latent_dim=args.latent_dim,
        in_channels=1,
        patch_size=args.patch_size,
    ).to(device)

    if hasattr(torch, "compile") and args.compile:
        print("Compiling model with torch.compile for faster execution...")
        model = torch.compile(model)

    print(f"Model initialized: RVAE with {args.latent_dim}-dim latent space")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    criterion = VAELoss(
        beta=0.0 if args.beta_annealing else args.beta,
    )

    scaler = (
        torch.amp.GradScaler() if device.type == "cuda" and not args.no_amp else None
    )
    if scaler:
        print("Using automatic mixed precision (AMP) for faster training")

    log_dir = Path(args.log_dir) / time.strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=log_dir)
    best_val = float("inf")

    train_logger = MetricLogger()
    val_logger = MetricLogger()

    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Learning rate: {args.lr}, Beta: {args.beta}")
    if args.beta_annealing:
        print(f"Beta annealing enabled: {args.beta_annealing_epochs} epochs warmup")

    for epoch in tqdm(range(1, args.epochs + 1), desc="Training Epochs"):
        if args.beta_annealing:
            if epoch <= args.beta_annealing_epochs:
                current_beta = args.beta * (epoch / args.beta_annealing_epochs)
            else:
                current_beta = args.beta
            criterion.beta = current_beta
        train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            train_logger,
            device,
            scaler=scaler,
            canonical_weight=args.canonical_weight,
            grad_max_norm=args.grad_max_norm,
        )

        evaluate(
            model,
            val_loader,
            criterion,
            val_logger,
            device,
            canonical_weight=args.canonical_weight,
        )

        log_scalar_metrics_tensorboard(
            writer, train_logger.get_averages(), global_step=epoch, prefix="train/"
        )
        log_scalar_metrics_tensorboard(
            writer, val_logger.get_averages(), global_step=epoch, prefix="val/"
        )

        if epoch % args.vis_every == 0:
            sample_batch = next(iter(val_loader))[: args.vis_samples].to(device)
            log_reconstructions_tensorboard(
                model,
                sample_batch,
                writer,
                global_step=epoch,
                device=device,
                tag="recon",
            )

        val_loss = val_logger.get_averages().get("val_loss", 0.0)
        if val_loss < best_val:
            best_val = val_loss
            if args.checkpoint:
                ckpt_path = Path(args.checkpoint)
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "epoch": epoch,
                        "best_val": best_val,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"  → Saved checkpoint (val_loss: {best_val:.4f})")

        # Step the learning rate scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log scheduler values
        writer.add_scalar("train/learning_rate", current_lr, epoch)
        if args.beta_annealing:
            writer.add_scalar("train/beta", criterion.beta, epoch)

        train_metrics = train_logger.get_averages()
        val_metrics = val_logger.get_averages()

        status_msg = (
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_metrics['train_loss']:.4f} "
            f"val_loss={val_loss:.4f} | "
            f"train_psnr={train_metrics.get('train_psnr', 0):.2f} "
            f"val_psnr={val_metrics.get('val_psnr', 0):.2f} | "
            f"lr={current_lr:.2e}"
        )
        if args.beta_annealing:
            status_msg += f" beta={criterion.beta:.3f}"
        print(status_msg)

        train_logger.reset()
        val_logger.reset()

    writer.close()
    print(f"\nTraining complete! Best validation loss: {best_val:.4f}")
    if args.checkpoint:
        print(f"Best model saved to: {args.checkpoint}")


def build_argparser() -> argparse.ArgumentParser:
    """Build argument parser for RVAE training script."""
    parser = argparse.ArgumentParser(
        description="Train standard RVAE on atom patches from STEM microscopy"
    )

    parser.add_argument(
        "--data", nargs="*", help="Paths to H5 files (default: data/*.h5)"
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help=(
            "Dataset path inside H5 file (e.g., 'Measurement_000/Channel_000/HAADF/HAADF'). "
            "If omitted, auto-detects a 2D image dataset."
        ),
    )
    parser.add_argument(
        "--patch-size", type=int, default=128, help="Size of extracted patches"
    )
    parser.add_argument(
        "--padding",
        type=int,
        default=32,
        help="Padding around patches for augmentation (32 pixels recommended for 128x128 patches to avoid rotation clipping)",
    )

    parser.add_argument(
        "--batch-size", type=int, default=512, help="Batch size for training"
    )
    parser.add_argument(
        "--val-split", type=float, default=0.1, help="Validation split fraction"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of dataloader workers (reduce if stalls; increase if CPU idle)",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Batches to prefetch per worker (increase to improve throughput if workers are not stalling)",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")

    parser.add_argument(
        "--latent-dim", type=int, default=16, help="Dimension of latent space"
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Beta coefficient for KL divergence (higher = stronger KL penalty, but risk of latent collapse with mean reduction)",
    )

    parser.add_argument(
        "--beta-annealing",
        action="store_true",
        help="Enable beta annealing (linear warmup from 0 to beta)",
    )
    parser.add_argument(
        "--beta-annealing-epochs",
        type=int,
        default=10,
        help="Number of epochs for beta warmup from 0 to beta (default: 10); increase to 20-30 for gentler warmup",
    )

    parser.add_argument(
        "--canonical-weight",
        type=float,
        default=0.2,
        help="Weight for canonical-frame consistency loss (0 to disable)",
    )
    parser.add_argument(
        "--grad-max-norm",
        type=float,
        default=None,
        help="Max gradient norm for clipping; set to a float to enable (None to disable)",
    )

    parser.add_argument(
        "--log-dir", type=str, default="runs/rvae", help="TensorBoard log directory"
    )
    parser.add_argument(
        "--vis-every",
        type=int,
        default=10,
        help="Visualize reconstructions every N epochs",
    )
    parser.add_argument(
        "--vis-samples", type=int, default=8, help="Number of samples to visualize"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/rvae_best.pt",
        help="Path to save best model checkpoint",
    )

    parser.add_argument(
        "--cpu", action="store_true", help="Force CPU even if CUDA is available"
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile optimization (off by default to avoid warmup stall)",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision training",
    )

    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_training(args)
