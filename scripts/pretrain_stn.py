from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from livae.data import PairedAdaptiveLatticeDataset
from livae.loss import cycle_consistency_loss
from livae.model import RVAE
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
    """Create training and validation dataloaders for STN pretraining."""
    images = [load_image_from_h5(p, dataset_name=dataset_name) for p in h5_paths]  # type: ignore
    dataset = PairedAdaptiveLatticeDataset(images, patch_size=patch_size, padding=padding)

    val_len = max(1, int(len(dataset) * val_split))
    train_len = max(1, len(dataset) - val_len)
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

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


def run_pretrain(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    h5_paths = args.data or glob.glob(str(Path("data") / "*.h5"))
    if not h5_paths:
        raise FileNotFoundError("No H5 data files found. Provide --data or place files in ./data")

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

    # Initialize RVAE but only train the RotationSTN head
    model = RVAE(latent_dim=args.latent_dim, in_channels=1, patch_size=args.patch_size).to(device)

    # Optimizer only on rotation STN parameters
    stn_params = list(model.encoder.rotation_stn.parameters())
    optimizer = torch.optim.AdamW(stn_params, lr=args.lr, weight_decay=args.weight_decay)

    writer = SummaryWriter(log_dir=Path(args.log_dir) / "stn_pretrain")

    print(f"Pretraining STN for {args.epochs} epochs...")
    best_val = float("inf")
    for epoch in tqdm(range(1, args.epochs + 1), desc="STN Pretrain"):
        model.train()
        total_loss, total_cycle, total_rotstd = 0.0, 0.0, 0.0
        n_batches = 0
        for batch in train_loader:
            # Paired dataset returns (x, x_rotated, angle_rad)
            if isinstance(batch, (list, tuple)) and len(batch) == 3:
                x, x_rot, angle = batch
            else:
                # Fallback: skip if format unexpected
                continue
            x = x.to(device)
            x_rot = x_rot.to(device)
            angle = (angle if isinstance(angle, torch.Tensor) else torch.tensor(angle)).to(device).float()

            optimizer.zero_grad(set_to_none=True)
            # Predict angles for original and rotated inputs
            _, _, theta_orig = model.encoder(x)
            _, _, theta_rot = model.encoder(x_rot)

            # Consistency: theta_rot - theta_orig ≈ -angle
            cycle_loss = cycle_consistency_loss(theta_orig, theta_rot, angle)
            loss = cycle_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(stn_params, max_norm=5.0)
            optimizer.step()

            with torch.no_grad():
                total_loss += loss.item()
                total_cycle += cycle_loss.item()
                total_rotstd += torch.std(theta_orig).item()
                n_batches += 1

        # Validation (optional monitoring)
        model.eval()
        val_cycle, val_rotstd, val_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)) and len(batch) == 3:
                    x, x_rot, angle = batch
                else:
                    continue
                x = x.to(device)
                x_rot = x_rot.to(device)
                angle = (angle if isinstance(angle, torch.Tensor) else torch.tensor(angle)).to(device).float()
                _, _, theta_orig = model.encoder(x)
                _, _, theta_rot = model.encoder(x_rot)
                c_loss = cycle_consistency_loss(theta_orig, theta_rot, angle)
                val_cycle += c_loss.item()
                val_rotstd += torch.std(theta_orig).item()
                val_batches += 1

        train_loss = total_loss / max(1, n_batches)
        train_cycle = total_cycle / max(1, n_batches)
        train_rotstd = total_rotstd / max(1, n_batches)
        val_cycle_avg = val_cycle / max(1, val_batches)
        val_rotstd_avg = val_rotstd / max(1, val_batches)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/cycle_loss", train_cycle, epoch)
        writer.add_scalar("train/rotation_std", train_rotstd, epoch)
        writer.add_scalar("val/cycle_loss", val_cycle_avg, epoch)
        writer.add_scalar("val/rotation_std", val_rotstd_avg, epoch)

        # Track best on validation cycle loss
        if val_cycle_avg < best_val:
            best_val = val_cycle_avg
            if args.checkpoint:
                ckpt_path = Path(args.checkpoint)
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"rotation_stn": model.encoder.rotation_stn.state_dict()}, ckpt_path)
                print(f"  → Saved STN checkpoint (val_cycle: {best_val:.4f})")

    writer.close()
    print("STN pretraining complete.")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain RotationSTN with angle consistency")
    parser.add_argument("--data", nargs="*", help="Paths to H5 files (default: data/*.h5)")
    parser.add_argument("--dataset-name", type=str, default=None, help="Dataset path inside H5 file")
    parser.add_argument("--patch-size", type=int, default=128, help="Patch size")
    parser.add_argument("--padding", type=int, default=32, help="Padding around patches for augmentation")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size")
    parser.add_argument("--val-split", type=float, default=0.1, help="Validation split fraction")
    parser.add_argument("--num-workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Prefetch factor per worker")
    parser.add_argument("--epochs", type=int, default=30, help="Pretrain epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--latent-dim", type=int, default=16, help="Latent dim (unused for STN pretrain)")
    parser.add_argument("--log-dir", type=str, default="runs/stn", help="TensorBoard log dir")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/stn_pretrained.pt", help="Save path for STN")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run_pretrain(args)
