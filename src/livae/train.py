from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from skimage.feature import peak_local_max
from scipy.spatial.distance import cdist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

__all__ = [
    "train_one_epoch",
    "evaluate",
    "train_rvae_one_epoch",
    "evaluate_rvae",
    "evaluate_rotation_invariance",
    "log_reconstructions_tensorboard",
    "compute_atom_position_accuracy",
    "log_scalar_metrics_tensorboard",
    "MetricLogger",
]


def angles_from_theta(theta: torch.Tensor) -> torch.Tensor:
    """Convert model theta output to angles in radians.

    Supports either raw angles [B] or 2D unit vectors [B,2] = (cos, sin).
    """
    if theta is None:
        return torch.empty(0, device="cpu")
    if theta.dim() == 2 and theta.size(-1) == 2:
        return torch.atan2(theta[:, 1], theta[:, 0])
    return theta


def circular_std(angles: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Circular standard deviation (radians) from angles tensor [B].

    Uses std = sqrt(-2 ln R) with R the mean resultant length.
    """
    if angles.numel() == 0:
        return torch.tensor(0.0, device=angles.device)
    s = torch.sin(angles).mean()
    c = torch.cos(angles).mean()
    R = torch.sqrt(s * s + c * c)
    R = torch.clamp(R, min=eps)
    return torch.sqrt(torch.clamp(-2.0 * torch.log(R), max=1e6))


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    metric_logger: MetricLogger,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    grad_max_norm: float | None = None,
    agc_lambda: float | None = None,
) -> None:
    """Generic training loop that works with both VAE and rVAE models.

    Automatically detects model type based on forward() output signature.
    VAE returns: (recon, mu, logvar)
    rVAE returns: (rotated_recon, recon, theta, mu, logvar)
    """
    model.train()
    total_loss = 0.0
    recon_loss_sum = 0.0
    kld_loss_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    canonical_psnr_sum = 0.0
    canonical_ssim_sum = 0.0
    rotation_std_sum = 0.0
    rotation_std_linear_sum = 0.0
    latent_mean_abs_sum = 0.0
    latent_std_sum = 0.0
    grad_norm_sum = 0.0
    grad_norm_pre_sum = 0.0
    grad_norm_post_sum = 0.0
    canonical_batches = 0
    n_batches = 0

    use_amp = scaler is not None

    for x in tqdm(data_loader, desc="Training", leave=False):
        if isinstance(x, (list, tuple)):
            x = x[0]

        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast("cuda", enabled=use_amp):
            if len(outputs) == 3:
                # VAE: (recon, mu, logvar)
                recon, mu, logvar = outputs
                loss, batch_recon_loss, batch_kld_loss = criterion(
                    recon, x, mu, logvar
                )
            elif len(outputs) == 5:
                # rVAE: (rotated_recon, recon, theta, mu, logvar)
                _, canonical_recon, theta, mu, logvar = outputs
                # New objective: only reconstruct the canonical representation
                canonical_input = rotate_to_canonical(
                    x, theta, model.encoder.rotation_stn
                )
                loss, batch_recon_loss, batch_kld_loss = criterion(
                    canonical_recon, canonical_input, mu, logvar
                )
            else:
                raise ValueError(f"Unexpected model output length: {len(outputs)}")

        if use_amp:
            scaler.scale(loss).backward()
            # Unscale for accurate norm/clip
            scaler.unscale_(optimizer)
        else:
            loss.backward()

        # Compute pre-clip norm (after unscale if AMP)
        pre_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.detach()
                pre_norm_sq += float(g.norm(2).item() ** 2)
        pre_norm = pre_norm_sq ** 0.5

        # Optional gradient clipping
        # Apply adaptive gradient clipping (AGC) or standard clip
        if agc_lambda is not None and agc_lambda > 0:
            apply_agc(model.parameters(), agc_lambda)
        elif grad_max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_max_norm)

        # Compute post-clip norm
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm_sq += param_norm.item() ** 2
        total_norm = total_norm_sq ** 0.5

        grad_norm_pre_sum += pre_norm
        grad_norm_post_sum += total_norm
        grad_norm_sum += total_norm

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        total_loss += loss.item()
        recon_loss_sum += batch_recon_loss.item()
        kld_loss_sum += batch_kld_loss.item()

        with torch.no_grad():
            psnr_sum += compute_psnr(rotated_recon, x)
            ssim_sum += compute_ssim(rotated_recon, x)
            latent_mean_abs_sum += torch.mean(torch.abs(mu)).item()
            latent_std_sum += torch.mean(torch.exp(0.5 * logvar)).item()

            if theta is not None:
                angles = angles_from_theta(theta)
                rotation_std_sum += circular_std(angles).item()
                rotation_std_linear_sum += torch.std(angles).item()

            if (
                canonical_recon is not None
                and theta is not None
                and hasattr(model, "encoder")
                and hasattr(model.encoder, "rotation_stn")
            ):
                canonical_input = rotate_to_canonical(
                    x, theta, model.encoder.rotation_stn
                )
                canonical_psnr_sum += compute_psnr(canonical_recon, canonical_input)
                canonical_ssim_sum += compute_ssim(canonical_recon, canonical_input)
                canonical_batches += 1

        n_batches += 1

    avg_recon_loss = recon_loss_sum / n_batches
    avg_kld_loss = kld_loss_sum / n_batches
    avg_total_loss = total_loss / n_batches
    # Ratio of recon to total loss; helps diagnose KL collapse
    recon_weight = avg_recon_loss / (avg_total_loss + 1e-8)

    metrics = {
        "train_loss": avg_total_loss,
        "train_recon_loss": avg_recon_loss,
        "train_kld_loss": avg_kld_loss,
        "train_recon_weight": recon_weight,
        "train_psnr": psnr_sum / n_batches,
        "train_ssim": ssim_sum / n_batches,
        "train_rotation_std": rotation_std_sum / n_batches,
        "train_rotation_std_linear": rotation_std_linear_sum / n_batches,
        "train_latent_mean_abs": latent_mean_abs_sum / n_batches,
        "train_latent_std": latent_std_sum / n_batches,
        "train_grad_norm": grad_norm_sum / n_batches,
        "train_grad_norm_pre": grad_norm_pre_sum / n_batches,
        "train_grad_norm_post": grad_norm_post_sum / n_batches,
    }

    if canonical_batches > 0:
        metrics["train_canonical_psnr"] = canonical_psnr_sum / canonical_batches
        metrics["train_canonical_ssim"] = canonical_ssim_sum / canonical_batches

    metric_logger.update(**metrics)


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    metric_logger: MetricLogger,
    device: torch.device,
) -> None:
    """Generic evaluation loop that works with both VAE and rVAE models.

    Automatically detects model type based on forward() output signature.
    VAE returns: (recon, mu, logvar)
    rVAE returns: (rotated_recon, recon, theta, mu, logvar)
    """
    model.eval()
    total_loss = 0.0
    recon_loss_sum = 0.0
    kld_loss_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    canonical_psnr_sum = 0.0
    canonical_ssim_sum = 0.0
    rotation_std_sum = 0.0
    rotation_std_linear_sum = 0.0
    latent_mean_abs_sum = 0.0
    latent_std_sum = 0.0
    canonical_batches = 0
    n_batches = 0

    with torch.no_grad():
        for x in data_loader:
            if isinstance(x, (list, tuple)):
                x = x[0]

            x = x.to(device)

            # Forward pass - handle both VAE and rVAE
            outputs = model(x)
            if len(outputs) == 3:
                # VAE: (recon, mu, logvar)
                recon, mu, logvar = outputs
                loss, batch_recon_loss, batch_kld_loss = criterion(
                    recon, x, mu, logvar
                )
            elif len(outputs) == 5:
                # rVAE: (rotated_recon, recon, theta, mu, logvar)
                _, canonical_recon, theta, mu, logvar = outputs
                # New objective: only reconstruct the canonical representation
                canonical_input = rotate_to_canonical(
                    x, theta, model.encoder.rotation_stn
                )
                loss, batch_recon_loss, batch_kld_loss = criterion(
                    canonical_recon, canonical_input, mu, logvar
                )
            else:
                raise ValueError(f"Unexpected model output length: {len(outputs)}")

            total_loss += loss.item()
            recon_loss_sum += batch_recon_loss.item()
            kld_loss_sum += batch_kld_loss.item()

            psnr_sum += compute_psnr(rotated_recon, x)
            ssim_sum += compute_ssim(rotated_recon, x)
            latent_mean_abs_sum += torch.mean(torch.abs(mu)).item()
            latent_std_sum += torch.mean(torch.exp(0.5 * logvar)).item()

            if theta is not None:
                angles = angles_from_theta(theta)
                rotation_std_sum += circular_std(angles).item()
                rotation_std_linear_sum += torch.std(angles).item()

            if (
                canonical_recon is not None
                and theta is not None
                and hasattr(model, "encoder")
                and hasattr(model.encoder, "rotation_stn")
            ):
                canonical_input = rotate_to_canonical(
                    x, theta, model.encoder.rotation_stn
                )
                canonical_psnr_sum += compute_psnr(canonical_recon, canonical_input)
                canonical_ssim_sum += compute_ssim(canonical_recon, canonical_input)
                canonical_batches += 1

            n_batches += 1

    avg_recon_loss = recon_loss_sum / n_batches
    avg_kld_loss = kld_loss_sum / n_batches
    avg_total_loss = total_loss / n_batches
    recon_weight = avg_recon_loss / (avg_total_loss + 1e-8)

    metrics = {
        "val_loss": avg_total_loss,
        "val_recon_loss": avg_recon_loss,
        "val_kld_loss": avg_kld_loss,
        "val_recon_weight": recon_weight,
        "val_psnr": psnr_sum / n_batches,
        "val_ssim": ssim_sum / n_batches,
        "val_rotation_std": rotation_std_sum / n_batches,
        "val_rotation_std_linear": rotation_std_linear_sum / n_batches,
        "val_latent_mean_abs": latent_mean_abs_sum / n_batches,
        "val_latent_std": latent_std_sum / n_batches,
    }

    if canonical_batches > 0:
        metrics["val_canonical_psnr"] = canonical_psnr_sum / canonical_batches
        metrics["val_canonical_ssim"] = canonical_ssim_sum / canonical_batches

    metric_logger.update(**metrics)


# =============================================================================


# rVAE-specific training functions have been removed as the generic


# train_one_epoch and evaluate functions now handle both VAE and rVAE cases.


# =============================================================================


class MetricLogger:
    def __init__(self):
        self.metrics = defaultdict(list)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.metrics[k].append(v)

    def get_averages(self) -> dict[str, Any]:
        return {k: np.mean(v) for k, v in self.metrics.items()}

    def reset(self):
        self.metrics.clear()


def get_rotation_stats(rotations: torch.Tensor) -> tuple[float, float]:
    angles = torch.atan2(rotations[:, 1], rotations[:, 0]) * (180.0 / np.pi)
    mean_angle = torch.mean(angles).item()
    std_angle = torch.std(angles).item()
    return mean_angle, std_angle


def compute_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Parameters
    ----------
    img1 : torch.Tensor
        First image tensor
    img2 : torch.Tensor
        Second image tensor
    max_val : float
        Maximum possible pixel value (default: 1.0 for normalized images)

    Returns
    -------
    float
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse)).item()


def compute_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int = 11,
    C1: float = 0.01**2,
    C2: float = 0.03**2,
) -> float:
    """Compute Structural Similarity Index between two images.

    Simplified SSIM implementation for grayscale images.

    Parameters
    ----------
    img1 : torch.Tensor
        First image tensor [B, C, H, W]
    img2 : torch.Tensor
        Second image tensor [B, C, H, W]
    window_size : int
        Size of the averaging window (default: 11)
    C1, C2 : float
        Stability constants

    Returns
    -------
    float
        SSIM value between -1 and 1 (1 is perfect similarity)
    """
    # Simplified SSIM using average pooling
    mu1 = torch.nn.functional.avg_pool2d(
        img1, window_size, stride=1, padding=window_size // 2
    )
    mu2 = torch.nn.functional.avg_pool2d(
        img2, window_size, stride=1, padding=window_size // 2
    )

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        torch.nn.functional.avg_pool2d(
            img1 * img1, window_size, stride=1, padding=window_size // 2
        )
        - mu1_sq
    )
    sigma2_sq = (
        torch.nn.functional.avg_pool2d(
            img2 * img2, window_size, stride=1, padding=window_size // 2
        )
        - mu2_sq
    )
    sigma12 = (
        torch.nn.functional.avg_pool2d(
            img1 * img2, window_size, stride=1, padding=window_size // 2
        )
        - mu1_mu2
    )

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return ssim_map.mean().item()


def rotate_to_canonical(
    x: torch.Tensor, theta: torch.Tensor, rotation_stn: nn.Module
) -> torch.Tensor:
    """Rotate a batch of images to the canonical frame using predicted angles."""

    rot_matrix = rotation_stn.get_rotation_matrix(theta).to(x.dtype)
    grid = F.affine_grid(rot_matrix, x.size(), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="reflection", align_corners=False)


def evaluate_rotation_invariance(
    model: nn.Module,
    images: torch.Tensor,
    angles: Iterable[float] = (0, 45, 90, 135, 180, 225, 270, 315),
    device: torch.device = torch.device("cpu"),
    max_batches: int | None = None,
) -> dict[str, float]:
    """Evaluate rotation invariance on a small set of images.

    For each input image, rotate by several angles, run the model,
    un-rotate the rotated reconstruction, and measure:
      - latent variance across rotations (lower is better)
      - reconstruction RMSE to original (lower is better)
      - PSNR/SSIM to original (higher is better)
      - angle prediction error if theta is returned (lower is better)

    Parameters
    ----------
    model : nn.Module
        rVAE model returning (rotated_recon, recon, theta, mu, logvar)
    images : torch.Tensor
        Batch of images [B, C, H, W], preferably a small fixed set
    angles : Iterable[float]
        Angles (degrees) to test
    device : torch.device
        Device for evaluation
    max_batches : int | None
        Optional limit on number of images evaluated for speed

    Returns
    -------
    dict[str, float]
        Aggregated metrics across images
    """
    model.eval()
    results_latent_var = []
    results_rmse = []
    results_psnr = []
    results_ssim = []
    results_angle_error = []

    angles = list(angles)
    if len(images.shape) != 4:
        raise ValueError("images must have shape [B, C, H, W]")

    with torch.no_grad():
        for idx, img in enumerate(images):
            if max_batches is not None and idx >= max_batches:
                break

            latents = []
            recons = []
            angle_errors = []

            for angle in angles:
                rotated = TF.rotate(
                    img, angle, interpolation=InterpolationMode.BILINEAR
                )
                rotated = rotated.unsqueeze(0).to(device)

                rotated_recon, _, theta, mu, _ = model(rotated)

                # Un-rotate to original orientation for comparison
                unrotated_recon = TF.rotate(
                    rotated_recon[0].detach().cpu(),
                    -angle,
                    interpolation=InterpolationMode.BILINEAR,
                )

                latents.append(mu.detach().cpu())
                recons.append(unrotated_recon)

                if theta is not None:
                    pred_angle = torch.atan2(theta[0, 1], theta[0, 0]) * (180.0 / np.pi)
                    err = float(
                        min(
                            abs(pred_angle.item() - angle),
                            360 - abs(pred_angle.item() - angle),
                        )
                    )
                    angle_errors.append(err)

            latents = torch.stack(latents)  # [A, latent_dim]
            recons = torch.stack(recons)  # [A, C, H, W]

            latent_var = torch.var(latents, dim=0).mean().item()
            results_latent_var.append(latent_var)

            # Compare each unrotated recon to original
            for r in recons:
                rmse = torch.sqrt(torch.mean((r - img.cpu()) ** 2)).item()
                results_rmse.append(rmse)
                results_psnr.append(compute_psnr(r.unsqueeze(0), img.unsqueeze(0)))
                results_ssim.append(compute_ssim(r.unsqueeze(0), img.unsqueeze(0)))

            if angle_errors:
                results_angle_error.append(np.mean(angle_errors))

    return {
        "rotation_latent_variance": float(np.mean(results_latent_var))
        if results_latent_var
        else 0.0,
        "rotation_recon_rmse": float(np.mean(results_rmse)) if results_rmse else 0.0,
        "rotation_recon_psnr": float(np.mean(results_psnr)) if results_psnr else 0.0,
        "rotation_recon_ssim": float(np.mean(results_ssim)) if results_ssim else 0.0,
        "rotation_angle_error": float(np.mean(results_angle_error))
        if results_angle_error
        else 0.0,
    }


def log_reconstructions_tensorboard(
    model: nn.Module,
    images: torch.Tensor,
    writer: SummaryWriter,
    global_step: int,
    device: torch.device,
    tag: str = "recon",
    normalize: bool = True,
    nrow: int | None = None,
) -> None:
    """Log original, reconstructed, and error maps to TensorBoard.

    Layout per sample: [original | rotated_recon | abs_diff]. For rVAE, also logs
    canonical frame comparison if the rotation STN is available.
    """
    model.eval()
    with torch.no_grad():
        images_device = images.to(device)
        outputs = model(images_device)
        if len(outputs) == 3:
            # VAE: (recon, mu, logvar)
            rotated_recon, _, _ = outputs
            canonical_recon = None
            theta = None
        elif len(outputs) == 5:
            # rVAE: (rotated_recon, recon, theta, mu, logvar)
            rotated_recon, canonical_recon, theta, _, _ = outputs
        else:
            raise ValueError(f"Unexpected model output length: {len(outputs)}")

        rotated_recon_cpu = rotated_recon.detach().cpu()
        images_cpu = images.detach().cpu()
        diff = torch.abs(images_cpu - rotated_recon_cpu)

        # Prepare grid
        triplets = torch.cat([images_cpu, rotated_recon_cpu, diff], dim=0)
        grid = make_grid(triplets, nrow=nrow or images_cpu.size(0), normalize=normalize)
        writer.add_image(f"{tag}/original_recon_diff", grid, global_step)

        if (
            canonical_recon is not None
            and theta is not None
            and hasattr(model, "encoder")
            and hasattr(model.encoder, "rotation_stn")
        ):
            canonical_input = rotate_to_canonical(
                images_device, theta, model.encoder.rotation_stn
            )
            canonical_input_cpu = canonical_input.detach().cpu()
            canonical_recon_cpu = canonical_recon.detach().cpu()
            canonical_diff = torch.abs(canonical_input_cpu - canonical_recon_cpu)

            canonical_triplets = torch.cat(
                [canonical_input_cpu, canonical_recon_cpu, canonical_diff], dim=0
            )
            canonical_grid = make_grid(
                canonical_triplets,
                nrow=nrow or images_cpu.size(0),
                normalize=normalize,
            )
            writer.add_image(
                f"{tag}/canonical_original_recon_diff", canonical_grid, global_step
            )


def compute_atom_position_accuracy(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    lattice_spacing: float,
    threshold_ratio: float = 0.35,
) -> dict[str, float]:
    """Compare atom peak positions between original and reconstructed images.

    Parameters
    ----------
    original : torch.Tensor
        Image tensor [1, H, W] or [C, H, W] with atoms
    reconstruction : torch.Tensor
        Reconstructed image tensor matching shape
    lattice_spacing : float
        Estimated lattice spacing (pixels)
    threshold_ratio : float
        Fraction of lattice spacing to consider a correct match
    """
    if original.dim() == 3 and original.size(0) == 1:
        original_np = original[0].detach().cpu().numpy()
    elif original.dim() == 3:
        original_np = original.mean(dim=0).detach().cpu().numpy()
    else:
        original_np = original.detach().cpu().numpy()

    if reconstruction.dim() == 3 and reconstruction.size(0) == 1:
        recon_np = reconstruction[0].detach().cpu().numpy()
    elif reconstruction.dim() == 3:
        recon_np = reconstruction.mean(dim=0).detach().cpu().numpy()
    else:
        recon_np = reconstruction.detach().cpu().numpy()

    if lattice_spacing <= 0:
        raise ValueError("lattice_spacing must be positive")

    min_distance = max(int(lattice_spacing * threshold_ratio), 1)
    orig_peaks = peak_local_max(original_np, min_distance=min_distance)
    recon_peaks = peak_local_max(recon_np, min_distance=min_distance)

    if orig_peaks.size == 0:
        return {
            "atom_detection_rate": 0.0,
            "atom_position_accuracy": 0.0,
            "atom_mean_position_error": float("inf"),
            "n_original_atoms": 0,
            "n_reconstructed_atoms": recon_peaks.shape[0],
        }

    if recon_peaks.size == 0:
        return {
            "atom_detection_rate": 0.0,
            "atom_position_accuracy": 0.0,
            "atom_mean_position_error": float("inf"),
            "n_original_atoms": orig_peaks.shape[0],
            "n_reconstructed_atoms": 0,
        }

    distances = cdist(orig_peaks, recon_peaks)
    min_distances = distances.min(axis=1)

    correct = (min_distances < lattice_spacing * threshold_ratio).sum()

    return {
        "atom_detection_rate": float(recon_peaks.shape[0] / orig_peaks.shape[0]),
        "atom_position_accuracy": float(correct / orig_peaks.shape[0]),
        "atom_mean_position_error": float(min_distances.mean()),
        "n_original_atoms": int(orig_peaks.shape[0]),
        "n_reconstructed_atoms": int(recon_peaks.shape[0]),
    }


def log_scalar_metrics_tensorboard(
    writer: SummaryWriter,
    metrics: dict[str, float],
    global_step: int,
    prefix: str = "",
) -> None:
    """Log a dictionary of scalar metrics to TensorBoard."""
    for k, v in metrics.items():
        writer.add_scalar(f"{prefix}{k}", v, global_step)
