"""
Metrics for evaluating VAE and rVAE models on atomic lattice reconstruction.

This module provides specialized metrics for assessing reconstruction quality,
latent space properties, and atom detection accuracy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
from skimage.feature import peak_local_max


__all__ = [
    "compute_psnr",
    "compute_ssim",
    "compute_reconstruction_metrics",
    "compute_latent_metrics",
    "compute_atom_detection_metrics",
    "compute_all_metrics",
]


def compute_psnr(
    img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0
) -> float:
    """Compute Peak Signal-to-Noise Ratio between two images.

    Parameters
    ----------
    img1 : torch.Tensor
        First image tensor [B, C, H, W]
    img2 : torch.Tensor
        Second image tensor [B, C, H, W]
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


def compute_reconstruction_metrics(
    original: torch.Tensor, reconstruction: torch.Tensor
) -> dict[str, float]:
    """Compute comprehensive reconstruction quality metrics.

    Parameters
    ----------
    original : torch.Tensor
        Original images [B, C, H, W]
    reconstruction : torch.Tensor
        Reconstructed images [B, C, H, W]

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - mse: Mean squared error
        - rmse: Root mean squared error
        - mae: Mean absolute error
        - psnr: Peak signal-to-noise ratio (dB)
        - ssim: Structural similarity index
    """
    mse = torch.mean((original - reconstruction) ** 2).item()
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(original - reconstruction)).item()
    psnr = compute_psnr(original, reconstruction)
    ssim = compute_ssim(original, reconstruction)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "psnr": psnr,
        "ssim": ssim,
    }


def compute_latent_metrics(
    mu: torch.Tensor, logvar: torch.Tensor
) -> dict[str, float]:
    """Compute metrics about the latent space distribution.

    Parameters
    ----------
    mu : torch.Tensor
        Mean of latent distribution [B, latent_dim]
    logvar : torch.Tensor
        Log variance of latent distribution [B, latent_dim]

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - latent_mean_abs: Mean absolute value of latent means
        - latent_mean_std: Standard deviation of latent means
        - latent_std_mean: Mean of latent standard deviations
        - latent_std_std: Standard deviation of latent standard deviations
        - latent_kl_per_dim: Average KL divergence per dimension
    """
    std = torch.exp(0.5 * logvar)

    # Statistics about the mean
    latent_mean_abs = torch.mean(torch.abs(mu)).item()
    latent_mean_std = torch.std(mu).item()

    # Statistics about the standard deviation
    latent_std_mean = torch.mean(std).item()
    latent_std_std = torch.std(std).item()

    # KL divergence per dimension (to unit Gaussian)
    kl_per_dim = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()).item()

    return {
        "latent_mean_abs": latent_mean_abs,
        "latent_mean_std": latent_mean_std,
        "latent_std_mean": latent_std_mean,
        "latent_std_std": latent_std_std,
        "latent_kl_per_dim": kl_per_dim,
    }


def compute_atom_detection_metrics(
    original: torch.Tensor,
    reconstruction: torch.Tensor,
    lattice_spacing: float,
    threshold_ratio: float = 0.35,
) -> dict[str, float]:
    """Compare atom peak positions between original and reconstructed images.

    This metric evaluates how well the VAE preserves atomic positions,
    which is critical for materials science applications.

    Parameters
    ----------
    original : torch.Tensor
        Original image tensor [1, H, W] or [C, H, W]
    reconstruction : torch.Tensor
        Reconstructed image tensor matching shape
    lattice_spacing : float
        Estimated lattice spacing in pixels
    threshold_ratio : float
        Fraction of lattice spacing to consider a correct match (default: 0.35)

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - atom_detection_rate: Ratio of detected atoms to original atoms
        - atom_position_accuracy: Fraction of correctly positioned atoms
        - atom_mean_position_error: Mean distance error for detected atoms (pixels)
        - n_original_atoms: Number of atoms in original image
        - n_reconstructed_atoms: Number of atoms in reconstruction
    """
    # Convert to numpy arrays
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

    # Detect peaks
    min_distance = max(int(lattice_spacing * threshold_ratio), 1)
    orig_peaks = peak_local_max(original_np, min_distance=min_distance)
    recon_peaks = peak_local_max(recon_np, min_distance=min_distance)

    # Handle edge cases
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

    # Compute distances between original and reconstructed peaks
    distances = cdist(orig_peaks, recon_peaks)
    min_distances = distances.min(axis=1)

    # Count correctly positioned atoms
    threshold = lattice_spacing * threshold_ratio
    correct = (min_distances < threshold).sum()

    return {
        "atom_detection_rate": float(recon_peaks.shape[0] / orig_peaks.shape[0]),
        "atom_position_accuracy": float(correct / orig_peaks.shape[0]),
        "atom_mean_position_error": float(min_distances.mean()),
        "n_original_atoms": int(orig_peaks.shape[0]),
        "n_reconstructed_atoms": int(recon_peaks.shape[0]),
    }


def compute_all_metrics(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    lattice_spacing: float | None = None,
) -> dict[str, float]:
    """Compute all available metrics for a batch of images.

    This is a convenience function that computes reconstruction, latent space,
    and (optionally) atom detection metrics in one call.

    Parameters
    ----------
    model : nn.Module
        VAE or rVAE model
    images : torch.Tensor
        Input images [B, C, H, W]
    device : torch.device
        Device for computation
    lattice_spacing : float | None
        Lattice spacing for atom detection metrics (if None, skips atom metrics)

    Returns
    -------
    dict[str, float]
        Dictionary containing all computed metrics
    """
    model.eval()
    metrics = {}

    with torch.no_grad():
        images = images.to(device)

        # Forward pass - handle both VAE and rVAE
        outputs = model(images)
        if len(outputs) == 3:
            # VAE: (recon, mu, logvar)
            recon, mu, logvar = outputs
        elif len(outputs) == 5:
            # rVAE: (rotated_recon, recon, theta, mu, logvar)
            recon, _, _, mu, logvar = outputs
        else:
            raise ValueError(f"Unexpected model output length: {len(outputs)}")

        # Reconstruction metrics
        recon_metrics = compute_reconstruction_metrics(images, recon)
        metrics.update(recon_metrics)

        # Latent space metrics
        latent_metrics = compute_latent_metrics(mu, logvar)
        metrics.update(latent_metrics)

        # Atom detection metrics (if lattice spacing provided)
        if lattice_spacing is not None:
            # Compute on first image in batch
            atom_metrics = compute_atom_detection_metrics(
                images[0], recon[0], lattice_spacing
            )
            metrics.update(atom_metrics)

    return metrics
