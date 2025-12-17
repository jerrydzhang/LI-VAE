"""
LI-VAE: Latent Invariance Variational Autoencoder

A package for unsupervised learning of rotationally invariant representations
from atomic resolution STEM microscopy images.
"""

from livae.data import PatchDataset, default_transform
from livae.filter import (
    bandpass_filter,
    fft_spectra,
    highpass_filter,
    lowpass_filter,
    normalize_image,
)
from livae.loss import VAELoss
from livae.metrics import (
    compute_all_metrics,
    compute_atom_detection_metrics,
    compute_latent_metrics,
    compute_psnr,
    compute_reconstruction_metrics,
    compute_ssim,
)
from livae.model import RVAE, VAE, Decoder, Encoder, RotationSTN
from livae.train import (
    MetricLogger,
    evaluate,
    evaluate_rotation_invariance,
    evaluate_rvae,
    log_reconstructions_tensorboard,
    log_scalar_metrics_tensorboard,
    train_one_epoch,
    train_rvae_one_epoch,
)
from livae.utils import estimate_lattice_constant, load_image_from_h5

__version__ = "0.1.0"

__all__ = [
    # Data
    "PatchDataset",
    "default_transform",
    # Filtering
    "normalize_image",
    "bandpass_filter",
    "fft_spectra",
    "lowpass_filter",
    "highpass_filter",
    # Loss
    "VAELoss",
    # Models
    "VAE",
    "RVAE",
    "Encoder",
    "Decoder",
    "RotationSTN",
    # Training
    "train_one_epoch",
    "evaluate",
    "train_rvae_one_epoch",
    "evaluate_rvae",
    "evaluate_rotation_invariance",
    "log_reconstructions_tensorboard",
    "log_scalar_metrics_tensorboard",
    "MetricLogger",
    # Metrics
    "compute_psnr",
    "compute_ssim",
    "compute_reconstruction_metrics",
    "compute_latent_metrics",
    "compute_atom_detection_metrics",
    "compute_all_metrics",
    # Utils
    "load_image_from_h5",
    "estimate_lattice_constant",
]
