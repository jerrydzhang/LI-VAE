import torch
import torch.nn as nn
import torch.nn.functional as F


def circular_distance(theta1: torch.Tensor, theta2: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute circular distance between angles accounting for wraparound.
    
    Args:
        theta1: Angle tensor [B, 1] or [B]
        theta2: Angle tensor [B, 1] or [B]
        eps: Small value for numerical stability
        
    Returns:
        Circular distance (mean over batch)
    """
    # Ensure both are [B, 1]
    if theta1.dim() == 1:
        theta1 = theta1.unsqueeze(1)
    if theta2.dim() == 1:
        theta2 = theta2.unsqueeze(1)
    
    # Compute angular difference
    diff = torch.abs(theta1 - theta2)
    
    # Account for wraparound: take minimum of diff and 2π - diff
    diff = torch.min(diff, 2 * torch.pi - diff)
    
    return torch.mean(diff)


def rotation_diversity_loss(theta: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    """Encourage diverse rotation angles by penalizing low standard deviation.
    
    Instead of using paired samples, we directly encourage the STN to produce
    diverse rotation estimates across a batch. This is simpler and more stable
    than cycle consistency.
    
    Args:
        theta: Rotation angles [B, 1]
        target_std: Target standard deviation (default 1.0 rad ≈ 57°)
        
    Returns:
        Loss that is 0 when std=target, increases as std deviates
    """
    batch_std = torch.std(theta)
    # Penalize deviation from target std
    loss = (batch_std - target_std) ** 2
    return loss


def cycle_consistency_loss(theta_original: torch.Tensor, theta_rotated: torch.Tensor, 
                           expected_angle: torch.Tensor) -> torch.Tensor:
    """Compute cycle consistency loss with expected angle difference.
    
    When patch is rotated by angle R, the STN should detect:
    - theta_original = θ on original patch
    - theta_rotated = θ - R on rotated patch (approximately)
    
    So: theta_rotated - theta_original ≈ -expected_angle
    
    Uses smooth circular loss: 1 - cos(diff) which is always differentiable
    and handles wraparound naturally.
    
    Args:
        theta_original: Rotation detected on original [B, 1]
        theta_rotated: Rotation detected on rotated patch [B, 1]
        expected_angle: Applied rotation angle in radians [B]
        
    Returns:
        Loss (scalar, mean over batch)
    """
    # Ensure shapes are [B, 1]
    if theta_original.dim() == 1:
        theta_original = theta_original.unsqueeze(1)
    if theta_rotated.dim() == 1:
        theta_rotated = theta_rotated.unsqueeze(1)
    if expected_angle.dim() == 0:
        expected_angle = expected_angle.unsqueeze(0)
    if expected_angle.dim() == 1:
        expected_angle = expected_angle.unsqueeze(1)
    
    # Compute predicted vs expected angle difference
    predicted_diff = theta_rotated - theta_original  # [B, 1]
    expected_diff = -expected_angle  # negative because rotation decreases detected angle
    
    # Smooth circular loss using cosine: 1 - cos(predicted - expected)
    # This naturally handles wraparound and is fully differentiable
    # cos(0) = 1, so loss = 0 when angles match
    # cos(π) = -1, so loss = 2 when angles differ by π
    diff = predicted_diff - expected_diff
    circular_loss = 1.0 - torch.cos(diff)
    
    return torch.mean(circular_loss)


class VAELoss(nn.Module):
    """VAE loss with configurable beta, using mean reductions for stable scaling."""

    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss.

        Uses mean reduction for proper per-element scaling.
        """
        # Reconstruction loss (mean over all elements)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")

        # KL divergence (mean over batch and latent dims)
        kld_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total_loss = recon_loss + self.beta * kld_loss
        return total_loss, recon_loss, kld_loss


class RVAELoss(nn.Module):
    """rVAE loss with beta (KL) and gamma (rotation diversity) weights.

    Can use either cycle consistency (paired rotations) or diversity loss
    (batch statistics) to encourage rotation detection.
    """

    def __init__(self, beta: float = 1.0, gamma: float = 0.0, use_diversity: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.use_diversity = use_diversity  # If True, use diversity loss instead of cycle loss

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        theta: torch.Tensor | None = None,
        theta_rotated: torch.Tensor | None = None,
        expected_angle: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute RVAE loss.

        Args:
            recon_x: Reconstructed images [B, C, H, W]
            x: Original images [B, C, H, W]
            mu: Latent mean [B, latent_dim]
            logvar: Latent log variance [B, latent_dim]
            theta: Rotation angle from original image [B, 1]
            theta_rotated: Rotation angle from rotated version [B, 1]
            expected_angle: Applied rotation angle in radians [B] or [B, 1]

        Returns:
            total_loss, recon_loss, kld_loss, rotation_loss
        """
        batch_size = x.size(0)
        
        # Reconstruction loss: sum over spatial dims, mean over batch
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size

        # KL divergence: sum over latent dims, mean over batch
        kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kld_loss = torch.mean(kld_per_sample)

        # Rotation loss: either cycle consistency or diversity
        if self.gamma > 0:
            if self.use_diversity and theta is not None:
                # Encourage diverse rotation predictions across batch
                rotation_loss = rotation_diversity_loss(theta, target_std=1.0)
            elif theta is not None and theta_rotated is not None and expected_angle is not None:
                # Enforce angle difference matches expected rotation
                rotation_loss = cycle_consistency_loss(theta, theta_rotated, expected_angle)
            else:
                rotation_loss = torch.tensor(0.0, device=recon_x.device)
        else:
            rotation_loss = torch.tensor(0.0, device=recon_x.device)

        total_loss = recon_loss + self.beta * kld_loss + self.gamma * rotation_loss

        return total_loss, recon_loss, kld_loss, rotation_loss
