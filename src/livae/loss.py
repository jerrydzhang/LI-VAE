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
    """rVAE loss with beta (KL) and gamma (cycle) weights.

    Cycle loss enforces that the detected rotation angle difference matches
    the expected rotation angle applied to generate the rotated patch.
    All losses use mean reduction for proper per-element scaling.
    """

    def __init__(self, beta: float = 1.0, gamma: float = 0.0) -> None:
        super().__init__()
        self.beta = beta
        self.gamma = gamma

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

        Uses mean reduction for proper per-element scaling.

        Args:
            recon_x: Reconstructed images [B, C, H, W]
            x: Original images [B, C, H, W]
            mu: Latent mean [B, latent_dim]
            logvar: Latent log variance [B, latent_dim]
            theta: Rotation angle from original image [B, 1]
            theta_rotated: Rotation angle from rotated version [B, 1]
            expected_angle: Applied rotation angle in radians [B] or [B, 1]

        Returns:
            total_loss, recon_loss, kld_loss, cycle_loss
        """
        batch_size = x.size(0)
        
        # Reconstruction loss: sum over spatial dims, mean over batch
        # This keeps gradients in trainable range while normalizing by batch
        recon_loss = F.mse_loss(recon_x, x, reduction="sum") / batch_size

        # KL divergence: sum over latent dims, mean over batch
        kld_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kld_loss = torch.mean(kld_per_sample)

        # Cycle consistency loss: enforce angle difference matches expected rotation
        # NOTE: We use stop_gradient on theta to make cycle loss primarily guide STN
        # without forcing the entire encoder representation to match. This reduces
        # the conflict between reconstruction (which wants canonical normalization)
        # and cycle loss (which wants rotation detection).
        if (theta is not None and theta_rotated is not None and 
            expected_angle is not None and self.gamma > 0):
            # Stop gradients on theta_original so cycle loss only updates STN through theta_rotated
            theta_sg = theta.detach()
            cycle_loss = cycle_consistency_loss(theta_sg, theta_rotated, expected_angle)
        else:
            cycle_loss = torch.tensor(0.0, device=recon_x.device)

        total_loss = recon_loss + self.beta * kld_loss + self.gamma * cycle_loss

        return total_loss, recon_loss, kld_loss, cycle_loss
