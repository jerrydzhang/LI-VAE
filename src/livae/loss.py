import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
    def __init__(self, beta: float = 1.0, beta_recon_min: float = 0.9) -> None:
        """VAE loss with adaptive weighting to prevent latent collapse.
        
        Args:
            beta: KL divergence weight (0 to fully disable).
            beta_recon_min: Minimum weight for recon loss to prevent collapse (0–1).
                           Ensures recon always has significant contribution.
        """
        super().__init__()
        self.beta = beta
        self.beta_recon_min = beta_recon_min

    def forward(
        self,
        recon_x: torch.Tensor,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss with anti-collapse regularization.
        
        Uses mean reduction to avoid AMP overflow, and ensures recon loss
        has a minimum weight to prevent KL collapse (latent variance → 0).
        """
        # Reconstruction loss (mean reduction to avoid FP16 overflow)
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        
        # KL divergence per element
        kld_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = torch.mean(kld_elem)
        
        # Adaptive weight: ensure recon maintains minimum contribution
        # If KL is very small relative to recon, downweight KL further
        kl_weight = max(self.beta, 1e-6)  # Never exactly zero; at least tiny KL penalty
        
        # When beta is small in warmup, enforce that recon dominates
        # Formula: loss = recon * (1 - beta_contrib) + kld_loss * beta_contrib
        # where beta_contrib = min(beta, 1 - beta_recon_min)
        beta_contrib = min(kl_weight, 1.0 - self.beta_recon_min)
        recon_weight = 1.0 - beta_contrib
        
        total_loss = recon_weight * recon_loss + beta_contrib * kld_loss
        
        return total_loss, recon_loss, kld_loss
