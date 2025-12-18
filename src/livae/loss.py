import torch
import torch.nn as nn
import torch.nn.functional as F


class VAELoss(nn.Module):
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
        # Use mean reductions to avoid FP16 overflow under AMP with large images/batches
        recon_loss = F.mse_loss(recon_x, x, reduction="mean")
        kld_elem = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        kld_loss = torch.mean(kld_elem)

        total_loss = recon_loss + self.beta * kld_loss
        return total_loss, recon_loss, kld_loss
