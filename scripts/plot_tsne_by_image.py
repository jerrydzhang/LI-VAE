#!/usr/bin/env python3
"""
Compute latent embeddings (t-SNE/PCA) and plot colored by source image.
Saves output to `runs/embedding_by_image.png`.
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from livae.model import VAE, RVAE
from livae.utils import load_image_from_h5, clean_state_dict
from livae.data import AdaptiveLatticeDataset

IS_RVAE = True


def collect_stats(model, loader, dataset):
    model.eval()
    all_mu = []
    all_logvar = []
    all_err = []
    all_idxs = []

    cum_lens = np.cumsum([0] + [len(c) for c in dataset.sample_coords])

    def map_index(global_idx):
        img_idx = np.searchsorted(cum_lens, global_idx, side="right") - 1
        local_idx = global_idx - cum_lens[img_idx]
        return int(img_idx), int(local_idx)

    start = 0
    for batch in loader:
        x = batch.to(device)
        if IS_RVAE:
            # rVAE: (rotated_recon, recon, theta, mu, logvar)
            _, recon, _, mu, logvar = model(x)
        else:
            recon, mu, logvar = model(x)
        err = (
            F.mse_loss(recon, x, reduction="none")
            .mean(dim=(1, 2, 3))
            .detach()
            .cpu()
            .numpy()
        )
        all_mu.append(mu.detach().cpu().numpy())
        all_logvar.append(logvar.detach().cpu().numpy())
        all_err.append(err)
        B = x.size(0)
        idxs = [map_index(i) for i in range(start, start + B)]
        all_idxs.extend(idxs)
        start += B

    mu = np.concatenate(all_mu, axis=0)
    logvar = np.concatenate(all_logvar, axis=0)
    err = np.concatenate(all_err, axis=0)
    return mu, logvar, err, all_idxs


def embed_latents(latent, errors=None, method="auto", seed=42):
    rng = np.random.RandomState(seed)
    emb = None
    used = None
    if method in ("auto", "tsne"):
        try:
            from sklearn.manifold import TSNE

            emb = TSNE(
                n_components=2, random_state=rng, init="random", perplexity=30
            ).fit_transform(latent)
            used = "t-SNE"
        except Exception:
            emb = None
    if emb is None:
        x = latent - latent.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(x, full_matrices=False)
        emb = U[:, :2] * S[:2]
        used = "PCA"
    return emb, used


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    if IS_RVAE:
        ckpt_path = root / "checkpoints" / "rvae_best.pt"
    else:
        ckpt_path = root / "checkpoints" / "vae_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    latent_dim = ckpt_args.get("latent_dim", 16)
    patch_size = ckpt_args.get("patch_size", 128)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if IS_RVAE:
        model = RVAE(latent_dim=latent_dim, in_channels=1, patch_size=patch_size).to(
            device
        )
    else:
        model = VAE(latent_dim=latent_dim, in_channels=1, patch_size=patch_size).to(
            device
        )

    model.load_state_dict(clean_state_dict(ckpt["model_state"]))
    model.eval()

    data_dir = root / "data"
    h5_paths = sorted([str(p) for p in data_dir.glob("*.h5")])
    images = [load_image_from_h5(p) for p in h5_paths]

    dataset = AdaptiveLatticeDataset(
        images, patch_size=patch_size, padding=16, transform=None
    )
    batch_size = 256 if torch.cuda.is_available() else 64
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    print("Collecting stats (this may take a minute)...")
    mu, logvar, rec_err, idx_map = collect_stats(model, loader, dataset)
    print(f"Collected: mu={mu.shape}, rec_err={rec_err.shape}")

    image_sources = np.array([im for im, _ in idx_map])
    unique_imgs, counts = np.unique(image_sources, return_counts=True)
    print("Samples per image:", dict(zip(unique_imgs.tolist(), counts.tolist())))

    emb_src, used_src = embed_latents(mu, rec_err, method="auto")

    plt.figure(figsize=(6, 6))
    cmap = plt.get_cmap("tab10")
    for i, img_idx in enumerate(np.unique(image_sources)):
        mask = image_sources == img_idx
        plt.scatter(
            emb_src[mask, 0],
            emb_src[mask, 1],
            s=8,
            color=cmap(i % 10),
            label=f"Image {img_idx} (n={mask.sum()})",
            alpha=0.8,
        )
    plt.legend(markerscale=3)
    plt.title(f"Latent Embedding colored by Image Source ({used_src})")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    out_dir = root / "runs" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "embedding_by_image.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved embedding plot to: {out_path}")
