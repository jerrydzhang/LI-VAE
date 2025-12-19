#!/usr/bin/env python3
"""
Compute latent embeddings (t-SNE / PCA) and plot colored by source image.
Automatically loads data from data/ and checkpoint from checkpoints/.
Saves output to `plots/latent_embeddings.png` and other subfolders.
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from livae.model import VAE, RVAE
from livae.utils import load_image_from_h5, clean_state_dict
from livae.data import AdaptiveLatticeDataset
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict

# -----------------------------
# Settings
# -----------------------------
IS_RVAE = False
DATA_DIR = Path("../data")
CHECKPOINT_DIR = Path("../checkpoints")
PLOTS_DIR = Path("../plots")
PATCH_SIZE = 128
PADDING = 16
BATCH_SIZE = 256
LATENT_DIM = 16


# -----------------------------
# Functions
# -----------------------------
@torch.no_grad()
def collect_stats(model, loader, dataset):
    model.eval()
    all_mu, all_logvar, all_err, all_idxs = [], [], [], []

    cum_lens = np.cumsum([0] + [len(c) for c in dataset.sample_coords])

    def map_index(global_idx):
        img_idx = np.searchsorted(cum_lens, global_idx, side="right") - 1
        local_idx = global_idx - cum_lens[img_idx]
        return int(img_idx), int(local_idx)

    start = 0
    for x in loader:
        x = x.to(next(model.parameters()).device)
        if IS_RVAE:
            _, recon, _, mu, logvar = model(x)
        else:
            recon, mu, logvar = model(x)
        err = F.mse_loss(recon, x, reduction="none").mean(dim=(1, 2, 3))
        all_mu.append(mu.cpu().numpy())
        all_logvar.append(logvar.cpu().numpy())
        all_err.append(err.cpu().numpy())
        B = x.size(0)
        idxs = [map_index(i) for i in range(start, start + B)]
        all_idxs.extend(idxs)
        start += B

    mu = np.concatenate(all_mu, axis=0)
    logvar = np.concatenate(all_logvar, axis=0)
    err = np.concatenate(all_err, axis=0)
    return mu, logvar, err, all_idxs


def embed_latents(latent, method="auto", seed=42):
    emb = None
    rng = np.random.RandomState(seed)
    if method in ("auto", "tsne"):
        try:
            emb = TSNE(
                n_components=2, random_state=rng, init="random", perplexity=30
            ).fit_transform(latent)
        except Exception:
            emb = None
    if emb is None:
        emb = PCA(n_components=2).fit_transform(latent)
    return emb


# ----- REVERTED plot_latents -----
def plot_latents(emb, out_path, image_sources=None):
    plt.figure(figsize=(6, 6))
    if image_sources is None:
        plt.scatter(emb[:, 0], emb[:, 1], s=8, cmap="tab10")
    else:
        cmap = plt.get_cmap("tab10")
        for i, img_idx in enumerate(np.unique(image_sources)):
            mask = image_sources == img_idx
            plt.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=8,
                color=cmap(i % 10),
                label=f"Image {img_idx} (n={mask.sum()})",
                alpha=0.8,
            )
        plt.legend(markerscale=2)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    plt.title("Latent Embedding")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")


# ---------------------------------


def plot_clusters_on_images(mu, idx_map, dataset, n_clusters=3, out_dir=None):
    if out_dir is None:
        out_dir = PLOTS_DIR / "clusters"
    out_dir.mkdir(parents=True, exist_ok=True)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(mu)

    img_patch_labels = defaultdict(list)
    for (img_idx, local_idx), label in zip(idx_map, labels):
        img_patch_labels[img_idx].append((local_idx, label))

    for img_idx, patches in img_patch_labels.items():
        coords = dataset.sample_coords[img_idx]
        cluster_map = np.zeros(dataset.images[img_idx].shape, dtype=int) - 1
        for local_idx, label in patches:
            x, y = map(int, coords[local_idx])
            cluster_map[y : y + dataset.patch_size, x : x + dataset.patch_size] = label

        plt.figure(figsize=(6, 6))
        plt.imshow(cluster_map, cmap="tab10", interpolation="none")
        plt.title(f"Image {img_idx} - Patch Clusters")
        plt.colorbar(label="Cluster ID")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_dir / f"image_{img_idx}_clusters.png", dpi=150)
        plt.close()
        print(
            f"Saved cluster map for image {img_idx} to {out_dir / f'image_{img_idx}_clusters.png'}"
        )


def plot_windows(mu, idx_map, window_sizes=[10, 20, 30, 60, 90, 120], out_dir=None):
    if out_dir is None:
        out_dir = PLOTS_DIR / "windows"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = np.array([i for _, i in idx_map])
    for ws in window_sizes:
        z_mean = mu  # just for demonstration
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        ax1.hist(z_mean[:, 0], bins=40, color="green")
        ax1.set_xlabel("Encoded angle", fontsize=16)
        ax1.set_ylabel("Count", fontsize=16)
        ax1.set_title(f"Window size = {ws}", fontsize=16)
        ax2.scatter(z_mean[:, 1], z_mean[:, 2], c=frames, cmap="viridis", s=8)
        ax2.set_xlabel("Latent 1", fontsize=16)
        ax2.set_ylabel("Latent 2", fontsize=16)
        clrbar = plt.colorbar(ax2.collections[0], ax=ax2)
        clrbar.set_label("Frame", fontsize=14)
        plt.tight_layout()
        plt.savefig(out_dir / f"latent_hist_scatter_ws{ws}.png", dpi=150)
        plt.close()
        print(
            f"Saved latent histogram & scatter for window size {ws} to {out_dir / f'latent_hist_scatter_ws{ws}.png'}"
        )


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from collections import defaultdict
from pathlib import Path


def plot_atom_clusters(
    mu, idx_map, dataset, n_clusters=3, out_dir=Path("../plots/atom_clusters")
):
    """
    Plot atom-level clusters for each image.

    mu: (N_atoms, latent_dim) latent vectors
    idx_map: list of (img_idx, local_idx) mapping latent to image/patch/atom
    dataset: AdaptiveLatticeDataset
    n_clusters: number of clusters for KMeans
    out_dir: directory to save plots
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Cluster the latent vectors
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    labels = kmeans.fit_predict(mu)

    # Organize atom coordinates per image
    img_atom_coords = defaultdict(list)
    for (img_idx, local_idx), label in zip(idx_map, labels):
        x, y = dataset.sample_coords[img_idx][local_idx]
        img_atom_coords[img_idx].append((x, y, label))

    # Plot each image
    cmap = plt.get_cmap("tab10")

    for img_idx, atoms in img_atom_coords.items():
        atoms = np.array(atoms)
        x, y, lbls = atoms[:, 0], atoms[:, 1], atoms[:, 2].astype(int)

        plt.figure(figsize=(6, 6))
        for cl in range(n_clusters):
            mask = lbls == cl
            plt.scatter(
                x[mask],
                y[mask],
                s=10,
                color=cmap(cl % 10),
                label=f"Cluster {cl}",
                alpha=0.8,
            )

        plt.gca().invert_yaxis()  # if needed to match image coords
        plt.title(f"Image {img_idx} - Atom Clusters")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend(markerscale=2)
        plt.axis("equal")
        plt.tight_layout()

        out_path = out_dir / f"image_{img_idx}_atom_clusters.png"
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved atom-level cluster plot for image {img_idx} to {out_path}")


# -----------------------------
# Main
# -----------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_name = "rvae_final_best5_final.pt" if IS_RVAE else "vae_best.pt"
    ckpt_path = CHECKPOINT_DIR / ckpt_name
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {})
    latent_dim = ckpt_args.get("latent_dim", LATENT_DIM)
    patch_size = ckpt_args.get("patch_size", PATCH_SIZE)

    model = (
        RVAE(latent_dim=latent_dim, in_channels=1, patch_size=patch_size).to(device)
        if IS_RVAE
        else VAE(latent_dim=latent_dim, in_channels=1, patch_size=patch_size).to(device)
    )
    model.load_state_dict(clean_state_dict(ckpt["model_state"]))
    model.eval()

    h5_paths = sorted([p for p in DATA_DIR.glob("*.h5")])
    if not h5_paths:
        raise FileNotFoundError("No H5 files found in data/")
    images = [load_image_from_h5(str(p)) for p in h5_paths]

    dataset = AdaptiveLatticeDataset(
        images, patch_size=patch_size, padding=PADDING, transform=None
    )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print("Extracting latent vectors...")
    mu, logvar, rec_err, idx_map = collect_stats(model, loader, dataset)
    image_sources = np.array([img_idx for img_idx, _ in idx_map])
    print(
        "Samples per image:", dict(zip(*np.unique(image_sources, return_counts=True)))
    )

    print("Embedding latents...")
    emb = embed_latents(mu, method="auto")
    plot_latents(emb, PLOTS_DIR / "latent_embeddings.png", image_sources=image_sources)

    plot_clusters_on_images(mu, idx_map, dataset)
    plot_windows(mu, idx_map)
    plot_atom_clusters(mu, idx_map, dataset, n_clusters=3)


if __name__ == "__main__":
    main()
