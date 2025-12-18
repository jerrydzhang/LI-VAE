from __future__ import annotations

import random
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
from scipy.spatial import KDTree
from skimage.feature import peak_local_max
from torch.utils.data import Dataset

from .filter import bandpass_filter, normalize_image
from .utils import estimate_lattice_constant

__all__ = [
    "PatchDataset",
    "AdaptiveLatticeDataset",
    "default_transform",
    "generate_lattice_grid",
]

TransformFn = Callable[..., torch.Tensor]


def generate_lattice_grid(
    image_shape: tuple[int, int],
    lattice_spacing: float,
    offset: tuple[float, float] = (0, 0),
) -> np.ndarray:
    """Generate a hexagonal or square lattice grid.

    Parameters
    ----------
    image_shape : tuple of int
        (height, width) of the image
    lattice_spacing : float
        Distance between grid points (in pixels)
    offset : tuple of float, optional
        (y, x) offset to shift the grid. Default is (0, 0).
    hexagonal : bool, optional
        If True, generates hexagonal lattice. If False, generates square grid.
        Default is True.

    Returns
    -------
    np.ndarray
        Array of shape (N, 2) containing (y, x) coordinates
    """
    h, w = image_shape
    y_off, x_off = offset

    dy = lattice_spacing
    dx = lattice_spacing * np.sqrt(3) / 2

    grid_points = []
    row_idx = 0
    y = y_off

    while y < h:
        x_start = x_off if row_idx % 2 == 0 else x_off + dx
        x = x_start

        while x < w:
            grid_points.append([y, x])
            x += 2 * dx

        y += dy
        row_idx += 1

    grid_points = np.array(grid_points)

    return grid_points


def default_transform(
    patch: torch.Tensor,
    flip_prob: float = 0.5,
    jitter_amount: int = 4,
    rotation: bool = True,
) -> torch.Tensor:
    """Default set of transforms: random flip, rotation, jitter, and scale"""
    scale_factor = random.uniform(0.9, 1.1)
    patch = TF.affine(
        patch,
        angle=0.0,
        translate=[0, 0],
        scale=scale_factor,
        shear=[0.0],
        interpolation=TF.InterpolationMode.BILINEAR,
    )

    if rotation:
        angle = random.uniform(0, 360)
        patch = TF.rotate(
            patch,
            angle=angle,
            interpolation=TF.InterpolationMode.BILINEAR,
            expand=False,
            fill=0,
        )

    if random.random() < flip_prob:
        patch = TF.hflip(patch)

    if random.random() < flip_prob:
        patch = TF.vflip(patch)

    if jitter_amount > 0:
        shift_x = random.randint(-jitter_amount, jitter_amount)
        shift_y = random.randint(-jitter_amount, jitter_amount)
        patch = torch.roll(patch, shifts=(shift_y, shift_x), dims=(-2, -1))

    return patch


def get_clean_peaks(img, min_distance=5, threshold_rel=0.01):
    """Detect peaks with refinement to local maxima.

    Parameters
    ----------
    img : np.ndarray
        2D image array
    min_distance : int
        Minimum distance between peaks
    threshold_rel : float
        Minimum intensity as fraction of image maximum (default 0.01 = 1%)
    """
    coords = peak_local_max(img, min_distance=min_distance, threshold_rel=threshold_rel)

    refined = []
    h, w = img.shape
    for r, c in coords:
        r_i, c_i = int(r), int(c)
        r1, r2 = max(0, r_i - 2), min(h, r_i + 3)
        c1, c2 = max(0, c_i - 2), min(w, c_i + 3)

        local_area = img[r1:r2, c1:c2]

        local_idx = np.unravel_index(np.argmax(local_area), local_area.shape)

        new_r = r1 + local_idx[0]
        new_c = c1 + local_idx[1]
        refined.append([new_r, new_c])

    return np.array(refined)


class PatchDataset(Dataset):
    def __init__(
        self,
        images: list[np.ndarray],
        patch_size: int,
        padding: int = 4,
        transform: TransformFn | None = default_transform,
    ):
        """Dataset of image patches centered on atomic positions.

        Parameters
        ----------
        images : list of np.ndarray
            List of 2D grayscale images (numpy arrays).
        patch_size : int
            Size of square patches to extract (in pixels).
        padding : int, optional
            Amount of extra padding to allow for croping and rotations without clipping.
            Default is 0.
        transform : callable, optional
            Optional transform to apply to each patch (e.g., data augmentation).
            Default is None.
        """

        self.patch_size = patch_size
        self.padding = padding
        self.transform = transform

        def preprocess_image(img: np.ndarray) -> np.ndarray:
            img = bandpass_filter(img, 20, 100)
            img = normalize_image(img)
            return img

        print("Preprocessing images (caching)...")
        self.images = [preprocess_image(img) for img in images]

        self.atom_coords = []

        for img in self.images:
            lattice_spacing = estimate_lattice_constant(img)
            coords = get_clean_peaks(img, min_distance=int(lattice_spacing * 0.15))
            off_edge_mask = (
                (coords[:, 0] >= self.patch_size // 2 + self.padding)
                & (coords[:, 0] <= img.shape[0] - self.patch_size // 2 - self.padding)
                & (coords[:, 1] >= self.patch_size // 2 + self.padding)
                & (coords[:, 1] <= img.shape[1] - self.patch_size // 2 - self.padding)
            )
            print(
                f"Detected {len(coords)} atoms, {np.sum(off_edge_mask)} after edge exclusion."
            )
            coords = coords[off_edge_mask]
            self.atom_coords.append(coords)

    def __len__(self) -> int:
        total_patches = 0
        for i in range(len(self.images)):
            total_patches += len(self.atom_coords[i])

        return total_patches

    def __getitem__(self, idx: int) -> torch.Tensor:
        img_idx = 0
        while idx >= len(self.atom_coords[img_idx]):
            idx -= len(self.atom_coords[img_idx])
            img_idx += 1

        cy, cx = self.atom_coords[img_idx][idx]

        img = self.images[img_idx]
        img_h, img_w = img.shape
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)

        shift_x = (img_w / 2.0) - cx
        shift_y = (img_h / 2.0) - cy

        patch = TF.affine(
            img_tensor,
            angle=0.0,
            translate=[shift_x, shift_y],
            scale=1.0,
            shear=[0.0],
            interpolation=TF.InterpolationMode.BILINEAR,
        ).squeeze(0)

        # Take a larger crop first to preserve content during rotation,
        # then rotate, and finally center-crop down to the requested size.
        padded_size = self.patch_size + 2 * self.padding
        patch_big = TF.center_crop(patch, [padded_size, padded_size])

        if self.transform:
            patch_big = self.transform(patch_big)

        patch_final = TF.center_crop(patch_big, [self.patch_size, self.patch_size])

        return patch_final

    def plot_peaks(
        self,
        img_idx: int,
        size: int | None = None,
        offset: tuple[int, int] = (0, 0),
    ) -> None:
        """Plot detected atom positions overlaid on the image.

        Parameters
        ----------
        img_idx : int
            Index of the image to plot.
        size : int, optional
            Size of the square region to display (in pixels). If None,
            displays the full image. Default is None.
        offset : tuple of int, optional
            (y, x) offset to apply the taken region. Useful when size is set.
            Default is (0, 0).
        """

        img = self.images[img_idx]
        coords = self.atom_coords[img_idx]

        if size is not None:
            y_off, x_off = offset
            img = img[y_off : y_off + size, x_off : x_off + size]
            coords = coords[
                (coords[:, 0] >= y_off)
                & (coords[:, 0] < y_off + size)
                & (coords[:, 1] >= x_off)
                & (coords[:, 1] < x_off + size)
            ]
            coords = coords - np.array([y_off, x_off])
        plt.figure(figsize=(6, 6))
        plt.imshow(img, cmap="gray")
        plt.scatter(
            coords[:, 1], coords[:, 0], s=30, edgecolor="red", facecolors="none"
        )
        plt.title(f"Image {img_idx} with Detected Atoms")
        plt.axis("off")
        plt.show()


class AdaptiveLatticeDataset(Dataset):
    """Dataset that uses local lattice vectors to adaptively sample lattice sites.

    Key idea: Use detected atoms to estimate local lattice vectors, then extrapolate
    to find neighboring lattice sites. This adapts to drift and distortion.
    """

    def __init__(
        self,
        images: list[np.ndarray],
        patch_size: int,
        padding: int = 32,
        transform: TransformFn | None = default_transform,
        detection_threshold: float = 0.6,
    ):
        """Dataset with adaptive lattice sampling.

        Parameters
        ----------
        images : list of np.ndarray
            List of 2D grayscale images (numpy arrays).
        patch_size : int
            Size of square patches to extract (in pixels).
        padding : int, optional
            Amount of extra padding for edge exclusion. Default is 4.
        transform : callable, optional
            Optional transform to apply to each patch. Default is default_transform.
        detection_threshold : float, optional
            Distance threshold (as fraction of lattice spacing) for matching
            predicted lattice sites to detected atoms. Default is 0.6.
        """
        self.patch_size = patch_size
        self.padding = padding
        self.transform = transform
        self.detection_threshold = detection_threshold

        def preprocess_image(img: np.ndarray) -> np.ndarray:
            img = bandpass_filter(img, 20, 100)
            img = normalize_image(img)
            return img

        self.images = [preprocess_image(img) for img in images]

        self.sample_coords = []
        self.labels = []

        for img in self.images:
            lattice_spacing = estimate_lattice_constant(img)

            atom_coords = get_clean_peaks(img, min_distance=int(lattice_spacing * 0.15))

            half_patch = self.patch_size // 2 + self.padding
            edge_mask = (
                (atom_coords[:, 0] >= half_patch)
                & (atom_coords[:, 0] <= img.shape[0] - half_patch)
                & (atom_coords[:, 1] >= half_patch)
                & (atom_coords[:, 1] <= img.shape[1] - half_patch)
            )
            atom_coords = atom_coords[edge_mask]

            tree = KDTree(atom_coords)
            threshold_dist = lattice_spacing * self.detection_threshold

            predicted_sites = []

            for atom in atom_coords:
                predicted_sites.append(atom.copy())

            for atom in atom_coords:
                k = min(7, len(atom_coords))
                _, indices = tree.query([atom], k=k)

                if len(indices[0]) < 3:  # type: ignore
                    continue

                neighbors = atom_coords[indices[0][1:]]  # type: ignore
                vectors = neighbors - atom

                best_v1, best_v2 = None, None
                max_independence = -1

                for i in range(len(vectors)):
                    for j in range(i + 1, len(vectors)):
                        v1, v2 = vectors[i], vectors[j]
                        norm1 = np.linalg.norm(v1)
                        norm2 = np.linalg.norm(v2)
                        if norm1 < 1e-6 or norm2 < 1e-6:
                            continue
                        independence = abs(np.cross(v1, v2)) / (norm1 * norm2)
                        if independence > max_independence:
                            max_independence = independence
                            best_v1, best_v2 = v1, v2

                if best_v1 is None or best_v2 is None:
                    continue

                neighbor_offsets = [
                    best_v1,
                    -best_v1,
                    best_v2,
                    -best_v2,
                    best_v1 + best_v2,
                    -(best_v1 + best_v2),
                    best_v1 - best_v2,
                    best_v2 - best_v1,
                ]

                for offset in neighbor_offsets:
                    predicted_pos = atom + offset

                    if not (
                        half_patch <= predicted_pos[0] <= img.shape[0] - half_patch
                        and half_patch <= predicted_pos[1] <= img.shape[1] - half_patch
                    ):
                        continue

                    predicted_sites.append(predicted_pos.copy())

            if len(predicted_sites) > 0:
                predicted_sites = np.array(predicted_sites)

                site_tree = KDTree(predicted_sites)

                cluster_radius = lattice_spacing * 0.35  # 25% of lattice spacing
                pairs = site_tree.query_pairs(r=cluster_radius)

                parent = list(range(len(predicted_sites)))

                def find(x):
                    if parent[x] != x:
                        parent[x] = find(parent[x])
                    return parent[x]

                def union(x, y):
                    px, py = find(x), find(y)
                    if px != py:
                        parent[px] = py

                for i, j in pairs:
                    union(i, j)

                clusters = {}
                for i in range(len(predicted_sites)):
                    root = find(i)
                    if root not in clusters:
                        clusters[root] = []
                    clusters[root].append(i)

                unique_sites = []
                for indices in clusters.values():
                    centroid = predicted_sites[indices].mean(axis=0)
                    unique_sites.append(centroid)

                predicted_sites = np.array(unique_sites)
            else:
                predicted_sites = np.array([]).reshape(0, 2)

            all_coords = []
            all_labels = []

            for site_pos in predicted_sites:
                dist_to_nearest, _ = tree.query([site_pos])

                if dist_to_nearest[0] < threshold_dist:
                    all_coords.append(site_pos)
                    all_labels.append(1)
                else:
                    all_coords.append(site_pos)
                    all_labels.append(0)

            all_coords = np.array(all_coords)
            all_labels = np.array(all_labels)

            n_atoms = np.sum(all_labels == 1)
            n_empty = np.sum(all_labels == 0)
            print(
                f"Adaptive lattice: {len(all_coords)} unique sites - "
                f"{n_atoms} with atoms, {n_empty} empty sites"
            )

            self.sample_coords.append(all_coords)
            self.labels.append(all_labels)

    def __len__(self) -> int:
        return sum(len(coords) for coords in self.sample_coords)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get patch and label"""
        img_idx = 0
        while idx >= len(self.sample_coords[img_idx]):
            idx -= len(self.sample_coords[img_idx])
            img_idx += 1

        cy, cx = self.sample_coords[img_idx][idx]
        img = self.images[img_idx]

        # Use padding to define a larger ROI to avoid black edges on rotation
        roi_buffer = max(16, 2 * self.padding)
        roi_size = self.patch_size + roi_buffer

        y_int, x_int = int(round(cy)), int(round(cx))

        y_start = y_int - roi_size // 2
        x_start = x_int - roi_size // 2
        y_end = y_start + roi_size
        x_end = x_start + roi_size

        h, w = img.shape
        pad_top = max(0, -y_start)
        pad_left = max(0, -x_start)
        pad_bottom = max(0, y_end - h)
        pad_right = max(0, x_end - w)

        y_start = max(0, y_start)
        x_start = max(0, x_start)
        y_end = min(h, y_end)
        x_end = min(w, x_end)

        roi = img[y_start:y_end, x_start:x_end]

        roi_tensor = torch.from_numpy(roi).float().unsqueeze(0).unsqueeze(0)

        if pad_top > 0 or pad_left > 0 or pad_bottom > 0 or pad_right > 0:
            roi_tensor = TF.pad(roi_tensor, [pad_left, pad_top, pad_right, pad_bottom])

        roi_h, roi_w = roi_tensor.shape[-2:]

        eff_x_start = x_int - roi_size // 2
        eff_y_start = y_int - roi_size // 2

        rel_cx = cx - eff_x_start
        rel_cy = cy - eff_y_start

        shift_x = (roi_w / 2.0) - rel_cx
        shift_y = (roi_h / 2.0) - rel_cy

        patch = TF.affine(
            roi_tensor,
            angle=0.0,
            translate=[shift_x, shift_y],
            scale=1.0,
            shear=[0.0],
            interpolation=TF.InterpolationMode.BILINEAR,
        ).squeeze(0)

        # First take a larger crop, apply transform (rotation), then crop back
        padded_size = self.patch_size + 2 * self.padding
        patch_big = TF.center_crop(patch, [padded_size, padded_size])

        if self.transform:
            patch_big = self.transform(patch_big)

        patch_cropped = TF.center_crop(patch_big, [self.patch_size, self.patch_size])

        min_val = patch_cropped.min()
        max_val = patch_cropped.max()
        if max_val > min_val:
            patch_final = (patch_cropped - min_val) / (max_val - min_val)
        else:
            patch_final = torch.zeros_like(patch_cropped)
        
        return patch_final

    def plot_lattice(
        self,
        img_idx: int,
        size: int | None = None,
        offset: tuple[int, int] = (0, 0),
    ) -> None:
        """Plot adaptive lattice sampling results."""
        img = self.images[img_idx]
        coords = self.sample_coords[img_idx]
        labels = self.labels[img_idx]

        if size is not None:
            y_off, x_off = offset
            img = img[y_off : y_off + size, x_off : x_off + size]

            in_view = (
                (coords[:, 0] >= y_off)
                & (coords[:, 0] < y_off + size)
                & (coords[:, 1] >= x_off)
                & (coords[:, 1] < x_off + size)
            )
            coords = coords[in_view] - np.array([y_off, x_off])
            labels = labels[in_view]

        plt.figure(figsize=(8, 8))
        plt.imshow(img, cmap="gray")

        atom_coords = coords[labels == 1]
        vacancy_coords = coords[labels == 0]

        if len(atom_coords) > 0:
            plt.scatter(
                atom_coords[:, 1],
                atom_coords[:, 0],
                s=50,
                c="green",
                marker="o",
                alpha=0.7,
                edgecolors="white",
                linewidths=1,
                label="Atoms",
            )
        if len(vacancy_coords) > 0:
            plt.scatter(
                vacancy_coords[:, 1],
                vacancy_coords[:, 0],
                s=60,
                c="red",
                marker="X",
                alpha=0.8,
                edgecolors="yellow",
                linewidths=1.5,
                label="Empty Sites",
            )

        plt.title(f"Adaptive Lattice - Image {img_idx}")
        plt.legend()
        plt.axis("off")
        plt.show()
