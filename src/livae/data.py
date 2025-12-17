from __future__ import annotations

from typing import Callable
import random

import numpy as np
import torchvision.transforms.functional as TF
from skimage.feature import peak_local_max
import torch
from torch.utils.data import Dataset

from .filter import bandpass_filter, normalize_image
from .utils import estimate_lattice_constant

__all__ = [
    "PatchDataset",
    "default_transform",
]

TransformFn = Callable[..., torch.Tensor]


def default_transform(
    patch: torch.Tensor,
    flip_prob: float = 0.5,
    jitter_amount: int = 2,
) -> torch.Tensor:
    """Default set of transforms: random flip, rotation and jitter."""
    angle = random.uniform(0, 360)
    patch = TF.rotate(patch, angle)

    if random.random() < flip_prob:
        patch = TF.hflip(patch)

    if random.random() < flip_prob:
        patch = TF.vflip(patch)

    if jitter_amount > 0:
        max_jitter = jitter_amount
        jitter_x = random.randint(-max_jitter, max_jitter)
        jitter_y = random.randint(-max_jitter, max_jitter)
        patch = TF.affine(
            patch, translate=[jitter_x, jitter_y], angle=0, scale=1.0, shear=[0]
        )

    return patch


class PatchDataset(Dataset):
    def __init__(
        self,
        images: list[np.ndarray],
        patch_size: int,
        padding: int = 16,
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

        self.images = [normalize_image(img) for img in images]

        self.atom_coords = []

        for img in self.images:
            filtered_img = bandpass_filter(img, low_cutoff=60, high_cutoff=300)
            lattice_spacing = estimate_lattice_constant(img)
            coords = peak_local_max(
                filtered_img,
                # This is done as the lattice spacing often misses the faiter sulfer atoms
                min_distance=int(lattice_spacing * 0.35),
                threshold_rel=0.2,
                exclude_border=False,
            )
            off_edge_mask = (
                (coords[:, 0] >= self.patch_size // 2 + self.padding)
                & (coords[:, 0] <= img.shape[0] - self.patch_size // 2 - self.padding)
                & (coords[:, 1] >= self.patch_size // 2 + self.padding)
                & (coords[:, 1] <= img.shape[1] - self.patch_size // 2 - self.padding)
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

        center_y, center_x = self.atom_coords[img_idx][idx]
        half_size = (self.patch_size // 2) + self.padding
        patch = self.images[img_idx][
            center_y - half_size : center_y + half_size,
            center_x - half_size : center_x + half_size,
        ]
        patch = torch.from_numpy(patch).float().unsqueeze(0)

        if self.transform:
            patch = self.transform(patch)

        patch = TF.center_crop(patch, [self.patch_size, self.patch_size])

        return patch
