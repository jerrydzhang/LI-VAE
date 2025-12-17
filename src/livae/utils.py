from __future__ import annotations


import numpy as np
from numpy.typing import NDArray
import h5py
from pathlib import Path
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

from .filter import fft_spectra

NumericArray = NDArray[np.generic]
FloatArray = NDArray[np.float64]

__all__ = [
    "estimate_lattice_constant",
]


def estimate_lattice_constant(
    image: NumericArray,
    min_atom_size: float = 10.0,
    max_atom_size: float = 60.0,
    prominence_factor: float = 0.1,
) -> float:
    """Estimate lattice spacing via FFT with image-size-adaptive parameters.

    Uses pre-whitening (Gaussian background subtraction), FFT magnitude analysis,
    and dynamic frequency search to robustly identify hexagonal lattice peaks
    across different image resolutions. Adapts automatically to image size.

    Parameters
    ----------
    image : array-like, shape (M, M)
        Square 2D grayscale image array (typically uint16 or float64 STEM data).
    min_atom_size : float, optional
        Minimum expected atomic feature size in pixels. Default is 10.0.
    max_atom_size : float, optional
        Maximum expected atomic feature size in pixels. Default is 60.0.
    prominence_factor : float, optional
        Peak prominence threshold as fraction of max radial profile value.
        Default is 0.1 (10% of max). Lower values find more peaks; higher
        values (up to ~0.3) reduce false positives.

    Returns
    -------
    spacing : float
        Estimated lattice spacing in pixels. Returns fallback value 15.0
        if FFT peak detection fails.

    Notes
    -----
    This method preprocesses the image with Gaussian background subtraction
    to enhance periodic features, computes the FFT magnitude spectrum, and
    derives a radial profile. It then searches for prominent peaks within
    a frequency range determined by the expected atomic sizes, adapting
    to the input image dimensions. The approach is robust across varying resolutions
    and noise levels, making it suitable for automated lattice constant
    estimation in microscopy images.

    Examples
    --------
    >>> import numpy as np
    >>> from utils import estimate_lattice_constant
    >>> img = np.random.rand(256, 256)
    >>> spacing = estimate_lattice_constant(img)
    >>> isinstance(spacing, float)
    True
    """
    img_size = image.shape[0]

    sigma = img_size * 0.005
    background = gaussian_filter(image, sigma=sigma)
    whitened = np.asarray(image, dtype=np.float64) - background.astype(np.float64)

    magnitude, _ = fft_spectra(whitened)

    center_y, center_x = img_size // 2, img_size // 2
    y_grid, x_grid = np.ogrid[:img_size, :img_size]
    radius_grid = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2).astype(
        np.int32
    )

    radial_sum = np.bincount(radius_grid.ravel(), magnitude.ravel(), minlength=img_size)
    radial_count = np.bincount(radius_grid.ravel(), minlength=img_size)
    radial_count[radial_count == 0] = 1
    radial_profile = radial_sum / radial_count

    search_r_min = int(img_size / max_atom_size)
    search_r_max = int(img_size / min_atom_size)
    search_r_min = max(2, search_r_min)
    search_r_max = min(len(radial_profile) - 1, search_r_max)

    profile_slice = radial_profile[search_r_min : search_r_max + 1]
    max_val = np.max(profile_slice)

    peaks, _ = find_peaks(profile_slice, prominence=max_val * prominence_factor)

    if len(peaks) == 0:
        return 15.0

    best_fft_radius = peaks[0] + search_r_min
    spacing = img_size / best_fft_radius

    return spacing


def load_image_from_h5(
    file_path: Path | str,
    dataset_name: str = "image",
) -> FloatArray:
    """Load a 2D image from an HDF5 file.

    Parameters
    ----------
    file_path : Path or str
        Path to the HDF5 file.
    dataset_name : str, optional
        Name of the dataset within the HDF5 file. Default is 'image'.

    Returns
    -------
    image : ndarray, shape (M, N)
        Loaded 2D image array.
    """
    with h5py.File(file_path, "r") as h5_file:
        image = h5_file[dataset_name][:]
    return image
