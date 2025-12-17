from __future__ import annotations

from typing import Tuple

import numpy as np
from numpy.typing import NDArray
from scipy import fft
from skimage import exposure
from skimage.filters import unsharp_mask

__all__ = [
    "fft_spectra",
    "normalize_image",
    "lowpass_filter",
    "highpass_filter",
    "bandpass_filter",
    "enhance_lattice_contrast",
]

FloatArray = NDArray[np.float64]
NumericArray = NDArray[np.generic]


def _to_float_image(image: NumericArray) -> FloatArray:
    """Validate a 2D image array and cast it to float64."""

    array = np.asarray(image)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array, got shape {array.shape}")
    return array.astype(np.float64, copy=False)


def _radial_mask(
    shape: Tuple[int, int], *, low_cutoff: float = 0.0, high_cutoff: float | None = None
) -> NDArray[np.bool_]:
    """Create a circular (or annular) frequency mask for FFT filtering."""

    rows, cols = shape
    center_y, center_x = rows // 2, cols // 2
    y_grid, x_grid = np.ogrid[:rows, :cols]
    radius = np.sqrt((x_grid - center_x) ** 2 + (y_grid - center_y) ** 2)

    mask = radius >= low_cutoff
    if high_cutoff is not None:
        mask &= radius <= high_cutoff
    return mask


def fft_spectra(image: NumericArray) -> tuple[FloatArray, FloatArray]:
    """Compute magnitude and phase spectra (shifted) for a 2D image.

    Parameters
    ----------
    image : array-like, shape (M, N)
            Input 2D image array. May be integer or float type.

    Returns
    -------
    magnitude : ndarray, shape (M, N)
            Magnitude spectrum (absolute values of FFT), centered.
    phase : ndarray, shape (M, N)
            Phase spectrum (angles of FFT), centered.

    Examples
    --------
    >>> import numpy as np
    >>> from filter import fft_spectra
    >>> img = np.random.rand(32, 32)
    >>> mag, phase = fft_spectra(img)
    >>> mag.shape, phase.shape
    ((32, 32), (32, 32))
    """
    array = _to_float_image(image)
    fft_image = fft.fftshift(fft.fft2(array))
    magnitude = np.abs(fft_image)
    phase = np.angle(fft_image)
    return magnitude, phase


def normalize_image(image: NumericArray) -> FloatArray:
    """Normalize an image to the [0, 1] range, preserving shape.

    Handles constant-valued inputs by returning zeros. Useful for peak
    detection preprocessing.

    Parameters
    ----------
    image : array-like, shape (M, N)
            Input 2D image array.

    Returns
    -------
    normalized : ndarray, shape (M, N), dtype float64
            Image normalized to [0, 1] range. If all values are identical,
            returns zeros.

    Examples
    --------
    >>> import numpy as np
    >>> from filter import normalize_image
    >>> img = np.array([[0, 50], [100, 150]])
    >>> norm = normalize_image(img)
    >>> norm.min(), norm.max()
    (0.0, 1.0)
    """
    array = np.asarray(image, dtype=np.float64)
    min_val = float(np.min(array))
    ptp = float(np.ptp(array))
    if ptp == 0.0:
        return np.zeros_like(array)
    return (array - min_val) / ptp


def lowpass_filter(image: NumericArray, cutoff_radius: float) -> FloatArray:
    """Apply a circular low-pass FFT filter with the given cutoff radius.

    Removes high-frequency components (noise, fine detail) by attenuating
    all frequency components beyond the cutoff radius.

    Parameters
    ----------
    image : array-like, shape (M, N)
            Input 2D image array.
    cutoff_radius : float
            Frequency cutoff radius (in pixel-space frequency). Frequencies
            with radius > cutoff_radius are attenuated.

    Returns
    -------
    filtered : ndarray, shape (M, N), dtype float64
            Smoothed image with reduced high-frequency content.

    Examples
    --------
    >>> import numpy as np
    >>> from filter import lowpass_filter
    >>> # Create noisy image
    >>> img = np.random.rand(64, 64) + 10.0
    >>> # Apply low-pass with cutoff at 20 pixels
    >>> smooth = lowpass_filter(img, cutoff_radius=20)
    >>> smooth.std() < img.std()  # Should have lower variance
    True
    """
    array = _to_float_image(image)
    fft_image = fft.fftshift(fft.fft2(array))
    mask = _radial_mask(array.shape, high_cutoff=cutoff_radius)
    filtered = fft_image * mask
    return np.real(fft.ifft2(fft.ifftshift(filtered)))


def highpass_filter(image: NumericArray, cutoff_radius: float) -> FloatArray:
    """Apply a circular high-pass FFT filter with the given cutoff radius.

    Removes low-frequency components (background, uneven illumination) by
    attenuating all frequency components within the cutoff radius.

    Parameters
    ----------
    image : array-like, shape (M, N)
            Input 2D image array.
    cutoff_radius : float
            Frequency cutoff radius. Frequencies with radius < cutoff_radius
            are attenuated.

    Returns
    -------
    filtered : ndarray, shape (M, N), dtype float64
            Image with reduced low-frequency content and preserved fine detail.

    Examples
    --------
    >>> import numpy as np
    >>> from filter import highpass_filter
    >>> # Create image with gradient background + texture
    >>> bg = np.linspace(0, 100, 64)[:, None]
    >>> texture = 5 * np.sin(np.arange(64) / 5)
    >>> img = bg + texture[None, :]
    >>> # Remove background gradient
    >>> enhanced = highpass_filter(img, cutoff_radius=10)
    >>> # Background mostly removed, texture enhanced
    """
    array = _to_float_image(image)
    fft_image = fft.fftshift(fft.fft2(array))
    mask = _radial_mask(array.shape, low_cutoff=cutoff_radius)
    filtered = fft_image * mask
    return np.real(fft.ifft2(fft.ifftshift(filtered)))


def bandpass_filter(
    image: NumericArray, low_cutoff: float, high_cutoff: float
) -> FloatArray:
    """Apply a circular band-pass FFT filter between low and high cutoffs.

    Isolates a specific band of frequency components, useful for enhancing
    periodic lattice structures while suppressing both background and noise.
    Ideal for STEM lattice image preprocessing prior to peak detection.

    Parameters
    ----------
    image : array-like, shape (M, N)
            Input 2D image array.
    low_cutoff : float
            Lower frequency cutoff radius (inner radius of annulus).
    high_cutoff : float
            Upper frequency cutoff radius (outer radius of annulus).
            Must be strictly greater than low_cutoff.

    Returns
    -------
    filtered : ndarray, shape (M, N), dtype float64
            Image containing only frequency components in the band.

    Raises
    ------
    ValueError
            If high_cutoff <= low_cutoff.

    Examples
    --------
    >>> import numpy as np
    >>> from filter import bandpass_filter
    >>> # Simulate STEM lattice with noise and background
    >>> img = np.random.rand(128, 128)  # Noise + background
    >>> # Extract lattice frequencies (typical scale: 5-50 pixels)
    >>> lattice = bandpass_filter(img, low_cutoff=5, high_cutoff=50)
    >>> # Lattice structures are now enhanced
    """
    if high_cutoff <= low_cutoff:
        raise ValueError("high_cutoff must be greater than low_cutoff")

    array = _to_float_image(image)
    fft_image = fft.fftshift(fft.fft2(array))
    mask = _radial_mask(array.shape, low_cutoff=low_cutoff, high_cutoff=high_cutoff)
    filtered = fft_image * mask
    return np.real(fft.ifft2(fft.ifftshift(filtered)))


def enhance_lattice_contrast(
    image: NumericArray,
    *,
    clip_limit: float = 0.03,
    tile_grid_size: int | tuple[int, int] = 16,
    unsharp_radius: float = 0.5,
    unsharp_amount: float = 2.5,
    denoise: bool = True,
) -> FloatArray:
    """Enhance lattice visibility via CLAHE and aggressive unsharp masking.

    This function assumes the input is a 2D image. It performs:
    1) Normalization to [0, 1]
    2) Contrast Limited Adaptive Histogram Equalization (CLAHE) with smaller tiles
    3) Aggressive unsharp masking to accentuate atomic peaks/edges
    4) Optional denoising via morphological opening to suppress spurious peaks

    Parameters
    ----------
    image : array-like, shape (M, N)
            Input 2D image array.
    clip_limit : float, optional
            Normalized clip limit for CLAHE. Lower values limit noise amplification.
            Default is 0.03 (higher than before for more local boost).
    tile_grid_size : int | (int, int), optional
            Size of contextual regions for CLAHE. Smaller = more local contrast.
            Default is 16 (smaller than before for finer lattice structure).
    unsharp_radius : float, optional
            Radius for unsharp masking (in pixels). Smaller = finer detail.
            Default is 0.5.
    unsharp_amount : float, optional
            Strength of unsharp masking. Higher = more aggressive sharpening.
            Default is 2.5 (stronger than before).
    denoise : bool, optional
            If True, apply morphological opening to suppress isolated noise.

    Returns
    -------
    enhanced : ndarray, shape (M, N), dtype float64
            Enhanced image in [0, 1].
    """
    from scipy.ndimage import binary_opening

    arr = _to_float_image(image)
    arr = normalize_image(arr)

    # Apply CLAHE for aggressive local contrast enhancement
    clahe = exposure.equalize_adapthist(
        arr, clip_limit=clip_limit, kernel_size=tile_grid_size
    )

    # Sharpen peaks aggressively with small radius
    sharp = unsharp_mask(
        clahe, radius=unsharp_radius, amount=unsharp_amount, preserve_range=True
    )
    sharp = normalize_image(sharp)

    # Optional morphological denoising: open to suppress noise speckles
    if denoise:
        from scipy.ndimage import binary_erosion, binary_dilation

        thresh = np.percentile(sharp, 70)
        binary = sharp > thresh
        # Erode then dilate to remove small isolated peaks
        opened = binary_erosion(binary, iterations=1)
        opened = binary_dilation(opened, iterations=1)
        sharp = sharp * opened.astype(float)
        sharp = normalize_image(sharp)

    return sharp
