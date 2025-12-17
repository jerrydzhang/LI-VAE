# Copilot Instructions for mic-hackathon

## Project Overview
**Nano-rVAE: Unsupervised Disentanglement of Atomic Lattices**

A hackathon project implementing Rotationally Invariant Variational Autoencoders (rVAE) to automatically identify and classify nanoscale material defects in atomic resolution STEM microscopy images of MoS₂ lattices. The challenge: isolate intrinsic structural defects (vacancies, dopants) from rotational variants using unsupervised learning.

**Reference:** Liu et al., *Advanced Materials* 2021

## Current Implementation Status
### Stage 1: Data Pipeline (✅ COMPLETE)
- **Peak Detection**: `livae.filter` module implements bandpass filtering and image normalization
- **Lattice Analysis**: `livae.utils.estimate_lattice_constant()` uses FFT-based analysis to automatically determine atomic spacing
- **Patch Extraction**: `livae.data.PatchDataset` extracts 32×32 pixel patches centered on detected atoms with configurable padding
- **Data Augmentation**: Random rotation (0-360°), horizontal/vertical flips, and spatial jitter transforms

### Stage 2: Model Architecture (⚠️ NOT YET IMPLEMENTED)
The rVAE model needs to be built with:
- **Encoder**: CNN backbone → (z, theta) where z ∈ ℝⁿ is latent feature vector, theta ∈ [0, 2π) is rotation angle
- **Decoder**: Takes z, applies spatial transformer to rotate output by theta
- **Loss Function**: Reconstruction loss + KL divergence + rotation invariance constraint

### Stage 3: Training & Analysis (⚠️ NOT YET IMPLEMENTED)
- Training loop with Adam optimizer
- Latent space visualization (t-SNE/UMAP)
- Clustering analysis (expected: Mo vs S atoms, vacancy defects)

## Critical Algorithm Constraints
- **NO supervised segmentation** – use `peak_local_max` only for object detection
- **Rotational invariance required** – encoder must output (z, theta); decoder must rotate output by theta
- **All training data must be centered atom patches** – ensures effective rotation invariance learning
- **Expected latent clusters**: atom type (Mo vs S) and defect type (vacancy)

## Environment Setup
- **Python**: 3.13.7 with venv (`.venv/`)
- **Package Name**: `LI-VAE` (installed as editable package via `pip install -e .`)
- **Module Structure**: `src/livae/` contains core functionality
- **Activation**: Source `.venv/bin/activate` (already active in terminal)
- **Executable**: `/home/jerry/School/MICROHACKATHON/mic-hackathon/.venv/bin/python`

## Key Dependencies
- **PyTorch** (2.9.1): Deep learning framework for rVAE model
- **scikit-image** (0.25.2): Peak detection (`peak_local_max`) and image preprocessing
- **scipy** (1.16.3): Signal processing (FFT, filtering)
- **h5py** (3.15.1): HDF5 file I/O for STEM data
- **numpy** (2.3.5): Array operations
- **matplotlib** (3.10.8): Visualization
- **ipykernel** (7.1.0): Jupyter notebook support
- **pytest** (9.0.2): Testing framework

## Data Format
- **Input Files**: `data/HAADF1.h5`, `data/HAADF2.h5`, `data/HAADF3.h5`
- **Structure**: HDF5 format containing atomic resolution STEM images of MoS₂ lattices
- **Processing Pipeline**: 
  1. Load image → `normalize_image()` (0-1 range)
  2. Apply `bandpass_filter()` (low_cutoff=60, high_cutoff=300)
  3. Estimate lattice spacing via `estimate_lattice_constant()` (FFT-based)
  4. Detect peaks with `peak_local_max()` (min_distance ≈ 0.05 × lattice_spacing)
  5. Extract 64×64 patches via `PatchDataset` (with padding for rotation)

## Implemented Modules
### `src/livae/filter.py`
- `normalize_image()`: Scale images to [0, 1] range
- `bandpass_filter()`: Frequency-domain filtering for atom detection
- `fft_spectra()`: Compute FFT magnitude and phase spectra
- `lowpass_filter()`, `highpass_filter()`: Basic frequency filters

### `src/livae/utils.py`
- `estimate_lattice_constant()`: FFT-based lattice spacing detection with adaptive parameters
- `load_image_from_h5()`: Load images from HDF5 files

### `src/livae/data.py`
- `PatchDataset`: PyTorch Dataset for extracting centered atom patches
  - Automatically detects peaks and extracts patches
  - Applies augmentation transforms (rotation, flip, jitter)
  - Handles edge cases with configurable padding
- `default_transform()`: Data augmentation pipeline

## Active Notebooks
- `notebooks/dataset.ipynb`: Dataset creation and exploration
- `notebooks/explore_data_structure.ipynb`: HDF5 data structure analysis
- `notebooks/filtering_peak_detection.ipynb`: Peak detection algorithm testing

## Development Workflow
1. **Data Exploration**: Use `notebooks/filtering_peak_detection.ipynb` for algorithm testing
2. **Implementation Pattern**: 
   - Experiment in notebooks for rapid iteration
   - Extract stable code to modules in `src/livae/`
   - Add unit tests in `tests/` for critical functions
3. **Next Steps**:
   - Implement rVAE encoder architecture (CNN → z, theta)
   - Implement rVAE decoder with spatial transformer network
   - Create training loop with combined loss function
   - Visualize latent space clustering (t-SNE/UMAP)

## Code Style Guidelines
- **Model Implementation**: Use PyTorch (torch.nn.Module)
- **Image Processing**: Use scikit-image for preprocessing, scipy for signal processing
- **Data Loading**: Keep HDF5 operations in context managers: `with h5py.File('path.h5', 'r') as f:`
- **Array Operations**: NumPy arrays as primary data structure
- **Type Hints**: Use modern Python typing (e.g., `list[np.ndarray]` instead of `List[np.ndarray]`)
- **Documentation**: Google-style docstrings with Parameters, Returns, Examples sections
