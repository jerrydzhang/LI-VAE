# Nano-rVAE: Unsupervised Disentanglement of Atomic Lattices> **Hackathon Track:** Disentangling Nanoscale Materials Structure
> 
> 
> **Reference Paper:** Liu et al., *Advanced Materials* 2021 
> 
> 

## Scientific Motivation**The Challenge:** Material properties are governed by nanoscale defects (vacancies, dopants, grain boundaries), but current analysis relies on subjective, manual descriptors.
**The Goal:** Develop an unsupervised machine learning pipeline to automatically identify, classify, and "disentangle" these features without human-defined labels.

**The Key Innovation:**
Standard VAEs fail on atomic lattices because they entangle **orientation** (rotation) with **intrinsic structure** (defect type). We implement a **Rotationally Invariant Variational Autoencoder (rVAE)** that mathematically factors out rotation, allowing the latent space to purely represent physical defects.

## Methodology
Our pipeline adapts the reference architecture (Liu et al.) from **continuous domain walls** to **discrete atomic lattices** (MoS_2).

### 1. Unsupervised Segmentation
**Reference Paper Approach:** Used supervised `ResHED-net` to detect continuous domain walls.

* **Our Hackathon Approach:** Uses unsupervised **Peak Finding** (e.g., `skimage.feature.peak_local_max`).
* **Why:** The MoS2 dataset (`STEM_MoS2_monolayer_Data.ipynb`) lacks the ground-truth masks required for the paper's supervised network. Atomic lattices are high-contrast and rigid, making deterministic peak finding a robust, label-free alternative to the paper's ResHED-net.

### 2. Physics-Informed Patching* 
**Strategy:** We extract small square patches (approx. 64 \times 64 pixels) centered strictly on the detected atom coordinates.

* **Purpose:** This "centers" the physics. By ensuring the atom is always in the middle of the frame, the rVAE can effectively learn rotational invariance. Random crops would fail here.

### 3. rVAE Manifold Learning* 
**Model:** A VAE with a specialized encoder that outputs both a feature vector z and a rotation angle \theta.

**Latent Space:** The vector z captures the "disentangled" physics (e.g., Mo vs. S vs. Vacancy) independent of how the lattice is rotated.
