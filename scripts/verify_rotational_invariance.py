import torch
import torch.nn.functional as F
from ray import tune
from ray.tune.experiment.trial import Trial
from pathlib import Path
import torchvision.transforms.functional as TF
import h5py
import pickle

from livae.model import RVAE
from livae.utils import load_image_from_h5, clean_state_dict

# --- Configuration ---
RESULTS_DIR = Path("/home/jez21005/ray_results/rvae_tune")
DATA_FILE = Path("data/HAADF1.h5")
# Automatically use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_best_trial_from_analysis(analysis: tune.ExperimentAnalysis) -> Trial:
    """
    Retrieves the best trial from a Ray Tune ExperimentAnalysis object.
    """
    best_trial = analysis.get_best_trial(metric="loss", mode="min")
    if best_trial is None:
        raise ValueError(
            "Could not find best trial. Ensure the experiment has completed."
        )
    return best_trial


def load_model_from_checkpoint(best_trial: Trial, device: str) -> RVAE:
    """
    Loads the RVAE model from the best trial's checkpoint.
    """
    config = best_trial.config
    latent_dim = config.get("latent_dim", 10)
    # patch_size is not in the config so we default to 64
    patch_size = 64

    model = RVAE(latent_dim=latent_dim, patch_size=patch_size)

    if best_trial.checkpoint is None:
        raise ValueError("No checkpoint found for the best trial.")

    with best_trial.checkpoint.as_directory() as checkpoint_dir:
        checkpoint_file = Path(checkpoint_dir) / "checkpoint.pkl"
        
        if not checkpoint_file.is_file():
            files_in_checkpoint = list(Path(checkpoint_dir).iterdir())
            raise FileNotFoundError(
                f"Checkpoint file '{checkpoint_file.name}' not found in {checkpoint_dir}. "
                f"Files found: {files_in_checkpoint}"
            )

        # Check file size
        if checkpoint_file.stat().st_size == 0:
            raise ValueError(f"Checkpoint file is empty: {checkpoint_file}")

        # Since we are on a GPU machine now, let's try pure pickle.load first.
        # This is the exact counterpart to how the file was saved.
        with open(checkpoint_file, "rb") as f:
            checkpoint_data = pickle.load(f)

    # The model state is stored under 'model_state_dict'
    model_state_dict = checkpoint_data["model_state_dict"]

    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    return model, patch_size


def get_image_patch(image_path: Path, patch_size: int, device: str) -> torch.Tensor:
    """
    Loads an image and extracts a central patch.
    """
    image = load_image_from_h5(image_path)
    image = torch.from_numpy(image).float()

    # Normalize image to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())

    # Get center crop
    center_y, center_x = image.shape[0] // 2, image.shape[1] // 2
    top = center_y - patch_size // 2
    left = center_x - patch_size // 2
    patch = TF.crop(image.unsqueeze(0), top, left, patch_size, patch_size)

    return patch.to(device)


def main():
    """
    Main function to verify rotational invariance.
    """
    print(f"Loading results from: {RESULTS_DIR}")
    if not RESULTS_DIR.exists():
        print(f"Error: Results directory not found at '{RESULTS_DIR}'")
        return

    analysis = tune.ExperimentAnalysis(str(RESULTS_DIR.resolve()))

    print("Finding the best trial...")
    best_trial = get_best_trial_from_analysis(analysis)

    trial_name = Path(best_trial.path).name
    print(f"Best trial found: {trial_name}")
    print(f"  - Loss: {best_trial.last_result.get('loss', 'N/A'):.4f}")
    print(f"  - Latent Dim: {best_trial.config.get('latent_dim', 'N/A')}")
    print(f"  - Beta: {best_trial.config.get('beta', 'N/A')}")

    print("\nLoading model from checkpoint...")
    model, patch_size = load_model_from_checkpoint(best_trial, DEVICE)
    print(f"Model loaded successfully. Using patch size: {patch_size}")

    print(f"\nLoading and preparing image patch from: {DATA_FILE}")
    original_patch = get_image_patch(DATA_FILE, patch_size, DEVICE)

    # Create a 90-degree rotated version
    rotated_patch = TF.rotate(original_patch, 90)
    print("Original and 90-degree rotated patches created.")

    print("\nPerforming inference to get latent vectors...")
    with torch.no_grad():
        _, _, _, mu_original, _ = model(original_patch)
        _, _, _, mu_rotated, _ = model(rotated_patch)

    print("Comparing latent vectors...")

    # --- Quantitative Analysis ---
    # Euclidean Distance
    euclidean_dist = torch.norm(mu_original - mu_rotated, p=2).item()

    # Cosine Similarity
    cosine_sim = F.cosine_similarity(mu_original, mu_rotated).item()

    print("\n--- Verification Results ---")
    print(f"Euclidean Distance between latent vectors (mu): {euclidean_dist:.6f}")
    print(f"Cosine Similarity between latent vectors (mu): {cosine_sim:.6f}")

    print("\n--- Interpretation ---")
    if cosine_sim > 0.99 and euclidean_dist < 0.1:
        print("The model appears to be HIGHLY rotationally invariant.")
        print(
            "A high cosine similarity (>0.99) and low Euclidean distance (<0.1) are excellent indicators."
        )
    elif cosine_sim > 0.95 and euclidean_dist < 0.5:
        print("The model appears to be LARGELY rotationally invariant.")
        print("The latent vectors are very similar, but not identical.")
    else:
        print("The model's rotational invariance is LOW or BROKEN.")
        print(
            "The latent vectors for the original and rotated images are significantly different."
        )


if __name__ == "__main__":
    main()
