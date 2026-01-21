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
RESULTS_DIR = Path("checkpoints/ray_results/rvae_tune")
DATA_FILE = Path("data/HAADF1.h5")
# Automatically use GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def analyze_trial(trial: Trial, device: str):
    """Loads model from a trial and runs invariance analysis."""
    try:
        print(f"Analyzing trial: {Path(trial.path).name}")
        print(f"  - Loss: {trial.last_result.get('loss', 'N/A'):.4f}")
        print(f"  - Beta: {trial.config.get('beta', 'N/A')}")
        print(f"  - Latent Dim: {trial.config.get('latent_dim', 'N/A')}")

        print("Loading model from checkpoint...")
        model, patch_size = load_model_from_checkpoint(trial, device)
        print(f"Model loaded successfully. Using patch size: {patch_size}")

        # Re-using the global DATA_FILE, could be passed as an arg
        original_patch = get_image_patch(DATA_FILE, patch_size, device).unsqueeze(0)
        rotated_patch = TF.rotate(original_patch, 90)

        print("Performing inference...")
        with torch.no_grad():
            _, _, _, mu_original, _ = model(original_patch)
            _, _, _, mu_rotated, _ = model(rotated_patch)

        euclidean_dist = torch.norm(mu_original - mu_rotated, p=2).item()
        cosine_sim = F.cosine_similarity(mu_original, mu_rotated).item()

        print("\n--- Invariance Results for this trial ---")
        print(f"Euclidean Distance: {euclidean_dist:.6f}")
        print(f"Cosine Similarity: {cosine_sim:.6f}")

        if cosine_sim > 0.99:
            print(">>> Verdict: HIGHLY rotationally invariant.")
        elif cosine_sim > 0.95:
            print(">>> Verdict: LARGELY rotationally invariant.")
        else:
            print(">>> Verdict: LOW rotational invariance.")

    except Exception as e:
        print(f"Could not analyze trial {trial.trial_id}: {e}")
        import traceback

        traceback.print_exc()


def load_model_from_checkpoint(best_trial: Trial, device: str) -> RVAE:
    """
    Loads the RVAE model from the best trial's checkpoint.
    """
    config = best_trial.config
    latent_dim = config.get("latent_dim", 10)
    patch_size = config.get("patch_size", 64)

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

        # Use torch.load with explicit CPU map_location to handle CUDA-saved checkpoints on CPU-only hosts
        checkpoint_data = torch.load(
            checkpoint_file,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

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

    print("Filtering and sorting trials by loss...")

    trials = analysis.trials
    completed_trials = [t for t in trials if t.status == "TERMINATED" and t.checkpoint]

    if not completed_trials:
        print("No completed trials with checkpoints found.")
        return

    sorted_trials = sorted(
        completed_trials, key=lambda t: t.last_result.get("loss", float("inf"))
    )

    top_n = 5
    print(f"\n--- Analyzing Top {top_n} Trials ---")

    for i, trial in enumerate(sorted_trials[:top_n]):
        print(f"\n--- Processing Trial #{i + 1}/{top_n} ---")
        analyze_trial(trial, DEVICE)


if __name__ == "__main__":
    main()
