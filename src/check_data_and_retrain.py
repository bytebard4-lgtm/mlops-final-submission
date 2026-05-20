"""
Continuous Training trigger:
Retrains the model only when data/iris_custom.csv has changed
(detected via MD5 hash comparison).
Uses shared utils for hashing and directory helpers.
"""

import os
import subprocess
from src.utils import get_file_hash, ensure_dir

DATA_PATH = "data/iris_custom.csv"
HASH_FILE = "data/data.hash"


def has_data_changed() -> bool:
    """Return True if the data file hash differs from the stored hash."""
    if not os.path.exists(HASH_FILE):
        print(f"No hash file found at {HASH_FILE} — treating as first run.")
        return True

    with open(HASH_FILE, "r") as f:
        old_hash = f.read().strip()

    new_hash = get_file_hash(DATA_PATH)
    changed = new_hash != old_hash

    if changed:
        print(f"Data changed: {old_hash[:8]}... → {new_hash[:8]}...")
    else:
        print("Data unchanged — skipping retraining.")

    return changed


def update_hash() -> None:
    """Persist the current data hash so future runs can compare."""
    ensure_dir(os.path.dirname(HASH_FILE))
    with open(HASH_FILE, "w") as f:
        f.write(get_file_hash(DATA_PATH))
    print(f"Hash updated: {HASH_FILE}")


def run_training() -> None:
    """Invoke the training pipeline as a subprocess."""
    print("Launching training pipeline...")
    result = subprocess.run(
        ["python", "src/train_models.py"],
        check=True
    )
    if result.returncode == 0:
        print("Training completed successfully.")


if __name__ == "__main__":
    if has_data_changed():
        run_training()
        update_hash()
    else:
        print("No retraining needed.")
