"""
Utility functions for the MLOps pipeline.
"""

import os
import hashlib


def get_file_hash(file_path: str) -> str:
    """Return the MD5 hash of a file."""
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def ensure_dir(path: str) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def model_exists(model_path: str) -> bool:
    """Return True if the model artifact file exists."""
    return os.path.isfile(model_path)
