"""
Simple Continuous Training trigger:
Retrains model if data changes.
"""

import os
import hashlib
import subprocess

DATA_PATH = "data/iris_custom.csv"
HASH_FILE = "data/data.hash"


def get_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def has_data_changed():
    if not os.path.exists(HASH_FILE):
        return True

    with open(HASH_FILE, "r") as f:
        old_hash = f.read()

    new_hash = get_hash(DATA_PATH)

    return new_hash != old_hash


def update_hash():
    with open(HASH_FILE, "w") as f:
        f.write(get_hash(DATA_PATH))


def run_training():
    subprocess.run(["python", "src/train_models.py"])


if __name__ == "__main__":

    if has_data_changed():
        print("Data changed → retraining model...")
        run_training()
        update_hash()
    else:
        print("No changes detected → skipping training")
