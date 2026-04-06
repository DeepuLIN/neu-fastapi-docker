# src/download_data.py

import os
import shutil
import kagglehub


def download_neu_dataset():

    print("Downloading NEU dataset from KaggleHub...")

    path = kagglehub.dataset_download(
        "kaustubhdikshit/neu-surface-defect-database"
    )

    print("✅ Downloaded to:", path)

    # Target directory in your project
    target_dir = os.path.join("data", "raw", "NEU-DET")

    # Create if not exists
    os.makedirs(target_dir, exist_ok=True)

    print("📂 Moving dataset to:", target_dir)

    # Copy files
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(target_dir, item)

        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

    print("Dataset ready at:", target_dir)


if __name__ == "__main__":
    download_neu_dataset()