# src/prepare_data.py
import os
import shutil
import random
from pathlib import Path
from sklearn.model_selection import train_test_split

random.seed(42)

# Get the folder containing this script
HERE = Path(__file__).parent

# Paths relative to this script
DATA_DIR = HERE.parent / "data" / "images"
OUT_DIR = HERE.parent / "data" / "10breeds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# List of breeds you want to process
DEFAULT_BREEDS = [
    "Labrador_Retriever",
    "German_Shepherd",
    "Golden_Retriever",
    "Boxer",
    "Beagle",
    "Pomeranian",
    "Siberian_Husky",
    "Doberman",
    "Shih-Tzu",
    "Yorkshire_Terrier",
]

def get_actual_folders():
    """
    Return all folders in DATA_DIR that actually contain images.
    This avoids OneDrive ghost/online-only folders.
    """
    actual = []
    for f in DATA_DIR.iterdir():
        if f.is_dir():
            has_files = any(p.is_file() for p in f.iterdir())
            if has_files:
                actual.append(f.name)
    return actual

def clean_folder_name(folder_name):
    """
    Remove any WordNet prefix (like n02099712-) and normalize underscores/dashes.
    """
    # Remove prefix if exists
    if "-" in folder_name and folder_name[:1].lower() == "n" and folder_name[1:9].isdigit():
        folder_name = folder_name.split("-", 1)[1]
    # Replace underscores with spaces or keep as underscores (optional)
    folder_name = folder_name.replace("_", " ")
    return folder_name


def find_matching_folder(breed_name, available_folders):
    for folder in available_folders:
        if folder.lower().endswith(breed_name.lower()):
            return folder
    return None

def prepare(breeds=None, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    if breeds is None:
        breeds = DEFAULT_BREEDS

    available_folders = get_actual_folders()

    for split in ("train", "val", "test"):
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    for breed in breeds:
        matched_folder = find_matching_folder(breed, available_folders)
        if matched_folder is None:
            print(f"WARNING: breed folder not found or empty: {breed}. Skipping.")
            continue

        src = DATA_DIR / matched_folder
        imgs = [p for p in src.iterdir() if p.is_file()]
        if len(imgs) == 0:
            print(f"No images for {breed}, skipping.")
            continue

        # split images
        train_and_val, test = train_test_split(imgs, test_size=test_ratio, random_state=42)
        val_size = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(train_and_val, test_size=val_size, random_state=42)

        # copy to output folders
        for split_name, split_list in zip(["train", "val", "test"], [train, val, test]):
            for p in split_list:
                dest = OUT_DIR / split_name / breed
                dest.mkdir(parents=True, exist_ok=True)
                shutil.copy(p, dest / p.name)

        print(f"{breed}: train={len(train)} val={len(val)} test={len(test)}")

if __name__ == "__main__":
    prepare()
