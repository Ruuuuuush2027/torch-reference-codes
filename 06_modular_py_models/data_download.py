"""A file defining the download function"""

import os
import zipfile

import pathlib
from pathlib import Path

import requests

def download_sample_to_path(path: str):
    """Given a path, download the pizza sushi steak data folder to the directory

    Args:
        path: a string leading to the path

    Returns:
        A tuple of (train_dir, test_dir)
    """

    data_path = Path(path)
    image_path = data_path / "pizza_steak_sushi"

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Download pizza, steak, sushi data
        with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
            request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            print("Downloading pizza, steak, sushi data...")
            f.write(request.content)

        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
            print("Unzipping pizza, steak, sushi data...")
            zip_ref.extractall(image_path)

        # Remove zip file
        os.remove(data_path / "pizza_steak_sushi.zip")
    
    train_dir = image_path / "train"
    test_dir = image_path / "test"

    return train_dir, test_dir
