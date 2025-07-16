#!/usr/bin/env python3
"""
Dataset download script for Food Classification project.

This script automatically downloads the pizza_steak_sushi dataset,
extracts it to the correct directory structure, and verifies the setup.
"""

import os
import zipfile
import requests
from pathlib import Path
from typing import Optional


def download_data(source: str, 
                  destination: str,
                  remove_source: bool = True) -> Path:
    """Downloads a zipped dataset from source and unzips to destination.

    Args:
        source (str): A link to a zipped file containing data.
        destination (str): A target directory to unzip data to.
        remove_source (bool): Whether to remove the source after downloading and extracting.
    
    Returns:
        pathlib.Path to downloaded data.
    """
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / destination

    # If the image folder doesn't exist, download it and prepare it... 
    if image_path.is_dir():
        print(f"[INFO] {image_path} directory exists, skipping download.")
    else:
        print(f"[INFO] Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)
        
        # Download pizza, steak, sushi data
        target_file = Path(source).name
        try:
            with open(data_path / target_file, "wb") as f:
                request = requests.get(source)
                request.raise_for_status()  # Raise an exception for bad status codes
                print(f"[INFO] Downloading {target_file} from {source}...")
                f.write(request.content)

            # Unzip pizza, steak, sushi data
            with zipfile.ZipFile(data_path / target_file, "r") as zip_ref:
                print(f"[INFO] Unzipping {target_file} data...") 
                zip_ref.extractall(image_path)

            # Remove .zip file
            if remove_source:
                os.remove(data_path / target_file)
                print(f"[INFO] Removed {target_file}")
                
        except requests.RequestException as e:
            print(f"[ERROR] Failed to download data: {e}")
            return None
        except zipfile.BadZipFile as e:
            print(f"[ERROR] Failed to extract zip file: {e}")
            return None
    
    return image_path


def verify_data_structure(data_path: Path) -> bool:
    """Verify that the downloaded data has the correct structure.
    
    Args:
        data_path (Path): Path to the dataset directory.
        
    Returns:
        bool: True if structure is correct, False otherwise.
    """
    print(f"[INFO] Verifying data structure for {data_path}...")
    
    # Check if main directories exist
    train_dir = data_path / "train"
    test_dir = data_path / "test"
    
    if not train_dir.exists():
        print(f"[ERROR] Training directory not found: {train_dir}")
        return False
        
    if not test_dir.exists():
        print(f"[ERROR] Test directory not found: {test_dir}")
        return False
    
    # Check for class directories
    expected_classes = ["pizza", "steak", "sushi"]
    
    for split in ["train", "test"]:
        split_dir = data_path / split
        for class_name in expected_classes:
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"[ERROR] Class directory not found: {class_dir}")
                return False
            
            # Count images in class directory
            image_files = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
            print(f"[INFO] Found {len(image_files)} images in {class_dir}")
            
            if len(image_files) == 0:
                print(f"[WARNING] No images found in {class_dir}")
    
    print(f"[SUCCESS] Data structure verified successfully!")
    return True


def main():
    """Main function to download and verify the pizza_steak_sushi dataset."""
    print("=" * 60)
    print("Food Classification Dataset Setup")
    print("=" * 60)
    
    # Dataset URL from mrdbourke's pytorch-deep-learning repository
    source_url = "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip"
    destination = "pizza_steak_sushi"
    
    # Download the dataset
    dataset_path = download_data(source=source_url, destination=destination)
    
    if dataset_path is None:
        print("[ERROR] Failed to download dataset. Please check your internet connection and try again.")
        return False
    
    # Verify the data structure
    if verify_data_structure(dataset_path):
        print("\n" + "=" * 60)
        print("Dataset setup completed successfully!")
        print(f"Dataset location: {dataset_path.absolute()}")
        print("You can now run the training script:")
        print("  cd PyTorch_Going_Modular/going_modular")
        print("  python train.py")
        print("=" * 60)
        return True
    else:
        print("[ERROR] Data verification failed. Please try running the script again.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)