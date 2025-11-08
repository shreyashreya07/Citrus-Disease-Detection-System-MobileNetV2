import os
import requests
from pathlib import Path

def create_folders():
    """Create dataset folder structure"""
    folders = [
        'dataset/train/blackspot',
        'dataset/train/canker',
        'dataset/train/greening',
        'dataset/train/healthy',
        'dataset/validation/blackspot',
        'dataset/validation/canker',
        'dataset/validation/greening',
        'dataset/validation/healthy'
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    print("✅ Folder structure created!")
    print("\nNow manually add images to these folders:")
    print("- Download citrus disease images from Google Images")
    print("- Or use Kaggle datasets")
    print("\nRecommended: 100+ images per class for good accuracy")

if __name__ == '__main__':
    create_folders()