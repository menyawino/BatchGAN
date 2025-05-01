"""
Dataset module for loading and processing valve histology images.
"""
import os
import random
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2


class ValveHistologyDataset(Dataset):
    """
    Dataset for unpaired valve histology images.
    For training CycleGAN-based batch correction models.
    """
    
    def __init__(
        self,
        healthy_dir: str,
        diseased_dir: str,
        img_size: int = 256,
        transform: Optional[Any] = None,
        augment: bool = True,
        max_samples: Optional[int] = None
    ):
        """
        Initialize the dataset.
        
        Args:
            healthy_dir: Directory containing healthy valve histology images
            diseased_dir: Directory containing diseased valve histology images
            img_size: Size to resize images to (square)
            transform: Custom transform to apply to images
            augment: Whether to apply data augmentation
            max_samples: Maximum number of samples to use (for debugging)
        """
        self.healthy_dir = healthy_dir
        self.diseased_dir = diseased_dir
        self.img_size = img_size
        self.custom_transform = transform
        self.augment = augment
        
        # Get image paths
        self.healthy_paths = self._get_image_paths(healthy_dir, max_samples)
        self.diseased_paths = self._get_image_paths(diseased_dir, max_samples)
        
        # Set up transforms
        self._setup_transforms()
        
    def _get_image_paths(self, directory: str, max_samples: Optional[int] = None) -> List[str]:
        """Get paths of all images in directory."""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
        
        # Get all image files
        image_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in valid_extensions:
                    image_paths.append(os.path.join(root, file))
        
        # Sort for reproducibility
        image_paths.sort()
        
        # Limit samples if specified
        if max_samples is not None:
            image_paths = image_paths[:max_samples]
            
        return image_paths
    
    def _setup_transforms(self):
        """Set up transformations for images."""
        # Base transform for both domains - resize and normalize
        self.base_transform = A.Compose([
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ToTensorV2()
        ])
        
        # Augmentation transform for training
        if self.augment:
            self.augment_transform = A.Compose([
                A.RandomRotate90(p=0.5),
                A.Flip(p=0.5),
                A.OneOf([
                    A.RandomBrightnessContrast(p=1.0),
                    A.HueSaturationValue(p=1.0),
                ], p=0.5),
                A.OneOf([
                    A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
                    A.GridDistortion(p=1.0),
                    A.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=1.0),
                ], p=0.25),
            ])
        else:
            self.augment_transform = None
    
    def __len__(self) -> int:
        """Get dataset size."""
        return max(len(self.healthy_paths), len(self.diseased_paths))
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a pair of images, one from each domain."""
        # Handle index wrapping for uneven datasets
        healthy_idx = idx % len(self.healthy_paths)
        diseased_idx = idx % len(self.diseased_paths)
        
        # Load images
        healthy_img = self._load_image(self.healthy_paths[healthy_idx])
        diseased_img = self._load_image(self.diseased_paths[diseased_idx])
        
        # Apply augmentations to both images
        if self.augment and self.augment_transform is not None:
            healthy_img = self.augment_transform(image=healthy_img)["image"]
            diseased_img = self.augment_transform(image=diseased_img)["image"]
        
        # Apply base transform (resize and normalize)
        healthy_tensor = self.base_transform(image=healthy_img)["image"]
        diseased_tensor = self.base_transform(image=diseased_img)["image"]
        
        # Apply any custom transforms
        if self.custom_transform is not None:
            healthy_tensor = self.custom_transform(healthy_tensor)
            diseased_tensor = self.custom_transform(diseased_tensor)
        
        return {
            "healthy": healthy_tensor,
            "diseased": diseased_tensor,
            "healthy_path": self.healthy_paths[healthy_idx],
            "diseased_path": self.diseased_paths[diseased_idx]
        }
    
    def _load_image(self, path: str) -> np.ndarray:
        """Load an image as a numpy array."""
        try:
            # Use PIL for better compatibility
            with Image.open(path) as img:
                img = img.convert('RGB')
                return np.array(img)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # Return a placeholder image
            return np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)


def get_data_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Dictionary with configuration options
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Extract parameters from config
    data_config = config.get("data", {})
    
    healthy_dir = data_config.get("healthy_dir")
    diseased_dir = data_config.get("diseased_dir")
    
    if not (healthy_dir and diseased_dir):
        raise ValueError("Data directories not specified in config")
        
    img_size = data_config.get("img_size", 256)
    batch_size = data_config.get("batch_size", 4)
    num_workers = data_config.get("num_workers", 4)
    
    # Check if we have separate directories for train/val/test
    if all(os.path.exists(data_config.get(f"{domain}_{split}_dir", "")) 
           for domain in ["healthy", "diseased"] 
           for split in ["train", "val", "test"]):
        # Use pre-split data
        train_dataset = ValveHistologyDataset(
            healthy_dir=data_config["healthy_train_dir"],
            diseased_dir=data_config["diseased_train_dir"],
            img_size=img_size,
            augment=True
        )
        
        val_dataset = ValveHistologyDataset(
            healthy_dir=data_config["healthy_val_dir"],
            diseased_dir=data_config["diseased_val_dir"],
            img_size=img_size,
            augment=False
        )
        
        test_dataset = ValveHistologyDataset(
            healthy_dir=data_config["healthy_test_dir"],
            diseased_dir=data_config["diseased_test_dir"],
            img_size=img_size,
            augment=False
        )
    else:
        # Split the data ourselves using random split
        full_dataset = ValveHistologyDataset(
            healthy_dir=healthy_dir,
            diseased_dir=diseased_dir,
            img_size=img_size,
            augment=False
        )
        
        # Get split ratios
        train_ratio = data_config.get("train_ratio", 0.7)
        val_ratio = data_config.get("val_ratio", 0.15)
        test_ratio = data_config.get("test_ratio", 0.15)
        
        # Ensure ratios sum to 1
        total = train_ratio + val_ratio + test_ratio
        train_ratio /= total
        val_ratio /= total
        test_ratio /= total
        
        # Calculate sizes
        dataset_size = len(full_dataset)
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        test_size = dataset_size - train_size - val_size
        
        # Create dataset splits
        train_dataset, val_dataset, test_dataset = random_split(
            full_dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Set augmentation for training set only
        train_dataset.dataset.augment = True
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader