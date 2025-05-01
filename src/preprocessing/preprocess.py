"""
Preprocessing pipeline for valve histology images.
Handles image loading, normalization, augmentation, and other preprocessing steps.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import shutil
import random

import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
import albumentations as A
from tqdm import tqdm


def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def standardize_histogram(img: np.ndarray, reference: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Standardize the color histogram of an image.
    
    Args:
        img: Input image as a numpy array (RGB)
        reference: Optional reference image for matching histograms
        
    Returns:
        Standardized image
    """
    # Convert images to LAB color space
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    
    # If reference image is provided, match histogram to that
    if reference is not None:
        ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)
        
        # Match each channel separately
        for i in range(3):  # L, A, B channels
            img_lab[:, :, i] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
                img_lab[:, :, i].astype(np.uint8)
            )
            
            # Histogram matching
            if i == 0:  # Only match lightness channel
                src_hist = cv2.calcHist([img_lab[:, :, i]], [0], None, [256], [0, 256])
                ref_hist = cv2.calcHist([ref_lab[:, :, i]], [0], None, [256], [0, 256])
                
                # Calculate cumulative distribution functions
                src_cdf = src_hist.cumsum()
                src_cdf_normalized = src_cdf / src_cdf[-1]
                
                ref_cdf = ref_hist.cumsum()
                ref_cdf_normalized = ref_cdf / ref_cdf[-1]
                
                # Create lookup table for histogram matching
                lookup_table = np.zeros(256, dtype=np.uint8)
                for j in range(256):
                    lookup_table[j] = np.argmin(np.abs(src_cdf_normalized[j] - ref_cdf_normalized))
                    
                img_lab[:, :, i] = cv2.LUT(img_lab[:, :, i], lookup_table)
    else:
        # Just apply CLAHE if no reference image
        for i in range(3):
            img_lab[:, :, i] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
                img_lab[:, :, i].astype(np.uint8)
            )
    
    # Convert back to RGB
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)


def normalize_staining(img: np.ndarray, target_stains: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Apply stain normalization to H&E stained histology images.
    Based on the method described in "Stain Normalization in Histopathology Images" by Macenko et al.
    
    Args:
        img: Input image as a numpy array (RGB)
        target_stains: Optional target staining matrix
        
    Returns:
        Stain normalized image
    """
    # This is a simplified implementation of the Macenko method
    # In a production system, consider using a dedicated library like:
    # - StainTools (https://github.com/Peter554/StainTools)
    # - TIAToolbox (https://github.com/TissueImageAnalytics/tiatoolbox)
    
    # Convert to optical density (OD) space
    OD = -np.log((img.astype(np.float32) + 1) / 256)
    
    # Remove pixels with zero OD
    mask = (OD > 0.15).all(axis=2)
    
    # Reshape to separate color channels
    OD_reshaped = OD[mask].reshape(-1, 3)
    
    if len(OD_reshaped) < 100:
        # Not enough pixels for robust stain separation
        return img
    
    # Perform singular value decomposition
    try:
        _, _, V = np.linalg.svd(OD_reshaped, full_matrices=False)
        
        # The first two singular vectors are our stain vectors
        stain_vectors = V[:2]
        
        # If the first component is not hematoxylin, swap components
        if stain_vectors[0][0] < stain_vectors[1][0]:
            stain_vectors = stain_vectors[[1, 0]]
            
        # Normalize stain vectors to unit length
        stain_vectors = stain_vectors / np.linalg.norm(stain_vectors, axis=1)[:, np.newaxis]
        
        # If target stains are provided, use those instead
        if target_stains is not None:
            stain_vectors = target_stains
            
        # Calculate stain concentrations
        stain_concentrations = np.linalg.lstsq(stain_vectors, OD_reshaped.T, rcond=None)[0]
        
        # Clip outliers
        stain_concentrations = np.clip(stain_concentrations, 0, np.percentile(stain_concentrations, 99))
        
        # Recreate the OD image
        OD_reshaped = (stain_concentrations.T @ stain_vectors).reshape(-1, 3)
        
        # Convert back to RGB space
        result = np.zeros_like(OD)
        result[mask] = 255 * np.exp(-OD_reshaped)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    except:
        logging.warning("Failed to perform stain normalization, returning original image")
        return img


def preprocess_image(
    img_path: str, 
    output_dir: str,
    target_size: Tuple[int, int] = (512, 512),
    normalize_stain: bool = True,
    standardize_hist: bool = True,
    reference_image: Optional[str] = None
) -> str:
    """
    Preprocess a single histology image for batch correction.
    
    Args:
        img_path: Path to the input image
        output_dir: Directory to save the preprocessed image
        target_size: Target size for resizing
        normalize_stain: Whether to apply stain normalization
        standardize_hist: Whether to standardize color histograms
        reference_image: Path to a reference image for histogram matching
        
    Returns:
        Path to the preprocessed image
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to load image: {img_path}")
        
    # Convert to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Load reference image if provided
    reference = None
    if reference_image and os.path.exists(reference_image):
        reference = cv2.imread(reference_image)
        if reference is not None:
            reference = cv2.cvtColor(reference, cv2.COLOR_BGR2RGB)
    
    # Apply preprocessing steps
    # 1. Stain normalization (for H&E stained histology)
    if normalize_stain:
        target_stains = None  # Could load standard stain vectors here for consistent normalization
        img = normalize_staining(img, target_stains)
    
    # 2. Standardize color histogram
    if standardize_hist:
        img = standardize_histogram(img, reference)
    
    # 3. Resize to target size
    img = cv2.resize(img, target_size)
    
    # Save the preprocessed image
    filename = os.path.basename(img_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    return output_path


def process_dataset(
    input_dir: str,
    output_dir: str,
    target_size: Tuple[int, int] = (512, 512),
    normalize_stain: bool = True,
    standardize_hist: bool = True,
    reference_image: Optional[str] = None
) -> int:
    """
    Process all images in a dataset.
    
    Args:
        input_dir: Directory containing the input images
        output_dir: Directory to save the preprocessed images
        target_size: Target size for resizing
        normalize_stain: Whether to apply stain normalization
        standardize_hist: Whether to standardize color histograms
        reference_image: Path to a reference image for histogram matching
        
    Returns:
        Number of images processed
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing dataset: {input_dir}")
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    image_files = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))
                
    logger.info(f"Found {len(image_files)} images to process")
    
    # Process each image
    processed_count = 0
    for img_path in tqdm(image_files, desc="Processing images"):
        try:
            # Keep the same relative path structure
            rel_path = os.path.relpath(img_path, input_dir)
            out_dir = os.path.join(output_dir, os.path.dirname(rel_path))
            os.makedirs(out_dir, exist_ok=True)
            
            # Process the image
            output_path = preprocess_image(
                img_path, 
                out_dir,
                target_size=target_size,
                normalize_stain=normalize_stain,
                standardize_hist=standardize_hist,
                reference_image=reference_image
            )
            
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing image {img_path}: {str(e)}")
            
    logger.info(f"Successfully processed {processed_count} images")
    return processed_count


def prepare_train_val_split(
    input_dir: str,
    output_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[str, str]:
    """
    Split a dataset into training and validation sets.
    
    Args:
        input_dir: Directory containing the input images
        output_dir: Base directory for the output
        train_ratio: Ratio of data to use for training
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dir, val_dir) paths
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Preparing train/val split with ratio {train_ratio}")
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    image_files = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    # Shuffle files
    random.shuffle(image_files)
    
    # Split into train and validation
    train_size = int(len(image_files) * train_ratio)
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]
    
    # Copy files to respective directories
    for files, target_dir in [(train_files, train_dir), (val_files, val_dir)]:
        for file_path in files:
            filename = os.path.basename(file_path)
            shutil.copy(file_path, os.path.join(target_dir, filename))
            
    logger.info(f"Split {len(image_files)} images into {len(train_files)} training and {len(val_files)} validation")
    
    return train_dir, val_dir


def main():
    """Main entry point for the preprocessing script."""
    parser = argparse.ArgumentParser(description="Preprocess histology images for batch correction")
    
    parser.add_argument("--healthy_dir", type=str, required=True, 
                        help="Directory with healthy valve histology images")
    parser.add_argument("--diseased_dir", type=str, required=True, 
                        help="Directory with diseased valve histology images")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory to save preprocessed images")
    parser.add_argument("--target_size", type=int, nargs=2, default=(512, 512), 
                        help="Target image size (width, height)")
    parser.add_argument("--normalize_stain", action="store_true", 
                        help="Apply stain normalization")
    parser.add_argument("--standardize_hist", action="store_true", 
                        help="Standardize color histograms")
    parser.add_argument("--reference_image", type=str, 
                        help="Reference image for histogram matching")
    parser.add_argument("--train_ratio", type=float, default=0.8, 
                        help="Ratio of data to use for training")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    parser.add_argument("--verbose", action="store_true", 
                        help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger.info("Starting preprocessing")
    
    # Process healthy samples
    healthy_processed_dir = os.path.join(args.output_dir, "processed", "healthy")
    process_dataset(
        args.healthy_dir, 
        healthy_processed_dir,
        target_size=tuple(args.target_size),
        normalize_stain=args.normalize_stain,
        standardize_hist=args.standardize_hist,
        reference_image=args.reference_image
    )
    
    # Process diseased samples
    diseased_processed_dir = os.path.join(args.output_dir, "processed", "diseased")
    process_dataset(
        args.diseased_dir, 
        diseased_processed_dir,
        target_size=tuple(args.target_size),
        normalize_stain=args.normalize_stain,
        standardize_hist=args.standardize_hist,
        reference_image=args.reference_image
    )
    
    # Create train/val splits
    logger.info("Creating train/val splits")
    # For healthy samples
    healthy_train_dir, healthy_val_dir = prepare_train_val_split(
        healthy_processed_dir,
        os.path.join(args.output_dir, "healthy"),
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    # For diseased samples
    diseased_train_dir, diseased_val_dir = prepare_train_val_split(
        diseased_processed_dir,
        os.path.join(args.output_dir, "diseased"),
        train_ratio=args.train_ratio,
        seed=args.seed
    )
    
    logger.info(f"Preprocessing complete. Data ready for training.")
    logger.info(f"Healthy train: {healthy_train_dir}")
    logger.info(f"Healthy val: {healthy_val_dir}")
    logger.info(f"Diseased train: {diseased_train_dir}")
    logger.info(f"Diseased val: {diseased_val_dir}")


if __name__ == "__main__":
    main()