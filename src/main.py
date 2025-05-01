"""
Main script for histology batch correction project.
This script provides command-line interfaces for training,
validation, and inference with the batch correction models.
"""
import os
import argparse
import logging
import sys
import yaml
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import numpy as np
import cv2
from tqdm import tqdm

from src.data.dataset import get_data_loaders
from src.models.trainer import CycleGANTrainer
from src.utils.metrics import calculate_metrics, generate_paper_figures


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


def train(config_path: str, resume_checkpoint: Optional[str] = None):
    """
    Train a batch correction model using the specified configuration.
    
    Args:
        config_path: Path to the YAML configuration file
        resume_checkpoint: Path to a checkpoint to resume training from
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with config: {config_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loaders
    logger.info("Creating data loaders")
    train_loader, val_loader, test_loader = get_data_loaders(config)
    
    # Create trainer
    trainer = CycleGANTrainer(config_path)
    
    # Resume from checkpoint if specified
    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        trainer.load_checkpoint(resume_checkpoint)
    
    # Train the model
    logger.info("Starting training")
    trainer.train(train_loader, val_loader)
    
    logger.info("Training complete")
    return trainer


def evaluate(config_path: str, checkpoint_path: str, output_dir: str = "results/evaluation"):
    """
    Evaluate a trained batch correction model.
    
    Args:
        config_path: Path to the YAML configuration file
        checkpoint_path: Path to the model checkpoint
        output_dir: Directory to save evaluation results
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating model from checkpoint: {checkpoint_path}")
    
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loaders (we'll use the test set)
    _, _, test_loader = get_data_loaders(config)
    
    # Create trainer and load checkpoint
    trainer = CycleGANTrainer(config_path)
    trainer.load_checkpoint(checkpoint_path)
    
    # Ensure the model is in evaluation mode
    trainer.model.eval()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect images and predictions for evaluation
    real_x_list, fake_y_list = [], []
    real_y_list, fake_x_list = [], []
    
    logger.info("Running inference on test set")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            real_x = batch["healthy"].to(trainer.device)
            real_y = batch["diseased"].to(trainer.device)
            
            # Forward pass
            outputs = trainer.model(real_x, real_y)
            
            # Collect images
            real_x_list.append(real_x.cpu())
            fake_y_list.append(outputs["fake_y"].cpu())
            real_y_list.append(real_y.cpu())
            fake_x_list.append(outputs["fake_x"].cpu())
    
    # Concatenate all batches
    real_x_all = torch.cat(real_x_list, dim=0)
    fake_y_all = torch.cat(fake_y_list, dim=0)
    real_y_all = torch.cat(real_y_list, dim=0)
    fake_x_all = torch.cat(fake_x_list, dim=0)
    
    # Calculate metrics
    logger.info("Calculating metrics")
    metrics = calculate_metrics(
        real_x_all, fake_y_all, real_y_all, fake_x_all,
        metrics=config["evaluation"]["metrics"]
    )
    
    # Log metrics
    logger.info("Evaluation metrics:")
    for name, value in metrics.items():
        logger.info(f"  {name}: {value}")
    
    # Save metrics to file
    metrics_path = output_dir / "metrics.yaml"
    with open(metrics_path, 'w') as f:
        yaml.dump(metrics, f)
        
    # Generate paper-ready figures
    logger.info("Generating publication-quality figures")
    figure_paths = generate_paper_figures(
        real_x_all, fake_y_all, real_y_all, fake_x_all,
        metrics, str(output_dir / "figures")
    )
    
    logger.info(f"Evaluation complete. Results saved to {output_dir}")
    logger.info(f"Generated figures: {figure_paths}")
    
    return metrics


def process_image(image_path: str, checkpoint_path: str, config_path: str,
                 direction: str = "diseased2healthy", output_dir: str = "results/predictions"):
    """
    Process a single image using a trained model.
    
    Args:
        image_path: Path to input image
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        direction: Direction of translation ('diseased2healthy' or 'healthy2diseased')
        output_dir: Directory to save processed images
    """
    logger = logging.getLogger(__name__)
    
    # Create trainer and load checkpoint
    trainer = CycleGANTrainer(config_path)
    trainer.load_checkpoint(checkpoint_path)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process the image
    logger.info(f"Processing image: {image_path}")
    output_img = trainer.predict(image_path, direction=direction)
    
    # Save the output
    filename = os.path.basename(image_path)
    base_name, ext = os.path.splitext(filename)
    output_path = os.path.join(output_dir, f"{base_name}_{direction}{ext}")
    
    # Convert from RGB to BGR for OpenCV
    output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, output_img_bgr)
    
    logger.info(f"Saved processed image to: {output_path}")
    return output_path


def process_directory(input_dir: str, checkpoint_path: str, config_path: str,
                     direction: str = "diseased2healthy", output_dir: str = "results/predictions"):
    """
    Process all images in a directory using a trained model.
    
    Args:
        input_dir: Directory containing input images
        checkpoint_path: Path to model checkpoint
        config_path: Path to configuration file
        direction: Direction of translation ('diseased2healthy' or 'healthy2diseased')
        output_dir: Directory to save processed images
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing all images in: {input_dir}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    image_files = []
    
    for root, _, files in os.walk(input_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Create trainer and load checkpoint
    trainer = CycleGANTrainer(config_path)
    trainer.load_checkpoint(checkpoint_path)
    
    # Process each image
    output_paths = []
    for image_path in tqdm(image_files, desc="Processing images"):
        try:
            # Process the image
            output_img = trainer.predict(image_path, direction=direction)
            
            # Save the output
            rel_path = os.path.relpath(image_path, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Convert from RGB to BGR for OpenCV
            output_img_bgr = cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, output_img_bgr)
            
            output_paths.append(output_path)
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {str(e)}")
    
    logger.info(f"Processed {len(output_paths)} images. Results saved to: {output_dir}")
    return output_paths


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Valve Histology Batch Correction Tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a batch correction model")
    train_parser.add_argument("--config", type=str, required=True,
                             help="Path to configuration file")
    train_parser.add_argument("--resume", type=str,
                             help="Path to checkpoint to resume training from")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a trained model")
    eval_parser.add_argument("--config", type=str, required=True,
                            help="Path to configuration file")
    eval_parser.add_argument("--checkpoint", type=str, required=True,
                            help="Path to model checkpoint")
    eval_parser.add_argument("--output_dir", type=str, default="results/evaluation",
                            help="Directory to save evaluation results")
    
    # Process single image command
    process_img_parser = subparsers.add_parser("process_image", 
                                              help="Process a single image")
    process_img_parser.add_argument("--image", type=str, required=True,
                                   help="Path to input image")
    process_img_parser.add_argument("--checkpoint", type=str, required=True,
                                   help="Path to model checkpoint")
    process_img_parser.add_argument("--config", type=str, required=True,
                                   help="Path to configuration file")
    process_img_parser.add_argument("--direction", type=str, 
                                   choices=["diseased2healthy", "healthy2diseased"],
                                   default="diseased2healthy",
                                   help="Direction of translation")
    process_img_parser.add_argument("--output_dir", type=str, 
                                   default="results/predictions",
                                   help="Directory to save processed images")
    
    # Process directory command
    process_dir_parser = subparsers.add_parser("process_directory", 
                                              help="Process all images in a directory")
    process_dir_parser.add_argument("--input_dir", type=str, required=True,
                                   help="Directory containing input images")
    process_dir_parser.add_argument("--checkpoint", type=str, required=True,
                                   help="Path to model checkpoint")
    process_dir_parser.add_argument("--config", type=str, required=True,
                                   help="Path to configuration file")
    process_dir_parser.add_argument("--direction", type=str, 
                                   choices=["diseased2healthy", "healthy2diseased"],
                                   default="diseased2healthy",
                                   help="Direction of translation")
    process_dir_parser.add_argument("--output_dir", type=str, 
                                   default="results/predictions",
                                   help="Directory to save processed images")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logging()
    
    # Execute the appropriate command
    if args.command == "train":
        train(args.config, args.resume)
    elif args.command == "evaluate":
        evaluate(args.config, args.checkpoint, args.output_dir)
    elif args.command == "process_image":
        process_image(args.image, args.checkpoint, args.config, 
                     args.direction, args.output_dir)
    elif args.command == "process_directory":
        process_directory(args.input_dir, args.checkpoint, args.config,
                         args.direction, args.output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()