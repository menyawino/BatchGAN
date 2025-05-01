"""
Trainer module for CycleGAN-based valve histology batch correction models.
Handles model training, validation, checkpointing, and inference.
"""
import os
import sys
import time
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.models.cyclegan import CycleGAN
from src.utils.metrics import calculate_ssim, calculate_psnr
from PIL import Image


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


class ImageBuffer:
    """
    Buffer to store historical generated images to reduce model oscillation.
    Based on the paper: "Image-to-Image Translation with Conditional Adversarial Networks".
    """
    def __init__(self, buffer_size: int = 50):
        """
        Initialize the image buffer.
        
        Args:
            buffer_size: Maximum number of images to store in the buffer
        """
        self.buffer_size = buffer_size
        self.buffer = []
        
    def query(self, images: torch.Tensor) -> torch.Tensor:
        """
        Query the buffer and potentially return historical images mixed with current ones.
        
        Args:
            images: Current batch of generated images
            
        Returns:
            Images to use for discriminator training (mix of current and historical)
        """
        if self.buffer_size == 0:
            return images
            
        result = []
        for image in images:
            image = image.unsqueeze(0)  # Add batch dimension
            
            # If buffer is not full, add current image to buffer and return it
            if len(self.buffer) < self.buffer_size:
                self.buffer.append(image)
                result.append(image)
            else:
                # Random chance to return a historical image
                if np.random.random() < 0.5:
                    # Choose random image from buffer
                    idx = np.random.randint(0, self.buffer_size)
                    temp = self.buffer[idx].clone()
                    # Replace buffer image with current one
                    self.buffer[idx] = image
                    result.append(temp)
                else:
                    # Return current image
                    result.append(image)
                    
        return torch.cat(result, dim=0)


class CycleGANTrainer:
    """
    Trainer for CycleGAN-based valve histology batch correction.
    """
    def __init__(self, config_path: str):
        """
        Initialize the trainer with the specified configuration.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup model, optimizers, and losses
        self.setup()
        
    def setup(self):
        """Set up model, optimizers, losses, and other training components."""
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Using device: {self.device}")
        
        # Extract model configuration
        model_config = self.config.get("model", {})
        
        # Create model
        self.model = CycleGAN(
            input_channels=model_config.get("input_channels", 3),
            output_channels=model_config.get("output_channels", 3),
            n_filters_gen=model_config.get("n_filters_gen", 64),
            n_filters_disc=model_config.get("n_filters_disc", 64),
            n_res_blocks=model_config.get("n_res_blocks", 9),
            use_attention=model_config.get("use_attention", True),
            use_spectral_norm=model_config.get("use_spectral_norm", True)
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Image buffers for historical fake images
        self.fake_X_buffer = ImageBuffer(model_config.get("buffer_size", 50))
        self.fake_Y_buffer = ImageBuffer(model_config.get("buffer_size", 50))
        
        # Extract training configuration
        train_config = self.config.get("training", {})
        
        # Create optimizers
        self.optimizer_G = optim.Adam(
            list(self.model.gen_XtoY.parameters()) + 
            list(self.model.gen_YtoX.parameters()),
            lr=train_config.get("lr_generator", 0.0002),
            betas=(train_config.get("beta1", 0.5), train_config.get("beta2", 0.999))
        )
        
        self.optimizer_D = optim.Adam(
            list(self.model.disc_X.parameters()) + 
            list(self.model.disc_Y.parameters()),
            lr=train_config.get("lr_discriminator", 0.0001),
            betas=(train_config.get("beta1", 0.5), train_config.get("beta2", 0.999))
        )
        
        # Learning rate schedulers
        self.scheduler_G = optim.lr_scheduler.LambdaLR(
            self.optimizer_G,
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - train_config.get("n_epochs", 100)) / 
                                        float(train_config.get("n_epochs_decay", 100))
        )
        
        self.scheduler_D = optim.lr_scheduler.LambdaLR(
            self.optimizer_D,
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - train_config.get("n_epochs", 100)) / 
                                        float(train_config.get("n_epochs_decay", 100))
        )
        
        # Loss weights
        self.lambda_A = train_config.get("lambda_A", 10.0)  # Cycle consistency for X→Y→X
        self.lambda_B = train_config.get("lambda_B", 10.0)  # Cycle consistency for Y→X→Y
        self.lambda_idt = train_config.get("lambda_identity", 0.5)  # Identity loss weight
        
        # Loss functions
        self.criterion_GAN = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_ssim = 0.0
        
        # Output directories
        output_dir = self.config.get("output_dir", "results")
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "models"
        self.sample_dir = self.output_dir / "samples"
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        
    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: Optional[DataLoader] = None,
        num_epochs: Optional[int] = None
    ):
        """
        Train the CycleGAN model.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            num_epochs: Number of epochs to train (overrides config)
        """
        # Get number of epochs from config if not specified
        train_config = self.config.get("training", {})
        num_epochs = num_epochs or train_config.get("n_epochs", 200)
        
        # Track best model
        best_model_path = None
        best_ssim = 0.0
        
        # Start training
        self.logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_losses = self.train_epoch(train_loader)
            
            # Update learning rates
            self.scheduler_G.step()
            self.scheduler_D.step()
            
            # Log training progress
            self.logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                            f"G_loss: {train_losses['G_loss']:.4f}, "
                            f"D_loss: {train_losses['D_loss']:.4f}")
            
            # Validate if validation loader is provided
            if val_loader:
                val_metrics = self.validate(val_loader)
                
                # Log validation metrics
                self.logger.info(f"Validation - "
                                f"SSIM: {val_metrics['ssim']:.4f}, "
                                f"PSNR: {val_metrics['psnr']:.4f}")
                
                # Save best model based on SSIM
                if val_metrics['ssim'] > best_ssim:
                    best_ssim = val_metrics['ssim']
                    best_model_path = self.checkpoint_dir / f"best_model_epoch_{epoch+1}.pt"
                    self.save_checkpoint(best_model_path)
                    self.logger.info(f"Saved best model with SSIM: {best_ssim:.4f}")
            
            # Save sample images
            if (epoch + 1) % train_config.get("sample_interval", 5) == 0:
                self.save_samples(val_loader if val_loader else train_loader, epoch + 1)
            
            # Save regular checkpoint
            if (epoch + 1) % train_config.get("checkpoint_interval", 10) == 0:
                checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt"
                self.save_checkpoint(checkpoint_path)
                self.logger.info(f"Saved checkpoint at epoch {epoch+1}")
                
        # Save final model
        final_path = self.checkpoint_dir / "final_model.pt"
        self.save_checkpoint(final_path)
        self.logger.info(f"Training completed. Final model saved to {final_path}")
        
        # Return best model path if available
        return best_model_path if best_model_path else final_path
        
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for a single epoch.
        
        Args:
            train_loader: DataLoader for training data
            
        Returns:
            Dictionary with training losses
        """
        # Set model to train mode
        self.model.train()
        
        # Initialize losses
        epoch_losses = {
            "G_loss": 0.0,
            "D_loss": 0.0,
            "G_GAN_loss": 0.0,
            "G_cycle_loss": 0.0,
            "G_identity_loss": 0.0
        }
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}", leave=False)
        
        # Iterate through batches
        for batch_idx, batch in enumerate(pbar):
            # Get input data
            real_X = batch["healthy"].to(self.device)
            real_Y = batch["diseased"].to(self.device)
            
            # Train generators
            self.optimizer_G.zero_grad()
            
            # Forward pass
            outputs = self.model(real_X, real_Y)
            
            # Calculate generator losses
            # Identity loss
            idt_loss_X = self.criterion_identity(outputs["idt_X"], real_X) * self.lambda_A * self.lambda_idt
            idt_loss_Y = self.criterion_identity(outputs["idt_Y"], real_Y) * self.lambda_B * self.lambda_idt
            identity_loss = idt_loss_X + idt_loss_Y
            
            # GAN loss for generators
            # D_Y should classify G(X) as real
            gan_loss_X2Y = self.criterion_GAN(self.model.disc_Y(outputs["fake_Y"]), torch.ones_like(self.model.disc_Y(outputs["fake_Y"])))
            # D_X should classify F(Y) as real
            gan_loss_Y2X = self.criterion_GAN(self.model.disc_X(outputs["fake_X"]), torch.ones_like(self.model.disc_X(outputs["fake_X"])))
            gan_loss = gan_loss_X2Y + gan_loss_Y2X
            
            # Cycle consistency loss
            cycle_loss_X = self.criterion_cycle(outputs["rec_X"], real_X) * self.lambda_A
            cycle_loss_Y = self.criterion_cycle(outputs["rec_Y"], real_Y) * self.lambda_B
            cycle_loss = cycle_loss_X + cycle_loss_Y
            
            # Total generator loss
            G_loss = gan_loss + cycle_loss + identity_loss
            
            # Backpropagation and optimizer step
            G_loss.backward()
            self.optimizer_G.step()
            
            # Train discriminators
            self.optimizer_D.zero_grad()
            
            # Get fake images from buffer (to reduce model oscillation)
            fake_X = self.fake_X_buffer.query(outputs["fake_X"].detach())
            fake_Y = self.fake_Y_buffer.query(outputs["fake_Y"].detach())
            
            # Calculate discriminator losses
            # Real loss for D_X
            real_loss_DX = self.criterion_GAN(self.model.disc_X(real_X), torch.ones_like(self.model.disc_X(real_X)))
            # Fake loss for D_X
            fake_loss_DX = self.criterion_GAN(self.model.disc_X(fake_X), torch.zeros_like(self.model.disc_X(fake_X)))
            D_X_loss = (real_loss_DX + fake_loss_DX) * 0.5
            
            # Real loss for D_Y
            real_loss_DY = self.criterion_GAN(self.model.disc_Y(real_Y), torch.ones_like(self.model.disc_Y(real_Y)))
            # Fake loss for D_Y
            fake_loss_DY = self.criterion_GAN(self.model.disc_Y(fake_Y), torch.zeros_like(self.model.disc_Y(fake_Y)))
            D_Y_loss = (real_loss_DY + fake_loss_DY) * 0.5
            
            # Total discriminator loss
            D_loss = D_X_loss + D_Y_loss
            
            # Backpropagation and optimizer step
            D_loss.backward()
            self.optimizer_D.step()
            
            # Update losses for logging
            epoch_losses["G_loss"] += G_loss.item()
            epoch_losses["D_loss"] += D_loss.item()
            epoch_losses["G_GAN_loss"] += gan_loss.item()
            epoch_losses["G_cycle_loss"] += cycle_loss.item()
            epoch_losses["G_identity_loss"] += identity_loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                "G_loss": G_loss.item(),
                "D_loss": D_loss.item()
            })
            
            # Increment global step
            self.global_step += 1
        
        # Calculate average losses for the epoch
        num_batches = len(train_loader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses
            
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model on the validation set.
        
        Args:
            val_loader: DataLoader for validation data
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        # Initialize metrics
        metrics = {
            "ssim": 0.0,
            "psnr": 0.0,
            "cycle_loss": 0.0
        }
        
        # Calculate metrics without gradient calculation
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                # Get input data
                real_X = batch["healthy"].to(self.device)
                real_Y = batch["diseased"].to(self.device)
                
                # Forward pass
                outputs = self.model(real_X, real_Y)
                
                # Calculate SSIM and PSNR for both directions
                ssim_X2Y = calculate_ssim(real_X.cpu(), outputs["rec_X"].cpu())
                ssim_Y2X = calculate_ssim(real_Y.cpu(), outputs["rec_Y"].cpu())
                psnr_X2Y = calculate_psnr(real_X.cpu(), outputs["rec_X"].cpu())
                psnr_Y2X = calculate_psnr(real_Y.cpu(), outputs["rec_Y"].cpu())
                
                # Average metrics for both directions
                metrics["ssim"] += (ssim_X2Y + ssim_Y2X) / 2
                metrics["psnr"] += (psnr_X2Y + psnr_Y2X) / 2
                
                # Calculate cycle loss
                cycle_loss_X = self.criterion_cycle(outputs["rec_X"], real_X)
                cycle_loss_Y = self.criterion_cycle(outputs["rec_Y"], real_Y)
                metrics["cycle_loss"] += (cycle_loss_X + cycle_loss_Y).item() / 2
        
        # Calculate average metrics
        num_batches = len(val_loader)
        for key in metrics:
            metrics[key] /= num_batches
            
        return metrics
    
    def save_samples(self, data_loader: DataLoader, epoch: int):
        """
        Save sample images from the model for visualization.
        
        Args:
            data_loader: DataLoader to get sample images from
            epoch: Current epoch number
        """
        self.model.eval()
        
        # Get a batch of data
        batch = next(iter(data_loader))
        real_X = batch["healthy"].to(self.device)
        real_Y = batch["diseased"].to(self.device)
        
        # Generate outputs
        with torch.no_grad():
            outputs = self.model(real_X, real_Y)
        
        # Get a few sample images (at most 4)
        n_samples = min(4, real_X.size(0))
        
        # Create figure
        fig, axes = plt.subplots(n_samples, 4, figsize=(16, 4*n_samples))
        
        # Helper function to convert tensor to numpy image
        def tensor_to_image(tensor):
            image = tensor.clone().detach().cpu().numpy()
            image = image.transpose(1, 2, 0)  # C,H,W to H,W,C
            image = (image * 0.5 + 0.5) * 255.0  # Denormalize from [-1,1] to [0,255]
            return image.astype(np.uint8)
        
        # Plot images
        for i in range(n_samples):
            # For single sample case
            if n_samples == 1:
                ax_row = axes
            else:
                ax_row = axes[i]
                
            images = [
                tensor_to_image(real_X[i]),  # Real healthy
                tensor_to_image(outputs["fake_Y"][i]),  # Fake diseased
                tensor_to_image(real_Y[i]),  # Real diseased
                tensor_to_image(outputs["fake_X"][i])   # Fake healthy
            ]
            
            titles = ["Real Healthy", "Generated Diseased", "Real Diseased", "Generated Healthy"]
            
            for j, (image, title) in enumerate(zip(images, titles)):
                ax_row[j].imshow(image)
                ax_row[j].set_title(title)
                ax_row[j].axis('off')
        
        # Save figure
        sample_path = self.sample_dir / f"epoch_{epoch}.png"
        plt.tight_layout()
        plt.savefig(sample_path)
        plt.close(fig)
        
        self.logger.info(f"Saved sample images to {sample_path}")
        
    def save_checkpoint(self, path: Union[str, Path]):
        """
        Save a model checkpoint.
        
        Args:
            path: Path to save the checkpoint
        """
        checkpoint = {
            'gen_XtoY_state_dict': self.model.gen_XtoY.state_dict(),
            'gen_YtoX_state_dict': self.model.gen_YtoX.state_dict(),
            'disc_X_state_dict': self.model.disc_X.state_dict(),
            'disc_Y_state_dict': self.model.disc_Y.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_state_dict': self.optimizer_D.state_dict(),
            'scheduler_G_state_dict': self.scheduler_G.state_dict(),
            'scheduler_D_state_dict': self.scheduler_D.state_dict(),
            'current_epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_ssim': self.best_ssim,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: Union[str, Path]):
        """
        Load a model checkpoint.
        
        Args:
            path: Path to the checkpoint
        """
        self.logger.info(f"Loading checkpoint from {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        
        # Load model states
        self.model.gen_XtoY.load_state_dict(checkpoint['gen_XtoY_state_dict'])
        self.model.gen_YtoX.load_state_dict(checkpoint['gen_YtoX_state_dict'])
        self.model.disc_X.load_state_dict(checkpoint['disc_X_state_dict'])
        self.model.disc_Y.load_state_dict(checkpoint['disc_Y_state_dict'])
        
        # Load optimizer states if available and we're not in inference mode
        if 'optimizer_G_state_dict' in checkpoint and hasattr(self, 'optimizer_G'):
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            self.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            self.scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
            self.scheduler_D.load_state_dict(checkpoint['scheduler_D_state_dict'])
            
        # Load training state
        self.current_epoch = checkpoint.get('current_epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_ssim = checkpoint.get('best_ssim', 0.0)
        
        self.logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
        
    def predict(self, image_path: Union[str, Path], direction: str = "diseased2healthy") -> np.ndarray:
        """
        Generate a prediction for a single image.
        
        Args:
            image_path: Path to the input image
            direction: Direction of translation ('diseased2healthy' or 'healthy2diseased')
            
        Returns:
            Processed image as a numpy array
        """
        self.model.eval()
        
        # Determine direction
        if direction not in ["diseased2healthy", "healthy2diseased"]:
            raise ValueError("Direction must be 'diseased2healthy' or 'healthy2diseased'")
            
        # Load and preprocess image
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input size
        img_size = self.config.get("data", {}).get("img_size", 256)
        img = cv2.resize(img, (img_size, img_size))
        
        # Normalize to [-1, 1]
        img = img.astype(np.float32) / 127.5 - 1.0
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        
        # Process image
        with torch.no_grad():
            if direction == "diseased2healthy":
                output = self.model.generate_X(img_tensor)
            else:
                output = self.model.generate_Y(img_tensor)
        
        # Convert output tensor to numpy array
        output_img = output.squeeze().cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize to [0, 255]
        output_img = ((output_img * 0.5 + 0.5) * 255.0).astype(np.uint8)
        
        return output_img