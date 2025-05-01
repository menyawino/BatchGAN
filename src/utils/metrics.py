"""
Evaluation metrics for assessing image quality and model performance.
Includes SSIM, PSNR, and other image quality metrics for histology batch correction.
"""
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, Tuple


def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Create a 2D Gaussian kernel for SSIM calculation.
    
    Args:
        size: Size of the Gaussian kernel
        sigma: Standard deviation of the Gaussian kernel
        
    Returns:
        2D Gaussian kernel
    """
    coords = torch.arange(size).to(dtype=torch.float)
    coords -= size // 2
    
    g = coords**2
    g = (-(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma**2)).exp()
    
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def calculate_ssim(
    img1: torch.Tensor, 
    img2: torch.Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float = 2.0,  # For normalized images in [-1, 1]
    size_average: bool = True
) -> float:
    """
    Calculate SSIM (Structural Similarity Index) between two image batches.
    
    Args:
        img1: First image batch (B, C, H, W)
        img2: Second image batch (B, C, H, W)
        window_size: Size of the Gaussian window
        sigma: Standard deviation of the Gaussian window
        data_range: Range of the input data (max - min)
        size_average: Whether to average over the batch
        
    Returns:
        SSIM value(s)
    """
    # Check input shapes
    if not img1.shape == img2.shape:
        raise ValueError(f"Input images must have the same shape, got {img1.shape} and {img2.shape}")
    
    # Create window
    kernel = _gaussian_kernel(window_size, sigma).to(img1.device, img1.dtype)
    window = kernel.expand(img1.size(1), 1, window_size, window_size)
    
    # Ensure the images have at least one batch dimension
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        
    # Parameters for SSIM calculation
    c1 = (0.01 * data_range)**2
    c2 = (0.03 * data_range)**2
    
    # Compute means
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=img1.size(1))
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=img1.size(1))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Compute sigma squares
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=img1.size(1)) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=img1.size(1)) - mu1_mu2
    
    # SSIM calculation
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    
    # Average SSIM scores across spatial dimensions, then across channels
    if size_average:
        return ssim_map.mean().item()
    else:
        # Return per-image SSIM scores (averaged over channels and spatial dimensions)
        return ssim_map.mean(dim=[1, 2, 3]).detach().cpu().numpy()


def calculate_psnr(
    img1: torch.Tensor, 
    img2: torch.Tensor,
    data_range: float = 2.0,  # For normalized images in [-1, 1]
    size_average: bool = True
) -> float:
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two image batches.
    
    Args:
        img1: First image batch (B, C, H, W)
        img2: Second image batch (B, C, H, W)
        data_range: Range of the input data (max - min)
        size_average: Whether to average over the batch
        
    Returns:
        PSNR value(s)
    """
    # Check input shapes
    if not img1.shape == img2.shape:
        raise ValueError(f"Input images must have the same shape, got {img1.shape} and {img2.shape}")
    
    # MSE (Mean Squared Error)
    mse = torch.mean((img1 - img2)**2, dim=[1, 2, 3])
    
    # PSNR calculation
    # PSNR = 10 * log10(MAX^2 / MSE)
    psnr = 10 * torch.log10((data_range**2) / (mse + 1e-8))
    
    if size_average:
        return psnr.mean().item()
    else:
        return psnr.detach().cpu().numpy()


def calculate_fid(real_features: torch.Tensor, fake_features: torch.Tensor) -> float:
    """
    Calculate FID (Fréchet Inception Distance) between real and fake image features.
    This is a simplified FID calculation that assumes features are already extracted.
    
    Args:
        real_features: Features of real images (N, D)
        fake_features: Features of fake images (N, D)
        
    Returns:
        FID score
    """
    # Move to CPU for numpy operations
    real_np = real_features.detach().cpu().numpy()
    fake_np = fake_features.detach().cpu().numpy()
    
    # Calculate mean and covariance
    mu1, sigma1 = real_np.mean(axis=0), np.cov(real_np, rowvar=False)
    mu2, sigma2 = fake_np.mean(axis=0), np.cov(fake_np, rowvar=False)
    
    # Calculate square root matrix
    # (This is a simplified version; real implementation should handle numerical issues)
    covmean = np.sqrt(sigma1 @ sigma2)
    
    # Calculate FID
    mu_diff = mu1 - mu2
    fid = mu_diff @ mu_diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    
    return float(fid)


def mse_loss(img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
    """
    Calculate Mean Squared Error loss between two images.
    
    Args:
        img1: First image (B, C, H, W)
        img2: Second image (B, C, H, W)
        
    Returns:
        MSE loss
    """
    return F.mse_loss(img1, img2)


def dice_coefficient(
    mask1: torch.Tensor, 
    mask2: torch.Tensor,
    smooth: float = 1.0
) -> torch.Tensor:
    """
    Calculate Dice coefficient for segmentation masks.
    
    Args:
        mask1: First binary mask (B, H, W)
        mask2: Second binary mask (B, H, W)
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient
    """
    # Flatten masks
    mask1_flat = mask1.view(mask1.size(0), -1)
    mask2_flat = mask2.view(mask2.size(0), -1)
    
    # Calculate intersection and union
    intersection = (mask1_flat * mask2_flat).sum(dim=1)
    union = mask1_flat.sum(dim=1) + mask2_flat.sum(dim=1)
    
    # Calculate Dice coefficient
    return ((2. * intersection + smooth) / (union + smooth)).mean()