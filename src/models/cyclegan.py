"""
CycleGAN model architecture for histology batch correction.
Includes generator and discriminator network implementations.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union


class ResidualBlock(nn.Module):
    """
    Residual block with instance normalization for the generator.
    """
    def __init__(self, n_filters: int, use_spectral_norm: bool = False):
        """
        Initialize the residual block.
        
        Args:
            n_filters: Number of filters in the convolutional layers
            use_spectral_norm: Whether to use spectral normalization
        """
        super(ResidualBlock, self).__init__()
        
        # First convolution layer with normalization and activation
        conv1 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        if use_spectral_norm:
            conv1 = nn.utils.spectral_norm(conv1)
        
        self.block1 = nn.Sequential(
            conv1,
            nn.InstanceNorm2d(n_filters),
            nn.ReLU(inplace=True)
        )
        
        # Second convolution layer with normalization (no activation)
        conv2 = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1)
        if use_spectral_norm:
            conv2 = nn.utils.spectral_norm(conv2)
            
        self.block2 = nn.Sequential(
            conv2,
            nn.InstanceNorm2d(n_filters)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block."""
        residual = x
        out = self.block1(x)
        out = self.block2(out)
        out += residual  # Skip connection
        return out


class AttentionBlock(nn.Module):
    """
    Self-attention block for the generator to capture long-range dependencies.
    Based on 'Self-Attention Generative Adversarial Networks' (SAGAN).
    """
    def __init__(self, n_filters: int):
        """
        Initialize the attention block.
        
        Args:
            n_filters: Number of input filters
        """
        super(AttentionBlock, self).__init__()
        
        # Learned query, key, and value projections
        self.query_conv = nn.Conv2d(n_filters, n_filters // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(n_filters, n_filters // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(n_filters, n_filters, kernel_size=1)
        
        # Scale factor for dot-product attention
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the attention block."""
        batch_size, C, height, width = x.size()
        
        # Project to queries, keys, and values
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)  # B x HW x C'
        key = self.key_conv(x).view(batch_size, -1, height * width)  # B x C' x HW
        value = self.value_conv(x).view(batch_size, -1, height * width)  # B x C x HW
        
        # Attention map
        attention = torch.bmm(query, key)  # B x HW x HW
        attention = F.softmax(attention, dim=-1)
        
        # Output
        out = torch.bmm(value, attention.permute(0, 2, 1))  # B x C x HW
        out = out.view(batch_size, C, height, width)
        
        # Apply attention with residual connection
        out = self.gamma * out + x
        
        return out


class Generator(nn.Module):
    """
    Generator network for CycleGAN, using a ResNet-based architecture.
    """
    def __init__(
        self, 
        input_channels: int = 3, 
        output_channels: int = 3,
        n_filters: int = 64,
        n_res_blocks: int = 9,
        use_attention: bool = True,
        use_spectral_norm: bool = False
    ):
        """
        Initialize the generator.
        
        Args:
            input_channels: Number of input image channels
            output_channels: Number of output image channels
            n_filters: Number of filters in the first and last layers
            n_res_blocks: Number of residual blocks
            use_attention: Whether to use self-attention blocks
            use_spectral_norm: Whether to use spectral normalization
        """
        super(Generator, self).__init__()
        
        # Initial convolution to increase channels
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, n_filters, kernel_size=7),
            nn.InstanceNorm2d(n_filters),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling layers
        for i in range(2):
            mult = 2**i
            in_channels = n_filters * mult
            out_channels = n_filters * mult * 2
            
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
                
            model += [
                conv,
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
        
        # Residual blocks
        mult = 4
        for i in range(n_res_blocks):
            model.append(ResidualBlock(n_filters * mult, use_spectral_norm))
            
            # Add attention blocks after specific residual blocks if enabled
            if use_attention and i == n_res_blocks // 2:
                model.append(AttentionBlock(n_filters * mult))
        
        # Upsampling layers
        for i in range(2):
            mult = 4 // (2**i)
            in_channels = n_filters * mult
            out_channels = n_filters * (mult // 2)
            
            model += [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ]
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(n_filters, output_channels, kernel_size=7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the generator."""
        return self.model(x)


class Discriminator(nn.Module):
    """
    PatchGAN discriminator for CycleGAN.
    Classifies patches of the image rather than the whole image.
    """
    def __init__(
        self, 
        input_channels: int = 3,
        n_filters: int = 64,
        n_layers: int = 4,
        use_spectral_norm: bool = True
    ):
        """
        Initialize the discriminator.
        
        Args:
            input_channels: Number of input image channels
            n_filters: Number of filters in the first layer
            n_layers: Number of convolutional layers
            use_spectral_norm: Whether to use spectral normalization
        """
        super(Discriminator, self).__init__()
        
        # Initial layer without normalization
        sequence = [
            nn.Conv2d(input_channels, n_filters, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Increasing channels while decreasing resolution
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            
            conv = nn.Conv2d(
                n_filters * nf_mult_prev,
                n_filters * nf_mult,
                kernel_size=4,
                stride=2 if n < n_layers-1 else 1,  # Last layer has stride 1
                padding=1
            )
            
            if use_spectral_norm:
                conv = nn.utils.spectral_norm(conv)
                
            sequence += [
                conv,
                nn.InstanceNorm2d(n_filters * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ]
        
        # Final layer to produce one-channel outputs per patch
        sequence += [
            nn.Conv2d(n_filters * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        ]
        
        self.model = nn.Sequential(*sequence)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator.
        
        Returns:
            Patch-based predictions (not averaged)
        """
        return self.model(x)


class CycleGAN(nn.Module):
    """
    Complete CycleGAN model with generators and discriminators.
    """
    def __init__(
        self, 
        input_channels: int = 3,
        output_channels: int = 3,
        n_filters_gen: int = 64,
        n_filters_disc: int = 64,
        n_res_blocks: int = 9,
        use_attention: bool = True,
        use_spectral_norm: bool = True
    ):
        """
        Initialize the CycleGAN model.
        
        Args:
            input_channels: Number of input image channels
            output_channels: Number of output image channels
            n_filters_gen: Number of filters in generator
            n_filters_disc: Number of filters in discriminator
            n_res_blocks: Number of residual blocks in generator
            use_attention: Whether to use attention in generator
            use_spectral_norm: Whether to use spectral normalization
        """
        super(CycleGAN, self).__init__()
        
        # Generators
        self.gen_XtoY = Generator(
            input_channels, output_channels, n_filters_gen, n_res_blocks,
            use_attention, use_spectral_norm
        )
        
        self.gen_YtoX = Generator(
            output_channels, input_channels, n_filters_gen, n_res_blocks,
            use_attention, use_spectral_norm
        )
        
        # Discriminators
        self.disc_X = Discriminator(input_channels, n_filters_disc, use_spectral_norm=use_spectral_norm)
        self.disc_Y = Discriminator(output_channels, n_filters_disc, use_spectral_norm=use_spectral_norm)
        
    def forward(
        self, 
        real_X: torch.Tensor, 
        real_Y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through both generator and discriminator paths.
        
        Args:
            real_X: Domain X real images (e.g., healthy)
            real_Y: Domain Y real images (e.g., diseased)
            
        Returns:
            Dictionary with all generated outputs
        """
        # Generator outputs
        fake_Y = self.gen_XtoY(real_X)  # G(X)
        rec_X = self.gen_YtoX(fake_Y)   # F(G(X)) ≈ X
        fake_X = self.gen_YtoX(real_Y)  # F(Y)
        rec_Y = self.gen_XtoY(fake_X)   # G(F(Y)) ≈ Y
        
        # Identity mapping
        idt_X = self.gen_YtoX(real_X)   # F(X) ≈ X
        idt_Y = self.gen_XtoY(real_Y)   # G(Y) ≈ Y
        
        return {
            "fake_Y": fake_Y,
            "rec_X": rec_X,
            "fake_X": fake_X,
            "rec_Y": rec_Y,
            "idt_X": idt_X,
            "idt_Y": idt_Y
        }
    
    def generate_Y(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate domain Y (diseased) from domain X (healthy).
        
        Args:
            x: Domain X image
            
        Returns:
            Domain Y image
        """
        return self.gen_XtoY(x)
    
    def generate_X(self, y: torch.Tensor) -> torch.Tensor:
        """
        Generate domain X (healthy) from domain Y (diseased).
        
        Args:
            y: Domain Y image
            
        Returns:
            Domain X image
        """
        return self.gen_YtoX(y)