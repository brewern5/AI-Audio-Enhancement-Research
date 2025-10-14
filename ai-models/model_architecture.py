"""
Lightweight U-Net architecture for audio enhancement optimized for edge deployment.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import math

class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution for reduced parameters."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
                 stride: int = 1, padding: int = 0, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, 
                                  stride, padding, dilation, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        
    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class LightweightBlock(nn.Module):
    """Lightweight residual block with depthwise separable convolutions."""
    
    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = DepthwiseSeparableConv1d(channels, channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class AudioUNet(nn.Module):
    """
    Lightweight U-Net architecture for audio enhancement.
    Optimized for edge deployment with reduced parameters and memory usage.
    """
    
    def __init__(self, 
                 input_channels: int = 1,
                 output_channels: int = 1,
                 base_channels: int = 32,
                 depth: int = 4,
                 dropout: float = 0.1):
        """
        Initialize the Audio U-Net.
        
        Args:
            input_channels: Number of input channels
            output_channels: Number of output channels  
            base_channels: Number of base channels (will be doubled at each level)
            depth: Number of encoder/decoder levels
            dropout: Dropout rate
        """
        super().__init__()
        self.depth = depth
        self.base_channels = base_channels
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_convs = nn.ModuleList()
        
        # Initial convolution
        self.initial_conv = nn.Conv1d(input_channels, base_channels, 7, padding=3)
        
        # Encoder layers
        for i in range(depth):
            in_ch = base_channels * (2 ** i)
            out_ch = base_channels * (2 ** (i + 1))
            
            # Encoder block
            encoder_block = nn.Sequential(
                LightweightBlock(in_ch, dropout=dropout),
                LightweightBlock(in_ch, dropout=dropout)
            )
            self.encoder_blocks.append(encoder_block)
            
            # Downsampling convolution
            if i < depth - 1:  # No downsampling after last encoder
                self.encoder_convs.append(
                    nn.Conv1d(in_ch, out_ch, 3, stride=2, padding=1)
                )
            else:
                self.encoder_convs.append(nn.Identity())
        
        # Bottleneck
        bottleneck_ch = base_channels * (2 ** depth)
        self.bottleneck = nn.Sequential(
            LightweightBlock(bottleneck_ch, dropout=dropout),
            LightweightBlock(bottleneck_ch, dropout=dropout)
        )
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_convs = nn.ModuleList()
        
        for i in range(depth):
            in_ch = base_channels * (2 ** (depth - i))
            out_ch = base_channels * (2 ** (depth - i - 1))
            
            # Upsampling convolution
            if i < depth - 1:
                self.decoder_convs.append(
                    nn.ConvTranspose1d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1)
                )
            else:
                self.decoder_convs.append(nn.Identity())
            
            # Decoder block
            decoder_block = nn.Sequential(
                LightweightBlock(out_ch, dropout=dropout),
                LightweightBlock(out_ch, dropout=dropout)
            )
            self.decoder_blocks.append(decoder_block)
        
        # Final convolution
        self.final_conv = nn.Conv1d(base_channels, output_channels, 7, padding=3)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.
        
        Args:
            x: Input tensor of shape (batch_size, channels, length)
            
        Returns:
            Enhanced audio tensor
        """
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder
        x = self.initial_conv(x)
        encoder_outputs.append(x)
        
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            encoder_outputs.append(x)
            
            if i < self.depth - 1:
                x = self.encoder_convs[i](x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        for i in range(self.depth):
            if i < self.depth - 1:
                x = self.decoder_convs[i](x)
            
            # Skip connection
            skip_idx = self.depth - i - 1
            if skip_idx < len(encoder_outputs):
                # Ensure dimensions match for skip connection
                if x.shape != encoder_outputs[skip_idx].shape:
                    # Resize to match
                    x = F.interpolate(x, size=encoder_outputs[skip_idx].shape[-1], 
                                    mode='linear', align_corners=False)
                x = x + encoder_outputs[skip_idx]
            
            x = self.decoder_blocks[i](x)
        
        # Final convolution
        x = self.final_conv(x)
        
        return x
    
    def get_model_size(self) -> dict:
        """Get model size information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate model size in MB (assuming float32)
        model_size_mb = total_params * 4 / (1024 * 1024)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": model_size_mb
        }

class SpectralLoss(nn.Module):
    """Spectral loss for audio enhancement training."""
    
    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute spectral loss between predicted and target audio.
        
        Args:
            pred: Predicted audio tensor
            target: Target audio tensor
            
        Returns:
            Spectral loss value
        """
        # Compute STFT
        pred_stft = torch.stft(pred.squeeze(1), self.n_fft, self.hop_length, 
                              return_complex=True)
        target_stft = torch.stft(target.squeeze(1), self.n_fft, self.hop_length, 
                                return_complex=True)
        
        # Magnitude loss
        pred_mag = torch.abs(pred_stft)
        target_mag = torch.abs(target_stft)
        mag_loss = F.mse_loss(pred_mag, target_mag)
        
        # Phase loss
        pred_phase = torch.angle(pred_stft)
        target_phase = torch.angle(target_stft)
        phase_loss = F.mse_loss(pred_phase, target_phase)
        
        return mag_loss + 0.1 * phase_loss

class CombinedLoss(nn.Module):
    """Combined loss function for audio enhancement."""
    
    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.mse_loss = nn.MSELoss()
        self.spectral_loss = SpectralLoss()
        self.l1_loss = nn.L1Loss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute combined loss.
        
        Args:
            pred: Predicted audio tensor
            target: Target audio tensor
            
        Returns:
            Combined loss value
        """
        mse = self.mse_loss(pred, target)
        spectral = self.spectral_loss(pred, target)
        l1 = self.l1_loss(pred, target)
        
        return self.alpha * mse + self.beta * spectral + self.gamma * l1

def create_model(config: dict) -> AudioUNet:
    """
    Create model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Initialized AudioUNet model
    """
    model = AudioUNet(
        input_channels=config.get('input_channels', 1),
        output_channels=config.get('output_channels', 1),
        base_channels=config.get('base_channels', 32),
        depth=config.get('depth', 4),
        dropout=config.get('dropout', 0.1)
    )
    
    return model

def test_model():
    """Test the model with sample input."""
    model = AudioUNet(base_channels=16, depth=3)
    
    # Test with sample input
    batch_size = 2
    length = 22050 * 3  # 3 seconds at 22kHz
    x = torch.randn(batch_size, 1, length)
    
    print(f"Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Output shape: {output.shape}")
    
    # Print model info
    model_info = model.get_model_size()
    print(f"Model parameters: {model_info['total_parameters']:,}")
    print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    return model

if __name__ == "__main__":
    # Test the model
    test_model()
