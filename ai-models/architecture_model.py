import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """A single encoder block: Conv1D -> BatchNorm -> LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=15, stride=1, padding=7):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class UpConvBlock(nn.Module):
    """A single decoder block: Upsample -> Conv1D -> BatchNorm -> LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, padding=2):
        super().__init__()
        # Using ConvTranspose1d to upsample
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        return self.relu(self.bn(self.upconv(x)))

class UNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()

        # --- Encoder ---
        self.enc1 = ConvBlock(in_channels, 16)
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2) # Downsample
        self.enc2 = ConvBlock(16, 32)
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(32, 64)
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)

        # --- Bottleneck ---
        self.bottleneck = ConvBlock(64, 128)

        # --- Decoder ---
        # The in_channels for the decoder is double because of the skip connection concatenation
        self.upconv3 = UpConvBlock(128, 64)
        self.dec3 = ConvBlock(128, 64) # 64 from upconv + 64 from enc3 skip
        self.upconv2 = UpConvBlock(64, 32)
        self.dec2 = ConvBlock(64, 32)  # 32 from upconv + 32 from enc2 skip
        self.upconv1 = UpConvBlock(32, 16)
        self.dec1 = ConvBlock(32, 16)  # 16 from upconv + 16 from enc1 skip
        
        # --- Output Layer ---
        self.final_conv = nn.Conv1d(16, out_channels, kernel_size=1)
        self.final_tanh = nn.Tanh()

    def forward(self, x):
        # --- Encoder Path ---
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        # --- Bottleneck ---
        b = self.bottleneck(p3)

        # --- Decoder Path with Skip Connections ---
        u3 = self.upconv3(b)
        # Concatenate skip connection from encoder
        skip3 = torch.cat([u3, e3], dim=1)
        d3 = self.dec3(skip3)

        u2 = self.upconv2(d3)
        skip2 = torch.cat([u2, e2], dim=1)
        d2 = self.dec2(skip2)

        u1 = self.upconv1(d2)
        skip1 = torch.cat([u1, e1], dim=1)
        d1 = self.dec1(skip1)
        
        # --- Final Output ---
        out = self.final_conv(d1)
        
        return self.final_tanh(out)

# --- How to use it ---
# Assuming stereo audio (2 channels)
model = UNet(in_channels=2, out_channels=2)

# Create a dummy input batch to test the model
# (batch_size, num_channels, sequence_length)
dummy_input = torch.randn(4, 2, 44100 * 2) # Batch of 4, 2-second stereo clips at 44.1kHz

output = model(dummy_input)

print(f"Model created successfully!")
print(f"Input shape: {dummy_input.shape}")
print(f"Output shape: {output.shape}")
# Note: The output length might be slightly different due to convolutions.
# You may need to pad the input or crop the output to ensure they match exactly.