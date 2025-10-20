import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchaudio
import pandas as pd
import numpy as np

# --- (Optional but Recommended) Lighter U-Net Model ---
# Using a model with fewer channels reduces memory usage significantly.
class LightUNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        # A simplified U-Net with fewer channels
        self.enc1 = nn.Conv1d(in_channels, 16, kernel_size=15, padding=7)
        self.pool1 = nn.MaxPool1d(2)
        self.enc2 = nn.Conv1d(16, 32, kernel_size=15, padding=7)
        self.pool2 = nn.MaxPool1d(2)
        
        self.bottleneck = nn.Conv1d(32, 64, kernel_size=15, padding=7)
        
        self.upconv2 = nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1)
        self.dec2 = nn.Conv1d(64, 32, kernel_size=5, padding=2) # 32 + 32
        self.upconv1 = nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1)
        self.dec1 = nn.Conv1d(32, 16, kernel_size=5, padding=2) # 16 + 16
        
        self.final_conv = nn.Conv1d(16, out_channels, kernel_size=1)
        self.final_tanh = nn.Tanh()

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)
        u2 = self.upconv2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.final_tanh(self.final_conv(d1))

# --- NEW Dataset that uses fixed-size chunks ---
class AudioChunkDataset(Dataset):
    def __init__(self, manifest_path, sample_rate=44100, chunk_duration_secs=2):
        self.manifest = pd.read_csv(manifest_path)
        self.sample_rate = sample_rate
        self.chunk_size = sample_rate * chunk_duration_secs

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        input_path = self.manifest.iloc[idx]['input_path']
        target_path = self.manifest.iloc[idx]['target_path']
        
        input_waveform, _ = torchaudio.load(input_path)
        target_waveform, _ = torchaudio.load(target_path)
        
        # Get a random chunk
        # If the file is shorter than the chunk size, it will be padded later.
        if input_waveform.shape[1] > self.chunk_size:
            start = np.random.randint(0, input_waveform.shape[1] - self.chunk_size)
            input_chunk = input_waveform[:, start:start + self.chunk_size]
            target_chunk = target_waveform[:, start:start + self.chunk_size]
        else:
            input_chunk = input_waveform
            target_chunk = target_waveform

        # Pad if the chunk (or original file) is shorter than the desired chunk size
        pad_len = self.chunk_size - input_chunk.shape[1]
        if pad_len > 0:
            input_chunk = torch.nn.functional.pad(input_chunk, (0, pad_len))
            target_chunk = torch.nn.functional.pad(target_chunk, (0, pad_len))
            
        return input_chunk, target_chunk

# --- 1. Setup and Hyperparameters ---
device = torch.device("cpu") # Forcing CPU
print(f"Using device: {device}")

# Hyperparameters optimized for low memory
NUM_EPOCHS = 25
LEARNING_RATE = 1e-4
BATCH_SIZE = 1 # Process one file at a time
ACCUMULATION_STEPS = 4 # Simulate a batch size of 1 * 4 = 4
MANIFEST_FILE = "../data/KaggleDirectories/dataset_manifest.csv"
MODEL_SAVE_PATH = "../data/KaggleDirectories/audio_upscaler_cpu_model.pth"

# --- 2. Initialize Components ---
# Use the new chunk-based dataset
audio_dataset = AudioChunkDataset(manifest_path=MANIFEST_FILE)
# No custom collate_fn needed! num_workers=0 is safer for low-memory.
train_loader = DataLoader(audio_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

# Use the lighter model
model = LightUNet(in_channels=2, out_channels=2).to(device)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Setup complete. Starting memory-efficient training...")

# --- 3. The UPDATED Training Loop with Gradient Accumulation ---
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    
    # Reset gradients at the start of the epoch
    optimizer.zero_grad()
    
    for i, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # Forward Pass
        outputs = model(inputs)
        
        # Crop output to match target length precisely
        min_len = min(outputs.shape[2], targets.shape[2])
        outputs = outputs[:, :, :min_len]
        targets = targets[:, :, :min_len]
        
        loss = criterion(outputs, targets)
        
        # Scale the loss for accumulation
        loss = loss / ACCUMULATION_STEPS
        
        # Backward Pass
        loss.backward()
        
        # --- Gradient Accumulation Step ---
        # Update weights only every ACCUMULATION_STEPS
        if (i + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()  # Update weights
            optimizer.zero_grad() # Reset gradients for the next accumulation cycle

        running_loss += loss.item() * ACCUMULATION_STEPS # Un-scale for logging

    # Print average loss for the epoch
    epoch_loss = running_loss / len(train_loader)
    print(f'--- End of Epoch [{epoch+1}/{NUM_EPOCHS}], Average Loss: {epoch_loss:.4f} ---')

print('Finished Training!')

# --- 4. Save the Trained Model ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to {MODEL_SAVE_PATH}")