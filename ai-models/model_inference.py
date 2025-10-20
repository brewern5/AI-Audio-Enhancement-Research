import torch
import torchaudio
import numpy as np

# --- 1. Setup ---
device = torch.device("cpu")
MODEL_PATH = "../data/KaggleDirectories/audio_upscaler_cpu_model.pth"
# Use one of your original MP3 files as input
INPUT_AUDIO_PATH = "../data/KaggleDirectories/mp3_converted/a1.mp3" 
OUTPUT_AUDIO_PATH = "../data/KaggleDirectories/upscaled_output.wav"
SAMPLE_RATE = 44100
CHUNK_DURATION_SECS = 2 # Use the same chunk size as in training
CHUNK_SIZE = SAMPLE_RATE * CHUNK_DURATION_SECS

# --- 2. Load Model ---
# Make sure the LightUNet class is defined in a previous cell
model = LightUNet(in_channels=2, out_channels=2).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval() # Set the model to evaluation mode (very important!)

print("Model loaded. Starting inference...")

# --- 3. Load and Process Audio in Chunks ---
input_waveform, _ = torchaudio.load(INPUT_AUDIO_PATH)
input_waveform = input_waveform.to(device)
output_chunks = []

# Process the audio chunk by chunk to avoid memory errors
with torch.no_grad(): # Disable gradient calculation for efficiency
    for i in range(0, input_waveform.shape[1], CHUNK_SIZE):
        chunk = input_waveform[:, i:i + CHUNK_SIZE]
        
        # Pad the last chunk if it's smaller than the required size
        if chunk.shape[1] < CHUNK_SIZE:
            pad_len = CHUNK_SIZE - chunk.shape[1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_len))

        # Add a batch dimension and run through the model
        chunk = chunk.unsqueeze(0) # Shape: [1, num_channels, chunk_size]
        output_chunk = model(chunk)
        output_chunks.append(output_chunk.squeeze(0)) # Remove batch dimension

# --- 4. Stitch Chunks Together and Save ---
# Concatenate all the processed chunks
output_waveform = torch.cat(output_chunks, dim=1)

# Trim any excess padding from the end by matching the original input length
output_waveform = output_waveform[:, :input_waveform.shape[1]]

# Save the final upscaled audio
torchaudio.save(OUTPUT_AUDIO_PATH, output_waveform.cpu(), SAMPLE_RATE)

print(f"Inference complete! Upscaled audio saved to: {OUTPUT_AUDIO_PATH}")