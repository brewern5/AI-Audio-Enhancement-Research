import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F # We need this for the padding function

# The Dataset class can remain the same as before
class AudioUpscalingDataset(Dataset):
    def __init__(self, manifest_path, target_sample_rate=44100):
        self.manifest = pd.read_csv(manifest_path)
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        input_path = self.manifest.iloc[idx]['input_path']
        target_path = self.manifest.iloc[idx]['target_path']
        
        try:
            input_waveform, orig_sr_input = torchaudio.load(input_path)
            target_waveform, orig_sr_target = torchaudio.load(target_path)
        except Exception as e:
            print(f"Error loading files at index {idx}: {e}")
            return torch.zeros(1, 1), torch.zeros(1, 1) # Return minimal tensor on error

        if orig_sr_input != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr_input, self.target_sample_rate)
            input_waveform = resampler(input_waveform)
        
        if orig_sr_target != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr_target, self.target_sample_rate)
            target_waveform = resampler(target_waveform)

        # We no longer need to manually fix lengths here. The collate_fn will handle it.
        return input_waveform, target_waveform

# --- NEW: The Custom Collate Function ---
def pad_collate_fn(batch):
    """
    Pads audio samples in a batch to the length of the longest sample.
    Args:
        batch: A list of tuples, where each tuple is (input_waveform, target_waveform).
    """
    # Separate the inputs and targets
    inputs, targets = zip(*batch)

    # Find the maximum length in the batch for inputs
    max_input_len = max(w.shape[1] for w in inputs)
    # Find the maximum length in the batch for targets
    max_target_len = max(w.shape[1] for w in targets)
    # Use the overall max length to be safe
    max_len = max(max_input_len, max_target_len)
    
    # Pad all inputs to the max_len
    # `pad` arguments are (left, right, top, bottom) for 2D tensors
    padded_inputs = torch.stack([
        F.pad(w, (0, max_len - w.shape[1])) for w in inputs
    ])

    # Pad all targets to the max_len
    padded_targets = torch.stack([
        F.pad(w, (0, max_len - w.shape[1])) for w in targets
    ])

    return padded_inputs, padded_targets

# --- UPDATED: Using the DataLoader ---

# 1. Define the path to your manifest file
manifest_file = "../data/KaggleDirectories/dataset_manifest.csv"

# 2. Create an instance of your custom Dataset
audio_dataset = AudioUpscalingDataset(manifest_path=manifest_file, target_sample_rate=44100)

# 3. Create the DataLoader, NOW WITH THE CUSTOM COLLATE FUNCTION
batch_size = 4
train_loader = DataLoader(
    audio_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    collate_fn=pad_collate_fn  # <-- This is the crucial change!
)

# 4. Now, this loop will work without errors
print("Running DataLoader with padding collate function...")
for i, (inputs, targets) in enumerate(train_loader):
    print(f"Batch {i+1}:")
    print(f"  Input batch shape: {inputs.shape}")
    print(f"  Target batch shape: {targets.shape}")
    # All tensors in a batch will now have the same length.
    break