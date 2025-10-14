"""
Training pipeline for audio enhancement model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple, Optional
import argparse
from tqdm import tqdm
import wandb
from sklearn.model_selection import train_test_split

from model_architecture import AudioUNet, CombinedLoss, create_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    """Dataset class for audio enhancement training."""
    
    def __init__(self, 
                 metadata_path: str,
                 sample_rate: int = 22050,
                 duration: float = 3.0,
                 augment: bool = True):
        """
        Initialize the dataset.
        
        Args:
            metadata_path: Path to metadata JSON file
            sample_rate: Target sample rate
            duration: Duration of audio segments
            augment: Whether to apply data augmentation
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.segment_length = int(sample_rate * duration)
        self.augment = augment
        
        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Flatten segments
        self.segments = []
        for file_meta in self.metadata:
            for segment in file_meta['segments']:
                self.segments.append(segment)
        
        logger.info(f"Loaded {len(self.segments)} audio segments")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        """Get a training sample."""
        segment = self.segments[idx]
        
        # Load lossy and lossless audio
        lossy_audio, _ = sf.read(segment['lossy_path'])
        lossless_audio, _ = sf.read(segment['lossless_path'])
        
        # Ensure same length
        min_length = min(len(lossy_audio), len(lossless_audio))
        lossy_audio = lossy_audio[:min_length]
        lossless_audio = lossless_audio[:min_length]
        
        # Apply augmentation if enabled
        if self.augment:
            lossy_audio, lossless_audio = self._augment_audio(lossy_audio, lossless_audio)
        
        # Convert to tensors
        lossy_tensor = torch.FloatTensor(lossy_audio).unsqueeze(0)  # Add channel dimension
        lossless_tensor = torch.FloatTensor(lossless_audio).unsqueeze(0)
        
        return lossy_tensor, lossless_tensor
    
    def _augment_audio(self, lossy: np.ndarray, lossless: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply data augmentation to audio pair."""
        # Random gain
        gain_factor = np.random.uniform(0.8, 1.2)
        lossy = lossy * gain_factor
        lossless = lossless * gain_factor
        
        # Random noise (only to lossy to simulate real-world conditions)
        if np.random.random() < 0.3:
            noise_level = np.random.uniform(0.001, 0.005)
            noise = np.random.normal(0, noise_level, lossy.shape)
            lossy = lossy + noise
        
        # Random time stretching (slight)
        if np.random.random() < 0.2:
            stretch_factor = np.random.uniform(0.95, 1.05)
            lossy = librosa.effects.time_stretch(lossy, rate=stretch_factor)
            lossless = librosa.effects.time_stretch(lossless, rate=stretch_factor)
        
        return lossy, lossless

class AudioTrainer:
    """Trainer class for audio enhancement model."""
    
    def __init__(self, 
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 use_wandb: bool = False):
        """
        Initialize the trainer.
        
        Args:
            model: The neural network model
            train_loader: Training data loader
            val_loader: Validation data loader
            device: Device to run training on
            use_wandb: Whether to use Weights & Biases logging
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_wandb = use_wandb
        
        # Initialize loss and optimizer
        self.criterion = CombinedLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        if use_wandb:
            wandb.init(project="audio-enhancement", config={
                "model": "AudioUNet",
                "learning_rate": 1e-3,
                "batch_size": train_loader.batch_size,
                "epochs": 100
            })
    
    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, (lossy, lossless) in enumerate(pbar):
            lossy = lossy.to(self.device)
            lossless = lossless.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            enhanced = self.model(lossy)
            loss = self.criterion(enhanced, lossless)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    "train_loss": loss.item(),
                    "epoch": self.current_epoch,
                    "batch": batch_idx
                })
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for lossy, lossless in tqdm(self.val_loader, desc="Validation"):
                lossy = lossy.to(self.device)
                lossless = lossless.to(self.device)
                
                enhanced = self.model(lossy)
                loss = self.criterion(enhanced, lossless)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pth', '_best.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")
    
    def train(self, num_epochs: int, save_dir: str = "checkpoints"):
        """Train the model."""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Log results
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            if self.use_wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": self.optimizer.param_groups[0]['lr']
                })
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            checkpoint_path = save_path / f"checkpoint_epoch_{epoch}.pth"
            self.save_checkpoint(str(checkpoint_path), is_best)
            
            # Early stopping
            if epoch > 10 and val_loss > self.best_val_loss * 1.1:
                logger.info("Early stopping triggered")
                break
        
        logger.info("Training completed!")

def create_data_loaders(metadata_path: str,
                       batch_size: int = 16,
                       train_split: float = 0.8,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation data loaders."""
    
    # Load metadata to split
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Split files
    train_files, val_files = train_test_split(
        metadata, train_size=train_split, random_state=42
    )
    
    # Save split metadata
    with open("train_metadata.json", 'w') as f:
        json.dump(train_files, f)
    with open("val_metadata.json", 'w') as f:
        json.dump(val_files, f)
    
    # Create datasets
    train_dataset = AudioDataset("train_metadata.json", augment=True)
    val_dataset = AudioDataset("val_metadata.json", augment=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train audio enhancement model")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing processed data")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--base_channels", type=int, default=32,
                       help="Base number of channels in model")
    parser.add_argument("--depth", type=int, default=4,
                       help="Model depth")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--save_dir", type=str, default="checkpoints",
                       help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases logging")
    
    args = parser.parse_args()
    
    # Create data loaders
    metadata_path = Path(args.data_dir) / "metadata" / "dataset_metadata.json"
    train_loader, val_loader = create_data_loaders(
        str(metadata_path), 
        batch_size=args.batch_size
    )
    
    # Create model
    config = {
        'base_channels': args.base_channels,
        'depth': args.depth,
        'dropout': 0.1
    }
    model = create_model(config)
    
    # Print model info
    model_info = model.get_model_size()
    logger.info(f"Model parameters: {model_info['total_parameters']:,}")
    logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Create trainer
    trainer = AudioTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        use_wandb=args.use_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train model
    trainer.train(num_epochs=args.epochs, save_dir=args.save_dir)

if __name__ == "__main__":
    main()
