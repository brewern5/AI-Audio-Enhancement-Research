"""
Data preparation pipeline for AI audio enhancement training.
Creates lossy/lossless audio pairs for training neural networks.
"""

import os
import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import argparse
from typing import Tuple, List
import logging
from concurrent.futures import ThreadPoolExecutor
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioDataPreparator:
    """Handles creation of lossy/lossless audio pairs for training."""
    
    def __init__(self, 
                 sample_rate: int = 22050,
                 duration: float = 3.0,
                 target_dir: str = "processed_data"):
        """
        Initialize the data preparator.
        
        Args:
            sample_rate: Target sample rate for audio processing
            duration: Duration of audio segments in seconds
            target_dir: Directory to save processed data
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.segment_length = int(sample_rate * duration)
        self.target_dir = Path(target_dir)
        self.target_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.target_dir / "lossless").mkdir(exist_ok=True)
        (self.target_dir / "lossy").mkdir(exist_ok=True)
        (self.target_dir / "metadata").mkdir(exist_ok=True)
    
    def create_lossy_audio(self, audio: np.ndarray, method: str = "mp3") -> np.ndarray:
        """
        Create lossy version of audio using various compression methods.
        
        Args:
            audio: Input audio array
            method: Compression method ('mp3', 'downsample', 'noise')
            
        Returns:
            Lossy audio array
        """
        if method == "mp3":
            # Simulate MP3 compression artifacts
            # This is a simplified approach - in practice, you'd use actual MP3 encoding
            # For now, we'll simulate by adding noise and slight frequency filtering
            noise_level = 0.001
            noise = np.random.normal(0, noise_level, audio.shape)
            lossy = audio + noise
            
            # Simulate frequency domain artifacts
            stft = librosa.stft(lossy)
            # Remove some high frequencies (simulate MP3 compression)
            stft[stft.shape[0]//2:, :] *= 0.7
            lossy = librosa.istft(stft)
            
        elif method == "downsample":
            # Downsample and upsample to simulate quality loss
            downsample_factor = 2
            downsampled = audio[::downsample_factor]
            # Upsample back to original rate
            lossy = np.repeat(downsampled, downsample_factor)
            # Truncate to original length
            lossy = lossy[:len(audio)]
            
        elif method == "noise":
            # Add noise to simulate transmission loss
            snr_db = 20  # Signal-to-noise ratio
            signal_power = np.mean(audio ** 2)
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
            lossy = audio + noise
            
        else:
            raise ValueError(f"Unknown compression method: {method}")
            
        return lossy.astype(np.float32)
    
    def process_audio_file(self, file_path: Path, output_id: int) -> dict:
        """
        Process a single audio file to create lossy/lossless pairs.
        
        Args:
            file_path: Path to input audio file
            output_id: Unique identifier for output files
            
        Returns:
            Metadata dictionary for the processed file
        """
        try:
            # Load audio
            audio, sr = librosa.load(str(file_path), sr=self.sample_rate)
            
            # Split into segments
            segments = []
            for i in range(0, len(audio) - self.segment_length + 1, self.segment_length // 2):
                segment = audio[i:i + self.segment_length]
                if len(segment) == self.segment_length:
                    segments.append(segment)
            
            metadata = {
                "original_file": str(file_path),
                "sample_rate": sr,
                "segments": []
            }
            
            # Process each segment
            for seg_idx, segment in enumerate(segments):
                # Create lossy versions using different methods
                for method in ["mp3", "downsample", "noise"]:
                    lossy_segment = self.create_lossy_audio(segment, method)
                    
                    # Save lossless segment
                    lossless_path = self.target_dir / "lossless" / f"{output_id}_{seg_idx}_{method}_lossless.wav"
                    sf.write(str(lossless_path), segment, self.sample_rate)
                    
                    # Save lossy segment
                    lossy_path = self.target_dir / "lossy" / f"{output_id}_{seg_idx}_{method}_lossy.wav"
                    sf.write(str(lossy_path), lossy_segment, self.sample_rate)
                    
                    # Store metadata
                    metadata["segments"].append({
                        "segment_idx": seg_idx,
                        "method": method,
                        "lossless_path": str(lossless_path),
                        "lossy_path": str(lossy_path),
                        "duration": self.duration
                    })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    def process_dataset(self, 
                       input_dir: str, 
                       max_files: int = None,
                       num_workers: int = 4) -> List[dict]:
        """
        Process entire dataset to create training pairs.
        
        Args:
            input_dir: Directory containing input audio files
            max_files: Maximum number of files to process (None for all)
            num_workers: Number of parallel workers
            
        Returns:
            List of metadata dictionaries
        """
        input_path = Path(input_dir)
        audio_files = list(input_path.glob("**/*.wav")) + list(input_path.glob("**/*.flac"))
        
        if max_files:
            audio_files = audio_files[:max_files]
        
        logger.info(f"Processing {len(audio_files)} audio files...")
        
        all_metadata = []
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i, file_path in enumerate(audio_files):
                future = executor.submit(self.process_audio_file, file_path, i)
                futures.append(future)
            
            for future in futures:
                metadata = future.result()
                if metadata:
                    all_metadata.append(metadata)
        
        # Save combined metadata
        metadata_path = self.target_dir / "metadata" / "dataset_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(all_metadata, f, indent=2)
        
        logger.info(f"Processed {len(all_metadata)} files successfully")
        logger.info(f"Metadata saved to {metadata_path}")
        
        return all_metadata

def main():
    """Main function for data preparation."""
    parser = argparse.ArgumentParser(description="Prepare audio data for training")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input audio files")
    parser.add_argument("--output_dir", type=str, default="processed_data",
                       help="Directory to save processed data")
    parser.add_argument("--sample_rate", type=int, default=22050,
                       help="Target sample rate")
    parser.add_argument("--duration", type=float, default=3.0,
                       help="Duration of audio segments in seconds")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to process")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers")
    
    args = parser.parse_args()
    
    # Create data preparator
    preparator = AudioDataPreparator(
        sample_rate=args.sample_rate,
        duration=args.duration,
        target_dir=args.output_dir
    )
    
    # Process dataset
    metadata = preparator.process_dataset(
        input_dir=args.input_dir,
        max_files=args.max_files,
        num_workers=args.workers
    )
    
    print(f"Data preparation complete! Processed {len(metadata)} files.")
    print(f"Output directory: {args.output_dir}")

if __name__ == "__main__":
    main()
