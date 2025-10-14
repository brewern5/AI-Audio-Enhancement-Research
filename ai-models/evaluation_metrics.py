"""
Evaluation metrics for audio enhancement models.
Includes spectral distance metrics, perceptual quality measures, and audio quality assessment.
"""

import torch
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr, spearmanr
import json

logger = logging.getLogger(__name__)

class SpectralMetrics:
    """Spectral distance metrics for audio quality assessment."""
    
    def __init__(self, sample_rate: int = 22050, n_fft: int = 1024, hop_length: int = 256):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def spectral_magnitude_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute spectral magnitude distance."""
        pred_stft = librosa.stft(pred, n_fft=self.n_fft, hop_length=self.hop_length)
        target_stft = librosa.stft(target, n_fft=self.n_fft, hop_length=self.hop_length)
        
        pred_mag = np.abs(pred_stft)
        target_mag = np.abs(target_stft)
        
        # Ensure same shape
        min_freq = min(pred_mag.shape[0], target_mag.shape[0])
        min_time = min(pred_mag.shape[1], target_mag.shape[1])
        
        pred_mag = pred_mag[:min_freq, :min_time]
        target_mag = target_mag[:min_freq, :min_time]
        
        return np.mean(np.abs(pred_mag - target_mag))
    
    def spectral_phase_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute spectral phase distance."""
        pred_stft = librosa.stft(pred, n_fft=self.n_fft, hop_length=self.hop_length)
        target_stft = librosa.stft(target, n_fft=self.n_fft, hop_length=self.hop_length)
        
        pred_phase = np.angle(pred_stft)
        target_phase = np.angle(target_stft)
        
        # Ensure same shape
        min_freq = min(pred_phase.shape[0], target_phase.shape[0])
        min_time = min(pred_phase.shape[1], target_phase.shape[1])
        
        pred_phase = pred_phase[:min_freq, :min_time]
        target_phase = target_phase[:min_freq, :min_time]
        
        # Phase difference
        phase_diff = np.abs(pred_phase - target_phase)
        # Handle phase wrapping
        phase_diff = np.minimum(phase_diff, 2 * np.pi - phase_diff)
        
        return np.mean(phase_diff)
    
    def log_spectral_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute log spectral distance."""
        pred_stft = librosa.stft(pred, n_fft=self.n_fft, hop_length=self.hop_length)
        target_stft = librosa.stft(target, n_fft=self.n_fft, hop_length=self.hop_length)
        
        pred_mag = np.abs(pred_stft)
        target_mag = np.abs(target_stft)
        
        # Ensure same shape
        min_freq = min(pred_mag.shape[0], target_mag.shape[0])
        min_time = min(pred_mag.shape[1], target_mag.shape[1])
        
        pred_mag = pred_mag[:min_freq, :min_time]
        target_mag = target_mag[:min_freq, :min_time]
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        pred_log = np.log(pred_mag + epsilon)
        target_log = np.log(target_mag + epsilon)
        
        lsd = np.mean(np.sqrt(np.mean((pred_log - target_log) ** 2, axis=0)))
        return lsd
    
    def spectral_centroid_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute spectral centroid distance."""
        pred_centroid = librosa.feature.spectral_centroid(y=pred, sr=self.sample_rate)[0]
        target_centroid = librosa.feature.spectral_centroid(y=target, sr=self.sample_rate)[0]
        
        # Ensure same length
        min_len = min(len(pred_centroid), len(target_centroid))
        pred_centroid = pred_centroid[:min_len]
        target_centroid = target_centroid[:min_len]
        
        return np.mean(np.abs(pred_centroid - target_centroid))
    
    def spectral_rolloff_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute spectral rolloff distance."""
        pred_rolloff = librosa.feature.spectral_rolloff(y=pred, sr=self.sample_rate)[0]
        target_rolloff = librosa.feature.spectral_rolloff(y=target, sr=self.sample_rate)[0]
        
        # Ensure same length
        min_len = min(len(pred_rolloff), len(target_rolloff))
        pred_rolloff = pred_rolloff[:min_len]
        target_rolloff = target_rolloff[:min_len]
        
        return np.mean(np.abs(pred_rolloff - target_rolloff))

class PerceptualMetrics:
    """Perceptual quality metrics for audio assessment."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def mfcc_distance(self, pred: np.ndarray, target: np.ndarray, n_mfcc: int = 13) -> float:
        """Compute MFCC distance."""
        pred_mfcc = librosa.feature.mfcc(y=pred, sr=self.sample_rate, n_mfcc=n_mfcc)
        target_mfcc = librosa.feature.mfcc(y=target, sr=self.sample_rate, n_mfcc=n_mfcc)
        
        # Ensure same shape
        min_freq = min(pred_mfcc.shape[0], target_mfcc.shape[0])
        min_time = min(pred_mfcc.shape[1], target_mfcc.shape[1])
        
        pred_mfcc = pred_mfcc[:min_freq, :min_time]
        target_mfcc = target_mfcc[:min_freq, :min_time]
        
        return np.mean(np.abs(pred_mfcc - target_mfcc))
    
    def chroma_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute chroma distance."""
        pred_chroma = librosa.feature.chroma_stft(y=pred, sr=self.sample_rate)
        target_chroma = librosa.feature.chroma_stft(y=target, sr=self.sample_rate)
        
        # Ensure same shape
        min_freq = min(pred_chroma.shape[0], target_chroma.shape[0])
        min_time = min(pred_chroma.shape[1], target_chroma.shape[1])
        
        pred_chroma = pred_chroma[:min_freq, :min_time]
        target_chroma = target_chroma[:min_freq, :min_time]
        
        return np.mean(np.abs(pred_chroma - target_chroma))
    
    def tonnetz_distance(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute tonnetz distance."""
        pred_tonnetz = librosa.feature.tonnetz(y=pred, sr=self.sample_rate)
        target_tonnetz = librosa.feature.tonnetz(y=target, sr=self.sample_rate)
        
        # Ensure same shape
        min_freq = min(pred_tonnetz.shape[0], target_tonnetz.shape[0])
        min_time = min(pred_tonnetz.shape[1], target_tonnetz.shape[1])
        
        pred_tonnetz = pred_tonnetz[:min_freq, :min_time]
        target_tonnetz = target_tonnetz[:min_freq, :min_time]
        
        return np.mean(np.abs(pred_tonnetz - target_tonnetz))

class AudioQualityEvaluator:
    """Comprehensive audio quality evaluation."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.spectral_metrics = SpectralMetrics(sample_rate)
        self.perceptual_metrics = PerceptualMetrics(sample_rate)
    
    def evaluate_audio_pair(self, pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """Evaluate a single audio pair."""
        # Ensure same length
        min_len = min(len(pred), len(target))
        pred = pred[:min_len]
        target = target[:min_len]
        
        results = {}
        
        # Spectral metrics
        results['spectral_magnitude_distance'] = self.spectral_metrics.spectral_magnitude_distance(pred, target)
        results['spectral_phase_distance'] = self.spectral_metrics.spectral_phase_distance(pred, target)
        results['log_spectral_distance'] = self.spectral_metrics.log_spectral_distance(pred, target)
        results['spectral_centroid_distance'] = self.spectral_metrics.spectral_centroid_distance(pred, target)
        results['spectral_rolloff_distance'] = self.spectral_metrics.spectral_rolloff_distance(pred, target)
        
        # Perceptual metrics
        results['mfcc_distance'] = self.perceptual_metrics.mfcc_distance(pred, target)
        results['chroma_distance'] = self.perceptual_metrics.chroma_distance(pred, target)
        results['tonnetz_distance'] = self.perceptual_metrics.tonnetz_distance(pred, target)
        
        # Basic metrics
        results['mse'] = np.mean((pred - target) ** 2)
        results['mae'] = np.mean(np.abs(pred - target))
        results['snr'] = self._compute_snr(pred, target)
        
        return results
    
    def _compute_snr(self, pred: np.ndarray, target: np.ndarray) -> float:
        """Compute signal-to-noise ratio."""
        signal_power = np.mean(target ** 2)
        noise_power = np.mean((pred - target) ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def evaluate_dataset(self, 
                        pred_files: List[str], 
                        target_files: List[str],
                        output_dir: str = "evaluation_results") -> Dict[str, float]:
        """Evaluate a dataset of audio files."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        all_results = []
        
        for pred_file, target_file in zip(pred_files, target_files):
            try:
                pred_audio, _ = sf.read(pred_file)
                target_audio, _ = sf.read(target_file)
                
                results = self.evaluate_audio_pair(pred_audio, target_audio)
                all_results.append(results)
                
            except Exception as e:
                logger.error(f"Error evaluating {pred_file}: {e}")
                continue
        
        # Compute average metrics
        avg_results = {}
        for metric in all_results[0].keys():
            values = [r[metric] for r in all_results if not np.isnan(r[metric]) and not np.isinf(r[metric])]
            if values:
                avg_results[f"avg_{metric}"] = np.mean(values)
                avg_results[f"std_{metric}"] = np.std(values)
        
        # Save results
        with open(output_path / "evaluation_results.json", 'w') as f:
            json.dump(avg_results, f, indent=2)
        
        return avg_results

class AudioVisualizer:
    """Visualization tools for audio analysis."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
    
    def plot_spectrogram_comparison(self, 
                                   pred: np.ndarray, 
                                   target: np.ndarray, 
                                   save_path: str = None):
        """Plot spectrogram comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Time domain
        time = np.linspace(0, len(pred) / self.sample_rate, len(pred))
        axes[0, 0].plot(time, pred, alpha=0.7, label='Predicted')
        axes[0, 0].plot(time, target, alpha=0.7, label='Target')
        axes[0, 0].set_title('Time Domain')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].legend()
        
        # Spectrograms
        pred_stft = librosa.stft(pred)
        target_stft = librosa.stft(target)
        
        pred_db = librosa.amplitude_to_db(np.abs(pred_stft))
        target_db = librosa.amplitude_to_db(np.abs(target_stft))
        
        librosa.display.specshow(pred_db, sr=self.sample_rate, ax=axes[0, 1])
        axes[0, 1].set_title('Predicted Spectrogram')
        
        librosa.display.specshow(target_db, sr=self.sample_rate, ax=axes[1, 0])
        axes[1, 0].set_title('Target Spectrogram')
        
        # Difference spectrogram
        diff_db = pred_db - target_db
        librosa.display.specshow(diff_db, sr=self.sample_rate, ax=axes[1, 1])
        axes[1, 1].set_title('Difference Spectrogram')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_comparison(self, 
                              metrics_dict: Dict[str, List[float]], 
                              save_path: str = None):
        """Plot metrics comparison."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (metric, values) in enumerate(metrics_dict.items()):
            if i < len(axes):
                axes[i].hist(values, bins=20, alpha=0.7)
                axes[i].set_title(f'{metric} Distribution')
                axes[i].set_xlabel(metric)
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

def evaluate_model(model_path: str, 
                  test_data_dir: str,
                  output_dir: str = "evaluation_results") -> Dict[str, float]:
    """
    Evaluate a trained model on test data.
    
    Args:
        model_path: Path to trained model
        test_data_dir: Directory containing test data
        output_dir: Directory to save evaluation results
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Load model
    model = torch.load(model_path, map_location='cpu')
    
    # Find test files
    test_dir = Path(test_data_dir)
    pred_files = list(test_dir.glob("**/*_enhanced.wav"))
    target_files = list(test_dir.glob("**/*_target.wav"))
    
    # Create evaluator
    evaluator = AudioQualityEvaluator()
    
    # Evaluate
    results = evaluator.evaluate_dataset(pred_files, target_files, output_dir)
    
    logger.info("Evaluation completed!")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.6f}")
    
    return results

if __name__ == "__main__":
    # Example usage
    evaluator = AudioQualityEvaluator()
    
    # Generate sample audio for testing
    t = np.linspace(0, 3, 22050 * 3)
    target = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 880 * t)
    pred = target + 0.1 * np.random.randn(len(target))
    
    # Evaluate
    results = evaluator.evaluate_audio_pair(pred, target)
    print("Evaluation results:")
    for metric, value in results.items():
        print(f"{metric}: {value:.6f}")
