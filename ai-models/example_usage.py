#!/usr/bin/env python3
"""
Example usage of the AI Audio Enhancement pipeline.
This script demonstrates how to use the complete pipeline for audio enhancement.
"""

import os
import sys
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
import logging

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_architecture import AudioUNet, test_model
from inference_pipeline import AudioEnhancer
from evaluation_metrics import AudioQualityEvaluator
from model_optimization import ModelOptimizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_audio(duration: float = 3.0, sample_rate: int = 22050) -> np.ndarray:
    """Create sample audio for testing."""
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Create a complex audio signal
    audio = (
        0.5 * np.sin(2 * np.pi * 440 * t) +  # A4 note
        0.3 * np.sin(2 * np.pi * 880 * t) +  # A5 note
        0.2 * np.sin(2 * np.pi * 1320 * t) + # E6 note
        0.1 * np.random.randn(len(t))        # Some noise
    )
    
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    return audio

def simulate_lossy_audio(audio: np.ndarray, method: str = "mp3") -> np.ndarray:
    """Simulate lossy audio compression."""
    if method == "mp3":
        # Simulate MP3 compression artifacts
        # Add noise and filter high frequencies
        noise_level = 0.01
        noise = np.random.normal(0, noise_level, audio.shape)
        lossy = audio + noise
        
        # Simulate frequency domain artifacts
        from scipy import signal
        # Low-pass filter to simulate MP3 compression
        b, a = signal.butter(4, 0.4, btype='low')
        lossy = signal.filtfilt(b, a, lossy)
        
    elif method == "downsample":
        # Downsample and upsample
        downsample_factor = 2
        downsampled = audio[::downsample_factor]
        lossy = np.repeat(downsampled, downsample_factor)
        lossy = lossy[:len(audio)]
        
    else:
        # Add noise
        snr_db = 20
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), audio.shape)
        lossy = audio + noise
    
    return lossy

def demonstrate_model_architecture():
    """Demonstrate the model architecture."""
    logger.info("=" * 50)
    logger.info("DEMONSTRATING MODEL ARCHITECTURE")
    logger.info("=" * 50)
    
    # Create and test model
    model = test_model()
    
    # Print model information
    model_info = model.get_model_size()
    logger.info(f"Model parameters: {model_info['total_parameters']:,}")
    logger.info(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    return model

def demonstrate_audio_enhancement():
    """Demonstrate audio enhancement pipeline."""
    logger.info("=" * 50)
    logger.info("DEMONSTRATING AUDIO ENHANCEMENT")
    logger.info("=" * 50)
    
    # Create sample audio
    logger.info("Creating sample audio...")
    original_audio = create_sample_audio(duration=3.0)
    
    # Simulate lossy audio
    logger.info("Simulating lossy audio...")
    lossy_audio = simulate_lossy_audio(original_audio, method="mp3")
    
    # Save audio files for demonstration
    sf.write("original_audio.wav", original_audio, 22050)
    sf.write("lossy_audio.wav", lossy_audio, 22050)
    logger.info("Saved original and lossy audio files")
    
    # Create a simple model for demonstration
    model = AudioUNet(base_channels=16, depth=3)
    model.eval()
    
    # Enhance audio
    logger.info("Enhancing audio...")
    with torch.no_grad():
        # Convert to tensor
        input_tensor = torch.FloatTensor(lossy_audio).unsqueeze(0).unsqueeze(0)
        
        # Run inference
        enhanced_tensor = model(input_tensor)
        enhanced_audio = enhanced_tensor.squeeze().numpy()
    
    # Save enhanced audio
    sf.write("enhanced_audio.wav", enhanced_audio, 22050)
    logger.info("Saved enhanced audio file")
    
    return original_audio, lossy_audio, enhanced_audio

def demonstrate_evaluation():
    """Demonstrate evaluation metrics."""
    logger.info("=" * 50)
    logger.info("DEMONSTRATING EVALUATION METRICS")
    logger.info("=" * 50)
    
    # Create sample audio
    original_audio = create_sample_audio(duration=2.0)
    lossy_audio = simulate_lossy_audio(original_audio, method="mp3")
    
    # Create a simple enhanced audio (in practice, this would come from your model)
    enhanced_audio = lossy_audio + 0.1 * np.random.randn(len(lossy_audio))
    
    # Evaluate
    evaluator = AudioQualityEvaluator()
    results = evaluator.evaluate_audio_pair(enhanced_audio, original_audio)
    
    logger.info("Evaluation results:")
    for metric, value in results.items():
        logger.info(f"  {metric}: {value:.6f}")
    
    return results

def demonstrate_model_optimization():
    """Demonstrate model optimization."""
    logger.info("=" * 50)
    logger.info("DEMONSTRATING MODEL OPTIMIZATION")
    logger.info("=" * 50)
    
    # Create a simple model
    model = AudioUNet(base_channels=16, depth=3)
    
    # Create sample input
    sample_input = torch.randn(1, 1, 22050 * 3)
    
    # Create optimizer
    optimizer = ModelOptimizer(model)
    
    # Demonstrate quantization
    logger.info("Demonstrating quantization...")
    quantized_model = optimizer.quantize_model(sample_input, "dynamic")
    
    # Demonstrate pruning
    logger.info("Demonstrating pruning...")
    pruned_model = optimizer.prune_model(sparsity=0.3)
    
    # Print model sizes
    original_size = model.get_model_size()
    logger.info(f"Original model size: {original_size['model_size_mb']:.2f} MB")
    
    logger.info("Model optimization demonstration completed!")

def demonstrate_inference_pipeline():
    """Demonstrate inference pipeline."""
    logger.info("=" * 50)
    logger.info("DEMONSTRATING INFERENCE PIPELINE")
    logger.info("=" * 50)
    
    # Create a simple model
    model = AudioUNet(base_channels=16, depth=3)
    
    # Save model for demonstration
    model_path = "demo_model.pth"
    torch.save(model.state_dict(), model_path)
    
    # Create enhancer
    enhancer = AudioEnhancer(
        model_path=model_path,
        model_type="pytorch",
        device="cpu",
        chunk_size=1024
    )
    
    # Create sample audio
    audio = create_sample_audio(duration=3.0)
    
    # Enhance audio
    logger.info("Enhancing audio with inference pipeline...")
    enhanced_audio = enhancer.enhance_chunk(audio)
    
    # Save results
    sf.write("pipeline_enhanced_audio.wav", enhanced_audio, 22050)
    logger.info("Saved enhanced audio from inference pipeline")
    
    # Get performance metrics
    metrics = enhancer._calculate_metrics()
    logger.info("Performance metrics:")
    for metric, value in metrics.items():
        logger.info(f"  {metric}: {value:.6f}")
    
    return enhanced_audio

def main():
    """Main demonstration function."""
    logger.info("AI Audio Enhancement Pipeline Demonstration")
    logger.info("=" * 60)
    
    try:
        # 1. Demonstrate model architecture
        model = demonstrate_model_architecture()
        
        # 2. Demonstrate audio enhancement
        original, lossy, enhanced = demonstrate_audio_enhancement()
        
        # 3. Demonstrate evaluation
        evaluation_results = demonstrate_evaluation()
        
        # 4. Demonstrate model optimization
        demonstrate_model_optimization()
        
        # 5. Demonstrate inference pipeline
        pipeline_enhanced = demonstrate_inference_pipeline()
        
        logger.info("=" * 60)
        logger.info("DEMONSTRATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        # Print summary
        logger.info("Generated files:")
        logger.info("  - original_audio.wav: Original audio")
        logger.info("  - lossy_audio.wav: Simulated lossy audio")
        logger.info("  - enhanced_audio.wav: Enhanced audio (simple model)")
        logger.info("  - pipeline_enhanced_audio.wav: Enhanced audio (pipeline)")
        
        logger.info("\nNext steps:")
        logger.info("  1. Train a model with your data using train.py")
        logger.info("  2. Use the trained model for real audio enhancement")
        logger.info("  3. Optimize the model for mobile deployment")
        logger.info("  4. Deploy on mobile devices for real-time enhancement")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise

if __name__ == "__main__":
    main()
