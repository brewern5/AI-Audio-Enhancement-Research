"""
Real-time inference pipeline for audio enhancement.
Supports both PyTorch and ONNX models for edge deployment.
"""

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import librosa
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import time
import threading
import queue
import argparse
from collections import deque
import json

# ONNX Runtime for optimized inference
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("ONNX Runtime not available. ONNX inference will be skipped.")

logger = logging.getLogger(__name__)

class AudioEnhancer:
    """Real-time audio enhancement using trained models."""
    
    def __init__(self, 
                 model_path: str,
                 model_type: str = "pytorch",
                 device: str = "cpu",
                 sample_rate: int = 22050,
                 chunk_size: int = 1024,
                 overlap: float = 0.5):
        """
        Initialize the audio enhancer.
        
        Args:
            model_path: Path to trained model
            model_type: Type of model ("pytorch", "onnx")
            device: Device to run inference on
            sample_rate: Audio sample rate
            chunk_size: Size of audio chunks for processing
            overlap: Overlap ratio between chunks
        """
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.hop_size = int(chunk_size * (1 - overlap))
        
        # Load model
        self.model = self._load_model()
        
        # Audio buffer for streaming
        self.audio_buffer = deque(maxlen=chunk_size * 2)
        self.output_buffer = deque(maxlen=chunk_size * 2)
        
        # Performance tracking
        self.inference_times = deque(maxlen=100)
        self.total_samples_processed = 0
        
    def _load_model(self):
        """Load the model based on type."""
        if self.model_type == "pytorch":
            return self._load_pytorch_model()
        elif self.model_type == "onnx":
            return self._load_onnx_model()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_pytorch_model(self):
        """Load PyTorch model."""
        logger.info(f"Loading PyTorch model from {self.model_path}")
        
        # Load model state dict
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Create model architecture (you'll need to import your model class)
        from model_architecture import AudioUNet
        model = AudioUNet(base_channels=32, depth=4)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.eval()
        model = model.to(self.device)
        
        return model
    
    def _load_onnx_model(self):
        """Load ONNX model."""
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available")
        
        logger.info(f"Loading ONNX model from {self.model_path}")
        
        # Create ONNX Runtime session
        providers = ['CPUExecutionProvider']
        if self.device == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = ort.InferenceSession(self.model_path, providers=providers)
        return session
    
    def preprocess_audio(self, audio: np.ndarray) -> torch.Tensor:
        """Preprocess audio for model input."""
        # Normalize audio
        audio = audio / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).unsqueeze(0)  # Add batch and channel dims
        
        return audio_tensor.to(self.device)
    
    def postprocess_audio(self, enhanced_tensor: torch.Tensor) -> np.ndarray:
        """Postprocess model output to audio."""
        # Convert to numpy
        enhanced_audio = enhanced_tensor.squeeze().cpu().numpy()
        
        # Ensure audio is in valid range
        enhanced_audio = np.clip(enhanced_audio, -1.0, 1.0)
        
        return enhanced_audio
    
    def enhance_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """Enhance a single audio chunk."""
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess_audio(audio_chunk)
        
        # Run inference
        with torch.no_grad():
            if self.model_type == "pytorch":
                enhanced_tensor = self.model(input_tensor)
            else:  # ONNX
                input_name = self.model.get_inputs()[0].name
                output_name = self.model.get_outputs()[0].name
                enhanced_tensor = self.model.run([output_name], {input_name: input_tensor.cpu().numpy()})[0]
                enhanced_tensor = torch.FloatTensor(enhanced_tensor)
        
        # Postprocess
        enhanced_audio = self.postprocess_audio(enhanced_tensor)
        
        # Track performance
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.total_samples_processed += len(audio_chunk)
        
        return enhanced_audio
    
    def enhance_audio_file(self, input_path: str, output_path: str) -> Dict[str, float]:
        """
        Enhance an entire audio file.
        
        Args:
            input_path: Path to input audio file
            output_path: Path to save enhanced audio
            
        Returns:
            Performance metrics
        """
        logger.info(f"Enhancing audio file: {input_path}")
        
        # Load audio
        audio, sr = librosa.load(input_path, sr=self.sample_rate)
        
        # Process in chunks
        enhanced_chunks = []
        num_chunks = len(audio) // self.hop_size
        
        for i in range(0, len(audio) - self.chunk_size + 1, self.hop_size):
            chunk = audio[i:i + self.chunk_size]
            if len(chunk) == self.chunk_size:
                enhanced_chunk = self.enhance_chunk(chunk)
                enhanced_chunks.append(enhanced_chunk)
        
        # Combine chunks with overlap handling
        enhanced_audio = self._combine_chunks(enhanced_chunks, len(audio))
        
        # Save enhanced audio
        sf.write(output_path, enhanced_audio, self.sample_rate)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics()
        
        logger.info(f"Enhanced audio saved to: {output_path}")
        return metrics
    
    def _combine_chunks(self, chunks: List[np.ndarray], target_length: int) -> np.ndarray:
        """Combine overlapping chunks into final audio."""
        if not chunks:
            return np.array([])
        
        # Initialize output array
        output = np.zeros(target_length)
        weights = np.zeros(target_length)
        
        # Add chunks with overlap handling
        for i, chunk in enumerate(chunks):
            start_idx = i * self.hop_size
            end_idx = start_idx + len(chunk)
            
            # Ensure we don't exceed target length
            end_idx = min(end_idx, target_length)
            chunk = chunk[:end_idx - start_idx]
            
            # Add chunk to output
            output[start_idx:end_idx] += chunk
            weights[start_idx:end_idx] += 1
        
        # Normalize by weights to handle overlap
        weights = np.maximum(weights, 1)  # Avoid division by zero
        output = output / weights
        
        return output
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics."""
        if not self.inference_times:
            return {}
        
        avg_inference_time = np.mean(self.inference_times)
        std_inference_time = np.std(self.inference_times)
        min_inference_time = np.min(self.inference_times)
        max_inference_time = np.max(self.inference_times)
        
        # Calculate real-time factor
        samples_per_second = self.sample_rate
        processing_time = sum(self.inference_times)
        real_time_duration = self.total_samples_processed / samples_per_second
        real_time_factor = processing_time / real_time_duration if real_time_duration > 0 else 0
        
        return {
            'avg_inference_time': avg_inference_time,
            'std_inference_time': std_inference_time,
            'min_inference_time': min_inference_time,
            'max_inference_time': max_inference_time,
            'real_time_factor': real_time_factor,
            'total_samples_processed': self.total_samples_processed
        }

class RealTimeAudioEnhancer:
    """Real-time audio enhancement with streaming support."""
    
    def __init__(self, 
                 model_path: str,
                 model_type: str = "pytorch",
                 device: str = "cpu",
                 sample_rate: int = 22050,
                 chunk_size: int = 1024):
        """
        Initialize real-time audio enhancer.
        
        Args:
            model_path: Path to trained model
            model_type: Type of model ("pytorch", "onnx")
            device: Device to run inference on
            sample_rate: Audio sample rate
            chunk_size: Size of audio chunks for processing
        """
        self.enhancer = AudioEnhancer(
            model_path=model_path,
            model_type=model_type,
            device=device,
            sample_rate=sample_rate,
            chunk_size=chunk_size
        )
        
        # Threading for real-time processing
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.processing_thread = None
        self.is_processing = False
        
    def start_processing(self):
        """Start real-time processing thread."""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
        logger.info("Real-time processing started")
    
    def stop_processing(self):
        """Stop real-time processing thread."""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()
        
        logger.info("Real-time processing stopped")
    
    def _processing_loop(self):
        """Main processing loop for real-time enhancement."""
        while self.is_processing:
            try:
                # Get input audio chunk
                input_chunk = self.input_queue.get(timeout=0.1)
                
                # Enhance chunk
                enhanced_chunk = self.enhancer.enhance_chunk(input_chunk)
                
                # Put enhanced chunk in output queue
                self.output_queue.put(enhanced_chunk)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                continue
    
    def process_audio_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single audio chunk in real-time.
        
        Args:
            audio_chunk: Input audio chunk
            
        Returns:
            Enhanced audio chunk or None if not ready
        """
        # Add to input queue
        self.input_queue.put(audio_chunk)
        
        # Try to get enhanced chunk
        try:
            enhanced_chunk = self.output_queue.get_nowait()
            return enhanced_chunk
        except queue.Empty:
            return None
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        return self.enhancer._calculate_metrics()

class BatchAudioEnhancer:
    """Batch processing for multiple audio files."""
    
    def __init__(self, 
                 model_path: str,
                 model_type: str = "pytorch",
                 device: str = "cpu",
                 batch_size: int = 4):
        """
        Initialize batch audio enhancer.
        
        Args:
            model_path: Path to trained model
            model_type: Type of model ("pytorch", "onnx")
            device: Device to run inference on
            batch_size: Batch size for processing
        """
        self.enhancer = AudioEnhancer(
            model_path=model_path,
            model_type=model_type,
            device=device
        )
        self.batch_size = batch_size
    
    def enhance_batch(self, 
                     input_paths: List[str], 
                     output_dir: str) -> Dict[str, Dict[str, float]]:
        """
        Enhance a batch of audio files.
        
        Args:
            input_paths: List of input audio file paths
            output_dir: Directory to save enhanced audio files
            
        Returns:
            Dictionary of performance metrics for each file
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {}
        
        for i, input_path in enumerate(input_paths):
            logger.info(f"Processing file {i+1}/{len(input_paths)}: {input_path}")
            
            # Create output path
            input_file = Path(input_path)
            output_path_file = output_path / f"{input_file.stem}_enhanced{input_file.suffix}"
            
            # Enhance audio
            metrics = self.enhancer.enhance_audio_file(input_path, str(output_path_file))
            results[input_path] = metrics
        
        return results

def main():
    """Main function for inference pipeline."""
    parser = argparse.ArgumentParser(description="Audio enhancement inference")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to input audio file")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save enhanced audio")
    parser.add_argument("--model_type", type=str, default="pytorch",
                       choices=["pytorch", "onnx"],
                       help="Type of model")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to run inference on")
    parser.add_argument("--chunk_size", type=int, default=1024,
                       help="Size of audio chunks for processing")
    
    args = parser.parse_args()
    
    # Create enhancer
    enhancer = AudioEnhancer(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device,
        chunk_size=args.chunk_size
    )
    
    # Enhance audio
    metrics = enhancer.enhance_audio_file(args.input_path, args.output_path)
    
    print("Enhancement completed!")
    print("Performance metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.6f}")

if __name__ == "__main__":
    main()
