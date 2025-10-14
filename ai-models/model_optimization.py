"""
Model optimization for edge deployment.
Includes quantization, pruning, and ONNX/TFLite conversion.
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
import torch.jit
import onnx
import onnxruntime as ort
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import json
import argparse

# TensorFlow Lite conversion (optional)
try:
    import tensorflow as tf
    TFLITE_AVAILABLE = True
except ImportError:
    TFLITE_AVAILABLE = False
    print("TensorFlow not available. TFLite conversion will be skipped.")

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Model optimization for edge deployment."""
    
    def __init__(self, model: nn.Module, device: str = "cpu"):
        """
        Initialize the model optimizer.
        
        Args:
            model: PyTorch model to optimize
            device: Device to run optimization on
        """
        self.model = model
        self.device = device
        self.model.eval()
        
    def quantize_model(self, 
                      calibration_data: torch.Tensor,
                      quantization_type: str = "dynamic") -> nn.Module:
        """
        Quantize the model for reduced size and faster inference.
        
        Args:
            calibration_data: Sample data for calibration
            quantization_type: Type of quantization ("dynamic", "static", "qat")
            
        Returns:
            Quantized model
        """
        logger.info(f"Quantizing model with {quantization_type} quantization...")
        
        if quantization_type == "dynamic":
            # Dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                self.model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
            )
            
        elif quantization_type == "static":
            # Static quantization
            self.model.qconfig = quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(self.model, inplace=True)
            
            # Calibrate with sample data
            with torch.no_grad():
                for i in range(10):  # Use multiple samples for calibration
                    sample_input = calibration_data[i:i+1].to(self.device)
                    _ = self.model(sample_input)
            
            quantized_model = torch.quantization.convert(self.model)
            
        elif quantization_type == "qat":
            # Quantization Aware Training
            self.model.qconfig = quantization.get_default_qat_qconfig('fbgemm')
            torch.quantization.prepare_qat(self.model, inplace=True)
            quantized_model = torch.quantization.convert(self.model)
            
        else:
            raise ValueError(f"Unknown quantization type: {quantization_type}")
        
        logger.info("Quantization completed!")
        return quantized_model
    
    def prune_model(self, 
                   sparsity: float = 0.5,
                   method: str = "magnitude") -> nn.Module:
        """
        Prune the model to reduce parameters.
        
        Args:
            sparsity: Target sparsity (0.0 to 1.0)
            method: Pruning method ("magnitude", "random")
            
        Returns:
            Pruned model
        """
        logger.info(f"Pruning model with {method} pruning to {sparsity} sparsity...")
        
        # Get parameters to prune
        parameters_to_prune = []
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # Apply pruning
        if method == "magnitude":
            torch.nn.utils.prune.global_unstructured(
                parameters_to_prune,
                pruning_method=torch.nn.utils.prune.L1Unstructured,
                amount=sparsity,
            )
        elif method == "random":
            torch.nn.utils.prune.global_unstructured(
                parameters_to_prune,
                pruning_method=torch.nn.utils.prune.RandomUnstructured,
                amount=sparsity,
            )
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        
        # Remove pruning reparameterization
        for module, param_name in parameters_to_prune:
            torch.nn.utils.prune.remove(module, param_name)
        
        logger.info("Pruning completed!")
        return self.model
    
    def convert_to_onnx(self, 
                        sample_input: torch.Tensor,
                        output_path: str,
                        opset_version: int = 11) -> str:
        """
        Convert model to ONNX format.
        
        Args:
            sample_input: Sample input tensor
            output_path: Path to save ONNX model
            opset_version: ONNX opset version
            
        Returns:
            Path to saved ONNX model
        """
        logger.info("Converting model to ONNX...")
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            sample_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'sequence_length'},
                'output': {0: 'batch_size', 2: 'sequence_length'}
            }
        )
        
        # Verify ONNX model
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        
        logger.info(f"ONNX model saved to {output_path}")
        return output_path
    
    def convert_to_tflite(self, 
                          sample_input: torch.Tensor,
                          output_path: str) -> Optional[str]:
        """
        Convert model to TensorFlow Lite format.
        
        Args:
            sample_input: Sample input tensor
            output_path: Path to save TFLite model
            
        Returns:
            Path to saved TFLite model or None if TensorFlow not available
        """
        if not TFLITE_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping TFLite conversion.")
            return None
        
        logger.info("Converting model to TensorFlow Lite...")
        
        # First convert to ONNX
        onnx_path = output_path.replace('.tflite', '.onnx')
        self.convert_to_onnx(sample_input, onnx_path)
        
        # Convert ONNX to TensorFlow
        import onnx_tf
        tf_rep = onnx_tf.backend.prepare(onnx_model)
        tf_model = tf_rep.export_graph()
        
        # Convert to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        
        tflite_model = converter.convert()
        
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TensorFlow Lite model saved to {output_path}")
        return output_path
    
    def optimize_for_mobile(self, 
                           sample_input: torch.Tensor,
                           output_dir: str = "optimized_models") -> Dict[str, str]:
        """
        Optimize model for mobile deployment.
        
        Args:
            sample_input: Sample input tensor
            output_dir: Directory to save optimized models
            
        Returns:
            Dictionary of saved model paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        saved_models = {}
        
        # 1. Quantized model
        quantized_model = self.quantize_model(sample_input, "dynamic")
        quantized_path = output_path / "model_quantized.pth"
        torch.save(quantized_model.state_dict(), quantized_path)
        saved_models['quantized'] = str(quantized_path)
        
        # 2. Pruned model
        pruned_model = self.prune_model(sparsity=0.3)
        pruned_path = output_path / "model_pruned.pth"
        torch.save(pruned_model.state_dict(), pruned_path)
        saved_models['pruned'] = str(pruned_path)
        
        # 3. ONNX model
        onnx_path = output_path / "model.onnx"
        self.convert_to_onnx(sample_input, str(onnx_path))
        saved_models['onnx'] = str(onnx_path)
        
        # 4. TensorFlow Lite model (if available)
        if TFLITE_AVAILABLE:
            tflite_path = output_path / "model.tflite"
            tflite_result = self.convert_to_tflite(sample_input, str(tflite_path))
            if tflite_result:
                saved_models['tflite'] = tflite_result
        
        # 5. TorchScript model
        scripted_model = torch.jit.script(self.model)
        scripted_path = output_path / "model_scripted.pt"
        scripted_model.save(str(scripted_path))
        saved_models['scripted'] = str(scripted_path)
        
        logger.info("Mobile optimization completed!")
        return saved_models

class ModelBenchmark:
    """Benchmark optimized models for performance comparison."""
    
    def __init__(self, device: str = "cpu"):
        self.device = device
    
    def benchmark_model(self, 
                       model: nn.Module,
                       sample_input: torch.Tensor,
                       num_iterations: int = 100) -> Dict[str, float]:
        """
        Benchmark model performance.
        
        Args:
            model: Model to benchmark
            sample_input: Sample input tensor
            num_iterations: Number of iterations for benchmarking
            
        Returns:
            Performance metrics
        """
        model.eval()
        model = model.to(self.device)
        sample_input = sample_input.to(self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Benchmark
        import time
        times = []
        
        with torch.no_grad():
            for _ in range(num_iterations):
                start_time = time.time()
                _ = model(sample_input)
                end_time = time.time()
                times.append(end_time - start_time)
        
        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        # Calculate throughput
        batch_size = sample_input.shape[0]
        throughput = batch_size / avg_time
        
        return {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'throughput': throughput
        }
    
    def benchmark_onnx_model(self, 
                            onnx_path: str,
                            sample_input: np.ndarray,
                            num_iterations: int = 100) -> Dict[str, float]:
        """Benchmark ONNX model."""
        # Create ONNX Runtime session
        session = ort.InferenceSession(onnx_path)
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, {'input': sample_input})
        
        # Benchmark
        import time
        times = []
        
        for _ in range(num_iterations):
            start_time = time.time()
            _ = session.run(None, {'input': sample_input})
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Calculate metrics
        avg_time = np.mean(times)
        std_time = np.std(times)
        min_time = np.min(times)
        max_time = np.max(times)
        
        batch_size = sample_input.shape[0]
        throughput = batch_size / avg_time
        
        return {
            'avg_inference_time': avg_time,
            'std_inference_time': std_time,
            'min_inference_time': min_time,
            'max_inference_time': max_time,
            'throughput': throughput
        }
    
    def compare_models(self, 
                      models: Dict[str, nn.Module],
                      sample_input: torch.Tensor,
                      num_iterations: int = 100) -> Dict[str, Dict[str, float]]:
        """
        Compare performance of multiple models.
        
        Args:
            models: Dictionary of model names and models
            sample_input: Sample input tensor
            num_iterations: Number of iterations for benchmarking
            
        Returns:
            Performance comparison results
        """
        results = {}
        
        for name, model in models.items():
            logger.info(f"Benchmarking {name}...")
            results[name] = self.benchmark_model(model, sample_input, num_iterations)
        
        return results

def optimize_model_for_edge(model_path: str,
                          sample_input: torch.Tensor,
                          output_dir: str = "optimized_models",
                          sparsity: float = 0.3) -> Dict[str, str]:
    """
    Complete optimization pipeline for edge deployment.
    
    Args:
        model_path: Path to trained model
        sample_input: Sample input tensor
        output_dir: Directory to save optimized models
        sparsity: Pruning sparsity
        
    Returns:
        Dictionary of saved model paths
    """
    # Load model
    model = torch.load(model_path, map_location='cpu')
    
    # Create optimizer
    optimizer = ModelOptimizer(model)
    
    # Optimize for mobile
    saved_models = optimizer.optimize_for_mobile(sample_input, output_dir)
    
    # Benchmark models
    benchmarker = ModelBenchmark()
    
    # Create comparison models
    models_to_compare = {
        'original': model,
        'quantized': torch.quantization.quantize_dynamic(
            model, {nn.Linear, nn.Conv1d}, dtype=torch.qint8
        )
    }
    
    # Compare performance
    results = benchmarker.compare_models(models_to_compare, sample_input)
    
    # Save benchmark results
    with open(Path(output_dir) / "benchmark_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Edge optimization completed!")
    return saved_models

def main():
    """Main function for model optimization."""
    parser = argparse.ArgumentParser(description="Optimize model for edge deployment")
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="optimized_models",
                       help="Directory to save optimized models")
    parser.add_argument("--sparsity", type=float, default=0.3,
                       help="Pruning sparsity")
    parser.add_argument("--sample_length", type=int, default=22050 * 3,
                       help="Length of sample input")
    
    args = parser.parse_args()
    
    # Create sample input
    sample_input = torch.randn(1, 1, args.sample_length)
    
    # Optimize model
    saved_models = optimize_model_for_edge(
        args.model_path,
        sample_input,
        args.output_dir,
        args.sparsity
    )
    
    print("Optimization completed!")
    print("Saved models:")
    for name, path in saved_models.items():
        print(f"  {name}: {path}")

if __name__ == "__main__":
    main()
