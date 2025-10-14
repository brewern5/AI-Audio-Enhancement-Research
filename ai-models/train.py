#!/usr/bin/env python3
"""
Main training script for AI audio enhancement.
This script orchestrates the entire training pipeline.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import json
import torch
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_preparation import AudioDataPreparator
from model_architecture import AudioUNet, create_model
from training_pipeline import AudioTrainer, create_data_loaders
from evaluation_metrics import AudioQualityEvaluator
from model_optimization import ModelOptimizer
from inference_pipeline import AudioEnhancer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def prepare_data(input_dir: str, output_dir: str, max_files: int = None):
    """Prepare training data from raw audio files."""
    logger.info("Starting data preparation...")
    
    preparator = AudioDataPreparator(
        sample_rate=22050,
        duration=3.0,
        target_dir=output_dir
    )
    
    metadata = preparator.process_dataset(
        input_dir=input_dir,
        max_files=max_files,
        num_workers=4
    )
    
    logger.info(f"Data preparation completed! Processed {len(metadata)} files.")
    return metadata

def train_model(data_dir: str, config: dict, save_dir: str = "checkpoints"):
    """Train the audio enhancement model."""
    logger.info("Starting model training...")
    
    # Create data loaders
    metadata_path = Path(data_dir) / "metadata" / "dataset_metadata.json"
    train_loader, val_loader = create_data_loaders(
        str(metadata_path),
        batch_size=config.get('batch_size', 16),
        train_split=0.8,
        num_workers=4
    )
    
    # Create model
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
        use_wandb=config.get('use_wandb', False)
    )
    
    # Train model
    trainer.train(
        num_epochs=config.get('epochs', 100),
        save_dir=save_dir
    )
    
    return trainer

def evaluate_model(model_path: str, test_data_dir: str, output_dir: str = "evaluation_results"):
    """Evaluate the trained model."""
    logger.info("Starting model evaluation...")
    
    # Create evaluator
    evaluator = AudioQualityEvaluator()
    
    # Find test files
    test_dir = Path(test_data_dir)
    pred_files = list(test_dir.glob("**/*_enhanced.wav"))
    target_files = list(test_dir.glob("**/*_target.wav"))
    
    if not pred_files or not target_files:
        logger.warning("No test files found. Skipping evaluation.")
        return {}
    
    # Evaluate
    results = evaluator.evaluate_dataset(pred_files, target_files, output_dir)
    
    logger.info("Evaluation completed!")
    for metric, value in results.items():
        logger.info(f"{metric}: {value:.6f}")
    
    return results

def optimize_model(model_path: str, sample_input: torch.Tensor, output_dir: str = "optimized_models"):
    """Optimize model for edge deployment."""
    logger.info("Starting model optimization...")
    
    # Load model
    model = torch.load(model_path, map_location='cpu')
    
    # Create optimizer
    optimizer = ModelOptimizer(model)
    
    # Optimize for mobile
    saved_models = optimizer.optimize_for_mobile(sample_input, output_dir)
    
    logger.info("Model optimization completed!")
    return saved_models

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(description="Train AI audio enhancement model")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing raw audio files")
    parser.add_argument("--output_dir", type=str, default="processed_data",
                       help="Directory to save processed data")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory to save model checkpoints")
    parser.add_argument("--config", type=str, default="config.json",
                       help="Path to configuration file")
    parser.add_argument("--max_files", type=int, default=None,
                       help="Maximum number of files to process")
    parser.add_argument("--skip_data_prep", action="store_true",
                       help="Skip data preparation step")
    parser.add_argument("--skip_training", action="store_true",
                       help="Skip training step")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Skip evaluation step")
    parser.add_argument("--skip_optimization", action="store_true",
                       help="Skip optimization step")
    
    args = parser.parse_args()
    
    # Load configuration
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            'base_channels': 32,
            'depth': 4,
            'dropout': 0.1,
            'batch_size': 16,
            'epochs': 100,
            'learning_rate': 1e-3,
            'use_wandb': False
        }
        # Save default config
        with open(args.config, 'w') as f:
            json.dump(config, f, indent=2)
        logger.info(f"Created default configuration file: {args.config}")
    
    logger.info("Starting AI Audio Enhancement Training Pipeline")
    logger.info(f"Configuration: {config}")
    
    # Step 1: Data Preparation
    if not args.skip_data_prep:
        logger.info("=" * 50)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("=" * 50)
        
        metadata = prepare_data(args.input_dir, args.output_dir, args.max_files)
        
        if not metadata:
            logger.error("No data processed. Exiting.")
            return
    else:
        logger.info("Skipping data preparation step.")
    
    # Step 2: Model Training
    if not args.skip_training:
        logger.info("=" * 50)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("=" * 50)
        
        trainer = train_model(args.output_dir, config, args.checkpoint_dir)
        
        # Get best model path
        best_model_path = Path(args.checkpoint_dir) / "checkpoint_epoch_0_best.pth"
        if not best_model_path.exists():
            # Find the latest checkpoint
            checkpoints = list(Path(args.checkpoint_dir).glob("checkpoint_epoch_*_best.pth"))
            if checkpoints:
                best_model_path = max(checkpoints, key=lambda x: x.stat().st_mtime)
            else:
                logger.error("No trained model found. Exiting.")
                return
    else:
        logger.info("Skipping training step.")
        best_model_path = Path(args.checkpoint_dir) / "checkpoint_epoch_0_best.pth"
    
    # Step 3: Model Evaluation
    if not args.skip_evaluation:
        logger.info("=" * 50)
        logger.info("STEP 3: MODEL EVALUATION")
        logger.info("=" * 50)
        
        # Create sample enhanced files for evaluation
        logger.info("Creating sample enhanced files for evaluation...")
        
        # Load model for inference
        enhancer = AudioEnhancer(str(best_model_path))
        
        # Process a few files to create test data
        test_dir = Path("test_data")
        test_dir.mkdir(exist_ok=True)
        
        # Get some test files
        metadata_path = Path(args.output_dir) / "metadata" / "dataset_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        test_files = metadata[:5]  # Use first 5 files for testing
        
        for i, file_meta in enumerate(test_files):
            for segment in file_meta['segments'][:2]:  # Use first 2 segments
                # Load lossy audio
                lossy_audio, _ = sf.read(segment['lossy_path'])
                
                # Enhance audio
                enhanced_audio = enhancer.enhance_chunk(lossy_audio)
                
                # Save enhanced audio
                enhanced_path = test_dir / f"test_{i}_{segment['segment_idx']}_enhanced.wav"
                target_path = test_dir / f"test_{i}_{segment['segment_idx']}_target.wav"
                
                sf.write(str(enhanced_path), enhanced_audio, 22050)
                sf.write(str(target_path), lossy_audio, 22050)  # Use lossy as target for now
        
        # Evaluate model
        results = evaluate_model(str(best_model_path), str(test_dir))
        
        # Save evaluation results
        with open("evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2)
    else:
        logger.info("Skipping evaluation step.")
    
    # Step 4: Model Optimization
    if not args.skip_optimization:
        logger.info("=" * 50)
        logger.info("STEP 4: MODEL OPTIMIZATION")
        logger.info("=" * 50)
        
        # Create sample input
        sample_input = torch.randn(1, 1, 22050 * 3)  # 3 seconds at 22kHz
        
        # Optimize model
        saved_models = optimize_model(str(best_model_path), sample_input)
        
        logger.info("Optimized models saved:")
        for name, path in saved_models.items():
            logger.info(f"  {name}: {path}")
    else:
        logger.info("Skipping optimization step.")
    
    logger.info("=" * 50)
    logger.info("TRAINING PIPELINE COMPLETED!")
    logger.info("=" * 50)
    
    # Print summary
    logger.info("Summary:")
    logger.info(f"  - Processed data: {args.output_dir}")
    logger.info(f"  - Model checkpoints: {args.checkpoint_dir}")
    logger.info(f"  - Best model: {best_model_path}")
    
    if not args.skip_optimization:
        logger.info("  - Optimized models available for edge deployment")

if __name__ == "__main__":
    main()
