"""
Main training script for pneumonia classification
Orchestrates the complete training and evaluation pipeline
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import json
import shutil
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from config import (
    Config, 
    get_raw_preprocessing_config,
    get_histogram_matching_config,
    get_zscore_config
)
from data_handler import ChestXrayDataManager
from model import create_model
from trainer import PneumoniaTrainer
from evaluator import PneumoniaEvaluator

def setup_experiment_directory(config: Config) -> Path:
    """Setup experiment directory with timestamped folders"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = "{}_{}_{}".format(config.model.backbone, config.data.preprocessing_type, timestamp)
    
    exp_dir = Path(config.experiment.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (exp_dir / "checkpoints").mkdir(exist_ok=True)
    (exp_dir / "logs").mkdir(exist_ok=True)
    (exp_dir / "evaluation").mkdir(exist_ok=True)
    (exp_dir / "configs").mkdir(exist_ok=True)
    
    # Save config
    config_path = exp_dir / "configs" / "config.yaml"
    config.save(str(config_path))
    
    # Update config paths to use experiment directory
    config.experiment.log_dir = str(exp_dir / "logs")
    config.experiment.output_dir = str(exp_dir)
    
    print(f"Experiment directory created: {exp_dir}")
    return exp_dir

def train_single_model(config: Config, experiment_dir: Path) -> dict:
    """Train a single model with given configuration"""
    print(f"\n{'='*60}")
    print(f"TRAINING MODEL: {config.model.backbone} with {config.data.preprocessing_type} preprocessing")
    print(f"{'='*60}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup data
    print("Setting up data...")
    data_manager = ChestXrayDataManager(config)
    train_loader, val_loader, test_loader = data_manager.setup_data()
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config.model)
    print(f"Model created: {config.model.backbone}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Setup trainer
    trainer = PneumoniaTrainer(model, config, train_loader, val_loader, test_loader, log_dir=str(experiment_dir / "logs"))
    
    # Train model
    print("Starting training...")
    training_history = trainer.train()
    
    # Load best model for evaluation
    best_model_path = Path(config.experiment.output_dir) / "checkpoints" / "best_model.pth"
    if best_model_path.exists():
        print("Loading best model for evaluation...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model loaded (epoch {checkpoint['epoch']}, val_loss: {checkpoint['val_loss']:.4f})")
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = PneumoniaEvaluator(model, config, device)
    evaluation_dir = experiment_dir / "evaluation"
    evaluation_results = evaluator.evaluate(test_loader, save_dir=str(evaluation_dir))
    
    # Save evaluation results
    results_path = evaluation_dir / "evaluation_results.json"
    evaluator.save_results(str(results_path))
    
    # Compile experiment results
    experiment_results = {
        'config': config.to_dict(),
        'training_history': training_history,
        'evaluation': evaluation_results,
        'experiment_dir': str(experiment_dir),
        'best_model_path': str(best_model_path) if best_model_path.exists() else None
    }
    
    # Save experiment summary
    summary_path = experiment_dir / "experiment_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(experiment_results, f, indent=2, default=str)
    
    print(f"\n🎉 Training completed!")
    print(f"📁 Results saved to: {experiment_dir}")
    print(f"📊 Test AUROC: {evaluation_results['metrics']['auroc']:.4f}")
    print(f"📊 Test F1: {evaluation_results['metrics']['f1']:.4f}")
    
    return experiment_results

def compare_preprocessing_methods(base_config: Config) -> dict:
    """Train and compare models with different preprocessing methods"""
    print(f"\n{'='*80}")
    print("COMPARING PREPROCESSING METHODS")
    print(f"{'='*80}")
    
    # Get different preprocessing configs
    configs = {
        'raw': get_raw_preprocessing_config(),
        'histogram_matching': get_histogram_matching_config(),
        'zscore': get_zscore_config()
    }
    
    # Update base paths for all configs
    for config in configs.values():
        config.update_paths(base_config.experiment.output_dir)
        config.training.epochs = base_config.training.epochs  # Use same training settings
        config.training.batch_size = base_config.training.batch_size
        config.model.backbone = base_config.model.backbone
    
    results_comparison = {}
    experiment_dirs = {}
    
    # Train each model
    for method_name, config in configs.items():
        print(f"\n🔄 Training with {method_name} preprocessing...")
        
        # Setup experiment directory
        experiment_dir = setup_experiment_directory(config)
        experiment_dirs[method_name] = experiment_dir
        
        # Train model
        try:
            results = train_single_model(config, experiment_dir)
            results_comparison[method_name] = results
            print(f"✅ {method_name} preprocessing completed successfully")
        except Exception as e:
            print(f"❌ Error training {method_name} preprocessing: {e}")
            results_comparison[method_name] = None
    
    # Create comparison summary
    comparison_summary = {
        'timestamp': datetime.now().isoformat(),
        'base_config': base_config.to_dict(),
        'results': results_comparison,
        'experiment_directories': {k: str(v) for k, v in experiment_dirs.items()}
    }
    
    # Save comparison results
    comparison_dir = Path(base_config.experiment.output_dir) / "preprocessing_comparison"
    comparison_dir.mkdir(exist_ok=True)
    
    comparison_path = comparison_dir / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(comparison_path, 'w') as f:
        json.dump(comparison_summary, f, indent=2, default=str)
    
    # Print comparison summary
    print(f"\n{'='*80}")
    print("PREPROCESSING COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    print(f"{'Method':<20} {'AUROC':<8} {'F1':<8} {'Accuracy':<10} {'Status':<10}")
    print("-" * 70)
    
    for method, results in results_comparison.items():
        if results and results['evaluation']['metrics']:
            metrics = results['evaluation']['metrics']
            print(f"{method:<20} {metrics['auroc']:<8.4f} {metrics['f1']:<8.4f} {metrics['accuracy']:<10.4f} {'✅':<10}")
        else:
            print(f"{method:<20} {'N/A':<8} {'N/A':<8} {'N/A':<10} {'❌':<10}")
    
    print(f"\n📁 Comparison results saved to: {comparison_path}")
    
    return comparison_summary

def hyperparameter_sweep(base_config: Config, param_grid: dict) -> dict:
    """Perform hyperparameter sweep"""
    print(f"\n{'='*80}")
    print("HYPERPARAMETER SWEEP")
    print(f"{'='*80}")
    
    sweep_results = {}
    best_config = None
    best_score = 0.0
    
    # Generate parameter combinations
    import itertools
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    combinations = list(itertools.product(*param_values))
    print(f"Total combinations to test: {len(combinations)}")
    
    for i, combination in enumerate(combinations):
        print(f"\n🔄 Testing combination {i+1}/{len(combinations)}")
        
        # Create config for this combination
        config = Config(**base_config.to_dict())
        
        # Update parameters
        for param_name, param_value in zip(param_names, combination):
            if '.' in param_name:
                # Handle nested parameters
                section, param = param_name.split('.')
                setattr(getattr(config, section), param, param_value)
            else:
                setattr(config, param_name, param_value)
        
        # Create experiment name
        param_str = "_".join([f"{p}={v}" for p, v in zip(param_names, combination)])
        exp_name = f"sweep_{param_str}_{datetime.now().strftime('%H%M%S')}"
        
        # Setup experiment directory
        exp_dir = Path(config.experiment.output_dir) / "hyperparameter_sweep" / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        config.experiment.log_dir = str(exp_dir / "logs")
        config.experiment.output_dir = str(exp_dir)
        
        try:
            # Train model
            results = train_single_model(config, exp_dir)
            
            # Track results
            sweep_results[param_str] = {
                'config': config.to_dict(),
                'results': results,
                'auroc': results['evaluation']['metrics']['auroc'],
                'f1': results['evaluation']['metrics']['f1']
            }
            
            # Check if this is the best so far
            score = results['evaluation']['metrics']['auroc']
            if score > best_score:
                best_score = score
                best_config = config
                print(f"🎯 New best score: {score:.4f}")
            
        except Exception as e:
            print(f"❌ Error in combination {param_str}: {e}")
            sweep_results[param_str] = {'error': str(e)}
    
    # Save sweep results
    sweep_dir = Path(base_config.experiment.output_dir) / "hyperparameter_sweep"
    sweep_summary_path = sweep_dir / f"sweep_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(sweep_summary_path, 'w') as f:
        json.dump(sweep_results, f, indent=2, default=str)
    
    print(f"\n🎉 Hyperparameter sweep completed!")
    print(f"📁 Results saved to: {sweep_summary_path}")
    
    if best_config:
        print(f"🏆 Best configuration achieved AUROC: {best_score:.4f}")
    
    return sweep_results

def main():
    parser = argparse.ArgumentParser(description="Pneumonia Classification Training")
    parser.add_argument("--mode", type=str, default="single",
                       choices=["single", "compare", "sweep"],
                       help="Training mode")
    parser.add_argument("--preprocessing", type=str, default="raw",
                       choices=["raw", "histogram_matching", "zscore"],
                       help="Preprocessing method")
    parser.add_argument("--config", type=str, help="Path to config file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--backbone", type=str, default="densenet121", help="Model backbone")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory")
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = Config.load(args.config)
    else:
        if args.preprocessing == "raw":
            config = get_raw_preprocessing_config()
        elif args.preprocessing == "histogram_matching":
            config = get_histogram_matching_config()
        elif args.preprocessing == "zscore":
            config = get_zscore_config()
    
    # Update config with command line arguments
    config.update_paths(args.base_dir)
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.lr
    config.model.backbone = args.backbone
    
    # Run specified mode
    if args.mode == "single":
        # Single model training
        experiment_dir = setup_experiment_directory(config)
        results = train_single_model(config, experiment_dir)
        
    elif args.mode == "compare":
        # Compare preprocessing methods
        results = compare_preprocessing_methods(config)
        
    elif args.mode == "sweep":
        # Hyperparameter sweep
        param_grid = {
            'training.learning_rate': [1e-5, 5e-5, 1e-4, 5e-4],
            'training.weight_decay': [1e-4, 1e-3, 1e-2],
            'training.batch_size': [8, 16, 32]
        }
        results = hyperparameter_sweep(config, param_grid)
    
    print(f"\n🎊 All training completed successfully!")

if __name__ == "__main__":
    main()
