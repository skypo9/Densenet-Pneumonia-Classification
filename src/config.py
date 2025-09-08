"""
Configuration file for pneumonia classification baseline model
Supports reproducible experiments with different preprocessing and hyperparameters
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import yaml

@dataclass
class DataConfig:
    """Data-related configuration"""
    # Dataset paths
    data_root: str = "data"
    dataset_name: str = "chest_xray_pneumonia"
    
    # Data splits (patient-level)
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Image preprocessing
    image_size: Tuple[int, int] = (224, 224)
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    
    # Preprocessing pipeline type
    preprocessing_type: str = "raw"  # "raw", "histogram_matching", "z_score"
    histogram_reference_percentile: float = 50.0
    z_score_clip_range: Tuple[float, float] = (-3.0, 3.0)
    
    # Data augmentation
    use_augmentation: bool = True
    rotation_range: float = 10.0
    horizontal_flip_prob: float = 0.5
    brightness_factor: float = 0.1
    contrast_factor: float = 0.1

@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Base architecture
    backbone: str = "densenet121"
    pretrained: bool = True
    num_classes: int = 1  # Binary classification (pneumonia vs normal)
    dropout_rate: float = 0.3
    
    # Feature extraction
    freeze_backbone: bool = False
    freeze_layers: int = 0  # Number of layers to freeze from start

@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Optimization
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 50
    patience: int = 10  # Early stopping patience
    
    # Loss function
    loss_type: str = "bce_with_logits"
    class_balancing: bool = True
    pos_weight: Optional[float] = None  # Will be calculated from data if None
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "step"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    min_lr: float = 1e-7
    
    # Gradient clipping
    grad_clip_norm: Optional[float] = 1.0

@dataclass
class EvaluationConfig:
    """Evaluation and metrics configuration"""
    # Metrics to compute
    metrics: List[str] = field(default_factory=lambda: [
        "auroc", "f1", "precision", "recall", "accuracy", "specificity"
    ])
    
    # Calibration
    calibration_bins: int = 10
    
    # Interpretability
    generate_gradcam: bool = True
    gradcam_layer: str = "backbone.features.denseblock4"  # DenseNet-121 specific
    num_gradcam_samples: int = 20
    
    # Threshold optimization
    optimize_threshold: bool = True
    threshold_metric: str = "f1"  # Metric to optimize threshold for

@dataclass
class ExperimentConfig:
    """Overall experiment configuration"""
    # Experiment metadata
    experiment_name: str = "pneumonia_baseline"
    description: str = "DenseNet-121 baseline for pneumonia classification"
    
    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True
    
    # Hardware
    device: str = "auto"  # "auto", "cpu", "cuda"
    num_workers: int = 0  # Set to 0 for Windows compatibility
    pin_memory: bool = False  # Disable for Windows CPU
    
    # Logging and saving
    log_dir: str = "logs"
    output_dir: str = "outputs"
    save_best_model: bool = True
    save_last_model: bool = True
    log_frequency: int = 10  # Log every N batches
    
    # Checkpointing
    resume_from_checkpoint: Optional[str] = None
    save_checkpoint_frequency: int = 5  # Save checkpoint every N epochs

@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    
    def to_dict(self):
        """Convert configuration to dictionary"""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'experiment': self.experiment.__dict__
        }
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert dataclass to dict
        config_dict = self.to_dict()
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(**config_dict['data']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            evaluation=EvaluationConfig(**config_dict['evaluation']),
            experiment=ExperimentConfig(**config_dict['experiment'])
        )
    
    def update_paths(self, base_dir: str):
        """Update relative paths to be relative to base_dir"""
        base_path = Path(base_dir)
        
        # Update data paths
        if not Path(self.data.data_root).is_absolute():
            self.data.data_root = str(base_path / self.data.data_root)
        
        # Update logging and output paths
        if not Path(self.experiment.log_dir).is_absolute():
            self.experiment.log_dir = str(base_path / self.experiment.log_dir)
        
        if not Path(self.experiment.output_dir).is_absolute():
            self.experiment.output_dir = str(base_path / self.experiment.output_dir)

# Predefined configurations for different experiments

def get_raw_preprocessing_config() -> Config:
    """Configuration for raw image preprocessing"""
    config = Config()
    config.data.preprocessing_type = "raw"
    config.experiment.experiment_name = "pneumonia_baseline_raw"
    config.experiment.description = "DenseNet-121 baseline with raw preprocessing"
    return config

def get_histogram_matching_config() -> Config:
    """Configuration for histogram matching preprocessing"""
    config = Config()
    config.data.preprocessing_type = "histogram_matching"
    config.experiment.experiment_name = "pneumonia_baseline_histogram"
    config.experiment.description = "DenseNet-121 baseline with histogram matching"
    return config

def get_zscore_config() -> Config:
    """Configuration for z-score normalization preprocessing"""
    config = Config()
    config.data.preprocessing_type = "z_score"
    config.experiment.experiment_name = "pneumonia_baseline_zscore"
    config.experiment.description = "DenseNet-121 baseline with z-score normalization"
    return config

def get_hyperparameter_sweep_configs() -> List[Config]:
    """Get configurations for hyperparameter sweep"""
    base_config = get_raw_preprocessing_config()
    
    configs = []
    
    # Learning rate sweep
    for lr in [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]:
        config = Config()
        config.__dict__.update(base_config.__dict__)
        config.training.learning_rate = lr
        config.experiment.experiment_name = f"pneumonia_lr_{lr:.0e}"
        configs.append(config)
    
    # Batch size sweep
    for bs in [16, 32, 64, 128]:
        config = Config()
        config.__dict__.update(base_config.__dict__)
        config.training.batch_size = bs
        config.experiment.experiment_name = f"pneumonia_bs_{bs}"
        configs.append(config)
    
    # Weight decay sweep
    for wd in [1e-5, 1e-4, 1e-3, 1e-2]:
        config = Config()
        config.__dict__.update(base_config.__dict__)
        config.training.weight_decay = wd
        config.experiment.experiment_name = f"pneumonia_wd_{wd:.0e}"
        configs.append(config)
    
    return configs

if __name__ == "__main__":
    # Create and save default configurations
    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)
    
    # Save default configurations
    get_raw_preprocessing_config().save("configs/raw_preprocessing.yaml")
    get_histogram_matching_config().save("configs/histogram_matching.yaml")
    get_zscore_config().save("configs/zscore_preprocessing.yaml")
    
    print("Default configuration files created in configs/ directory")
    print("Available configurations:")
    print("- raw_preprocessing.yaml")
    print("- histogram_matching.yaml") 
    print("- zscore_preprocessing.yaml")
