"""
Training module for pneumonia classification
Implements training loop with validation, early stopping, and logging
"""

import os
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

from config import Config
from model import PneumoniaClassifier, save_checkpoint, load_checkpoint

class PneumoniaTrainer:
    """Trainer class for pneumonia classification"""
    
    def __init__(self, 
                 model: PneumoniaClassifier,
                 config: Config,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 test_loader: Optional[DataLoader] = None,
                 log_dir: Optional[str] = None):
        
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Setup logging directory
        if log_dir is not None:
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = Path(config.training.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup device
        self.device = self._setup_device()
        self.model.to(self.device)
        
        # Setup loss function
        self.criterion = self._setup_loss_function()
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup logging
        self.writer = self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
        self.patience_counter = 0
        
        # Create output directories
        self.output_dir = Path(config.experiment.output_dir) / config.experiment.experiment_name
        self.log_dir = Path(config.experiment.log_dir) / config.experiment.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_device(self) -> torch.device:
        """Setup computing device"""
        if self.config.experiment.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(self.config.experiment.device)
        
        print(f"Using device: {device}")
        return device
    
    def _setup_loss_function(self) -> nn.Module:
        """Setup loss function with class balancing"""
        if self.config.training.loss_type == "bce_with_logits":
            pos_weight = None
            
            if self.config.training.class_balancing:
                # Compute positive class weight
                pos_count = 0
                neg_count = 0
                
                for _, labels, _ in self.train_loader:
                    pos_count += labels.sum().item()
                    neg_count += (1 - labels).sum().item()
                
                if self.config.training.pos_weight is not None:
                    pos_weight = torch.tensor(self.config.training.pos_weight)
                else:
                    pos_weight = torch.tensor(neg_count / pos_count) if pos_count > 0 else torch.tensor(1.0)
                
                print(f"Class balancing: pos_weight = {pos_weight.item():.3f}")
            
            return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        elif self.config.training.loss_type == "bce":
            return nn.BCELoss()
        
        elif self.config.training.loss_type == "focal":
            return FocalLoss(alpha=0.25, gamma=2.0)
        
        else:
            raise ValueError(f"Unknown loss type: {self.config.training.loss_type}")
    
    def _setup_optimizer(self) -> torch.optim.Optimizer:
        """Setup optimizer"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )
    
    def _setup_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Setup learning rate scheduler"""
        if not self.config.training.use_scheduler:
            return None
        
        if self.config.training.scheduler_type == "reduce_on_plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=self.config.training.scheduler_factor,
                patience=self.config.training.scheduler_patience,
                min_lr=self.config.training.min_lr
            )
        
        elif self.config.training.scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.training.num_epochs,
                eta_min=self.config.training.min_lr
            )
        
        elif self.config.training.scheduler_type == "step":
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.training.num_epochs // 3,
                gamma=self.config.training.scheduler_factor
            )
        
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.training.scheduler_type}")
    
    def _setup_logging(self) -> SummaryWriter:
        """Setup TensorBoard logging"""
        log_dir = self.log_dir / "tensorboard"
        return SummaryWriter(log_dir=str(log_dir))
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        num_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch+1}/{self.config.training.num_epochs}")
        
        for batch_idx, (images, labels, _) in enumerate(pbar):
            # Move to device
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs.squeeze(), labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.training.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.training.grad_clip_norm
                )
            
            self.optimizer.step()
            
            # Update metrics
            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)
            
            # Update progress bar
            avg_loss = running_loss / num_samples
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})
            
            # Log batch metrics
            if batch_idx % self.config.experiment.log_frequency == 0:
                self.writer.add_scalar(
                    "Loss/Train_Batch", 
                    loss.item(), 
                    self.current_epoch * len(self.train_loader) + batch_idx
                )
        
        epoch_loss = running_loss / num_samples
        return {"loss": epoch_loss}
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        all_outputs = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs.squeeze(), labels)
                
                # Store predictions and labels
                running_loss += loss.item() * images.size(0)
                all_outputs.extend(torch.sigmoid(outputs.squeeze()).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Compute metrics
        val_loss = running_loss / len(self.val_loader.dataset)
        metrics = self._compute_metrics(np.array(all_outputs), np.array(all_labels))
        metrics["loss"] = val_loss
        
        return metrics
    
    def _compute_metrics(self, outputs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Compute validation metrics"""
        from sklearn.metrics import (
            roc_auc_score, f1_score, precision_score, recall_score, 
            accuracy_score, confusion_matrix
        )
        
        # Binary predictions using 0.5 threshold
        predictions = (outputs > 0.5).astype(int)
        
        metrics = {}
        
        # AUROC
        if len(np.unique(labels)) > 1:  # Need both classes for AUROC
            metrics["auroc"] = roc_auc_score(labels, outputs)
        else:
            metrics["auroc"] = 0.0
        
        # Classification metrics
        metrics["f1"] = f1_score(labels, predictions)
        metrics["precision"] = precision_score(labels, predictions, zero_division=0)
        metrics["recall"] = recall_score(labels, predictions, zero_division=0)
        metrics["accuracy"] = accuracy_score(labels, predictions)
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        metrics["specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return metrics
    
    def train(self) -> Dict[str, List[float]]:
        """Full training loop"""
        print(f"Starting training for {self.config.training.num_epochs} epochs")
        print(f"Model parameters: {self.model.get_trainable_parameters()['trainable']:,}")
        
        start_time = time.time()
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics["loss"])
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            self.val_losses.append(val_metrics["loss"])
            self.val_metrics.append(val_metrics)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                if self.config.training.scheduler_type == "reduce_on_plateau":
                    self.scheduler.step(val_metrics["loss"])
                else:
                    self.scheduler.step()
            
            # Log metrics
            self._log_epoch_metrics(train_metrics, val_metrics, epoch)
            
            # Check for improvement
            current_metric = val_metrics.get("auroc", val_metrics.get("f1", 0.0))
            
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.patience_counter = 0
                
                # Save best model
                if self.config.experiment.save_best_model:
                    self._save_model("best_model.pth", epoch, val_metrics)
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint
            if (epoch + 1) % self.config.experiment.save_checkpoint_frequency == 0:
                self._save_model(f"checkpoint_epoch_{epoch+1}.pth", epoch, val_metrics)
            
            # Print epoch summary
            print(f"Epoch {epoch+1}/{self.config.training.num_epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val AUROC: {val_metrics.get('auroc', 0.0):.4f}")
            print(f"  Val F1: {val_metrics.get('f1', 0.0):.4f}")
            print(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            print()
        
        # Save final model
        if self.config.experiment.save_last_model:
            self._save_model("last_model.pth", self.current_epoch, val_metrics)
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_metrics": self.val_metrics
        }
    
    def _log_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict, epoch: int):
        """Log metrics to TensorBoard"""
        # Loss
        self.writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
        self.writer.add_scalar("Loss/Validation", val_metrics["loss"], epoch)
        
        # Validation metrics
        for metric_name, metric_value in val_metrics.items():
            if metric_name != "loss":
                self.writer.add_scalar(f"Metrics/{metric_name}", metric_value, epoch)
        
        # Learning rate
        self.writer.add_scalar("Learning_Rate", self.optimizer.param_groups[0]['lr'], epoch)
    
    def _save_model(self, filename: str, epoch: int, metrics: Dict[str, float]):
        """Save model checkpoint"""
        checkpoint_path = self.output_dir / filename
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            metrics=metrics,
            checkpoint_path=str(checkpoint_path)
        )
        print(f"Model saved: {checkpoint_path}")
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, self.train_losses, label="Train Loss")
        axes[0, 0].plot(epochs, self.val_losses, label="Validation Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # AUROC
        auroc_scores = [m.get("auroc", 0.0) for m in self.val_metrics]
        axes[0, 1].plot(epochs, auroc_scores, label="Validation AUROC", color='orange')
        axes[0, 1].set_title("Validation AUROC")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("AUROC")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 Score
        f1_scores = [m.get("f1", 0.0) for m in self.val_metrics]
        axes[1, 0].plot(epochs, f1_scores, label="Validation F1", color='green')
        axes[1, 0].set_title("Validation F1 Score")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("F1 Score")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Multiple metrics
        precision_scores = [m.get("precision", 0.0) for m in self.val_metrics]
        recall_scores = [m.get("recall", 0.0) for m in self.val_metrics]
        
        axes[1, 1].plot(epochs, precision_scores, label="Precision", alpha=0.7)
        axes[1, 1].plot(epochs, recall_scores, label="Recall", alpha=0.7)
        axes[1, 1].plot(epochs, f1_scores, label="F1", alpha=0.7)
        axes[1, 1].set_title("Validation Metrics")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Score")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_training_history(self):
        """Save training history to JSON"""
        history = {
            "config": {
                "experiment_name": self.config.experiment.experiment_name,
                "model_config": self.config.model.__dict__,
                "training_config": self.config.training.__dict__,
                "data_config": self.config.data.__dict__
            },
            "training_history": {
                "train_losses": self.train_losses,
                "val_losses": self.val_losses,
                "val_metrics": self.val_metrics
            },
            "best_metrics": {
                "best_metric_value": self.best_metric,
                "total_epochs": len(self.train_losses)
            }
        }
        
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2, default=str)
        
        print(f"Training history saved: {history_path}")

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

if __name__ == "__main__":
    # Test training setup
    from config import get_raw_preprocessing_config
    from data_handler import ChestXrayDataManager
    from model import create_model
    
    # Setup
    config = get_raw_preprocessing_config()
    config.update_paths(".")
    config.training.num_epochs = 2  # Quick test
    
    # Create data
    data_manager = ChestXrayDataManager(config)
    train_loader, val_loader, test_loader = data_manager.setup_data()
    
    # Create model
    model = create_model(config.model)
    
    # Create trainer
    trainer = PneumoniaTrainer(model, config, train_loader, val_loader, test_loader)
    
    print("Training setup successful!")
    print(f"Device: {trainer.device}")
    print(f"Model parameters: {model.get_trainable_parameters()['trainable']:,}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Test one epoch
    print("\nTesting one training epoch...")
    train_metrics = trainer.train_epoch()
    print(f"Train loss: {train_metrics['loss']:.4f}")
    
    val_metrics = trainer.validate_epoch()
    print(f"Val loss: {val_metrics['loss']:.4f}")
    print(f"Val AUROC: {val_metrics.get('auroc', 0.0):.4f}")
