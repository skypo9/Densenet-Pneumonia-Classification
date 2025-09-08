"""
Evaluation module for pneumonia classification
Implements comprehensive metrics, calibration, and interpretability analysis
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score, confusion_matrix,
    classification_report, brier_score_loss
)
from sklearn.calibration import calibration_curve
import cv2

from config import Config
from model import PneumoniaClassifier, GradCAM

class PneumoniaEvaluator:
    """Comprehensive evaluator for pneumonia classification"""
    
    def __init__(self, model: PneumoniaClassifier, config: Config, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        
        # Setup GradCAM
        if config.evaluation.generate_gradcam:
            self.gradcam = GradCAM(self.model, config.evaluation.gradcam_layer)
        else:
            self.gradcam = None
        
        # Results storage
        self.results = {}
    
    def evaluate(self, test_loader: DataLoader, save_dir: Optional[str] = None) -> Dict:
        """Comprehensive evaluation on test set"""
        print("Starting comprehensive evaluation...")
        
        # Get predictions
        predictions, probabilities, labels, paths = self._get_predictions(test_loader)
        
        # Compute metrics
        metrics = self._compute_comprehensive_metrics(probabilities, labels, predictions)
        
        # Calibration analysis
        calibration_metrics = self._evaluate_calibration(probabilities, labels)
        
        # Threshold optimization
        optimal_threshold, threshold_metrics = self._optimize_threshold(probabilities, labels)
        
        # Generate visualizations
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            
            self._plot_roc_curve(probabilities, labels, save_path)
            self._plot_precision_recall_curve(probabilities, labels, save_path)
            self._plot_confusion_matrix(predictions, labels, save_path)
            self._plot_calibration_curve(probabilities, labels, save_path)
            self._plot_prediction_histogram(probabilities, labels, save_path)
            
            # Generate GradCAM visualizations
            if self.gradcam:
                self._generate_gradcam_visualizations(test_loader, save_path)
        
        # Compile all results
        self.results = {
            'metrics': metrics,
            'calibration': calibration_metrics,
            'optimal_threshold': optimal_threshold,
            'threshold_metrics': threshold_metrics,
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist(),
            'labels': labels.tolist(),
            'paths': paths
        }
        
        # Print summary
        self._print_evaluation_summary()
        
        return self.results
    
    def _get_predictions(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Get model predictions on test set"""
        all_probabilities = []
        all_labels = []
        all_paths = []
        
        print("Getting model predictions...")
        
        with torch.no_grad():
            for images, labels, paths in test_loader:
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.sigmoid(outputs.squeeze()).cpu().numpy()
                
                all_probabilities.extend(probabilities)
                all_labels.extend(labels.numpy())
                all_paths.extend(paths)
        
        probabilities = np.array(all_probabilities)
        labels = np.array(all_labels)
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities, labels, all_paths
    
    def _compute_comprehensive_metrics(self, probabilities: np.ndarray, 
                                     labels: np.ndarray, 
                                     predictions: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive classification metrics"""
        metrics = {}
        
        # Binary classification metrics
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['precision'] = precision_score(labels, predictions, zero_division=0)
        metrics['recall'] = recall_score(labels, predictions, zero_division=0)
        metrics['f1'] = f1_score(labels, predictions)
        
        # ROC metrics
        if len(np.unique(labels)) > 1:
            metrics['auroc'] = roc_auc_score(labels, probabilities)
            metrics['auprc'] = average_precision_score(labels, probabilities)
        else:
            metrics['auroc'] = 0.0
            metrics['auprc'] = 0.0
        
        # Confusion matrix derived metrics
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['ppv'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # Positive Predictive Value
        metrics['npv'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
        
        # Balanced metrics
        metrics['balanced_accuracy'] = (metrics['sensitivity'] + metrics['specificity']) / 2
        metrics['f1_macro'] = (2 * metrics['ppv'] * metrics['sensitivity']) / (metrics['ppv'] + metrics['sensitivity']) if (metrics['ppv'] + metrics['sensitivity']) > 0 else 0.0
        
        # Brier score (calibration)
        metrics['brier_score'] = brier_score_loss(labels, probabilities)
        
        return metrics
    
    def _evaluate_calibration(self, probabilities: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """Evaluate model calibration"""
        n_bins = self.config.evaluation.calibration_bins
        
        # Calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, probabilities, n_bins=n_bins, strategy='uniform'
        )
        
        # Expected Calibration Error (ECE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        # Maximum Calibration Error (MCE)
        mce = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            
            if in_bin.sum() > 0:
                accuracy_in_bin = labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
        
        return {
            'ece': ece,
            'mce': mce,
            'fraction_of_positives': fraction_of_positives.tolist(),
            'mean_predicted_value': mean_predicted_value.tolist()
        }
    
    def _optimize_threshold(self, probabilities: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict[str, float]]:
        """Find optimal threshold based on specified metric"""
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_threshold = 0.5
        best_score = 0.0
        threshold_scores = []
        
        target_metric = self.config.evaluation.threshold_metric
        
        for threshold in thresholds:
            predictions = (probabilities > threshold).astype(int)
            
            if target_metric == "f1":
                score = f1_score(labels, predictions)
            elif target_metric == "balanced_accuracy":
                tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                score = (sensitivity + specificity) / 2
            elif target_metric == "accuracy":
                score = accuracy_score(labels, predictions)
            else:
                score = f1_score(labels, predictions)
            
            threshold_scores.append(score)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        # Compute metrics at optimal threshold
        optimal_predictions = (probabilities > best_threshold).astype(int)
        optimal_metrics = self._compute_comprehensive_metrics(probabilities, labels, optimal_predictions)
        optimal_metrics['threshold'] = best_threshold
        
        return best_threshold, optimal_metrics
    
    def _plot_roc_curve(self, probabilities: np.ndarray, labels: np.ndarray, save_path: Path):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(labels, probabilities)
        auc = roc_auc_score(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_precision_recall_curve(self, probabilities: np.ndarray, labels: np.ndarray, save_path: Path):
        """Plot Precision-Recall curve"""
        precision, recall, _ = precision_recall_curve(labels, probabilities)
        auprc = average_precision_score(labels, probabilities)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {auprc:.3f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrix(self, predictions: np.ndarray, labels: np.ndarray, save_path: Path):
        """Plot confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Normal', 'Pneumonia'], 
                   yticklabels=['Normal', 'Pneumonia'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(save_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_calibration_curve(self, probabilities: np.ndarray, labels: np.ndarray, save_path: Path):
        """Plot calibration curve"""
        fraction_of_positives, mean_predicted_value = calibration_curve(
            labels, probabilities, n_bins=self.config.evaluation.calibration_bins
        )
        
        plt.figure(figsize=(8, 6))
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        plt.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        plt.xlabel('Mean Predicted Probability')
        plt.ylabel('Fraction of Positives')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(save_path / 'calibration_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_prediction_histogram(self, probabilities: np.ndarray, labels: np.ndarray, save_path: Path):
        """Plot histogram of predictions by class"""
        plt.figure(figsize=(10, 6))
        
        plt.subplot(1, 2, 1)
        plt.hist(probabilities[labels == 0], bins=30, alpha=0.7, label='Normal', color='blue')
        plt.hist(probabilities[labels == 1], bins=30, alpha=0.7, label='Pneumonia', color='red')
        plt.xlabel('Predicted Probability')
        plt.ylabel('Count')
        plt.title('Prediction Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # Box plot
        data_to_plot = [probabilities[labels == 0], probabilities[labels == 1]]
        plt.boxplot(data_to_plot, labels=['Normal', 'Pneumonia'])
        plt.ylabel('Predicted Probability')
        plt.title('Prediction Distribution by Class')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'prediction_histogram.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_gradcam_visualizations(self, test_loader: DataLoader, save_path: Path):
        """Generate GradCAM visualizations for sample images"""
        if not self.gradcam:
            return
        
        gradcam_dir = save_path / "gradcam"
        gradcam_dir.mkdir(exist_ok=True)
        
        sample_count = 0
        max_samples = self.config.evaluation.num_gradcam_samples
        
        print(f"Generating GradCAM visualizations for {max_samples} samples...")
        
        # Don't use no_grad() for GradCAM generation since we need gradients
        for images, labels, paths in test_loader:
            if sample_count >= max_samples:
                break
            
            images = images.to(self.device)
            
            for i in range(min(images.size(0), max_samples - sample_count)):
                image = images[i:i+1]
                label = labels[i].item()
                path = paths[i]
                
                # Ensure gradients are enabled for this image
                image = image.clone().detach().requires_grad_(True)
                
                # Get prediction (no_grad not needed here since we want gradients)
                output = self.model(image)
                probability = torch.sigmoid(output).item()
                prediction = int(probability > 0.5)
                
                # Generate GradCAM
                try:
                    original_img, cam = self.gradcam.visualize_cam(image)
                    
                    # Create visualization
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # Original image
                    axes[0].imshow(original_img, cmap='gray')
                    axes[0].set_title('Original Image')
                    axes[0].axis('off')
                    
                    # GradCAM
                    axes[1].imshow(cam, cmap='jet', alpha=0.8)
                    axes[1].set_title('GradCAM')
                    axes[1].axis('off')
                    
                    # Overlay
                    axes[2].imshow(original_img, cmap='gray')
                    axes[2].imshow(cam, cmap='jet', alpha=0.4)
                    axes[2].set_title('Overlay')
                    axes[2].axis('off')
                    
                    # Add prediction info
                    true_label = "Pneumonia" if label == 1 else "Normal"
                    pred_label = "Pneumonia" if prediction == 1 else "Normal"
                    title = f'True: {true_label}, Pred: {pred_label} (p={probability:.3f})'
                    fig.suptitle(title)
                    
                    # Save
                    filename = f"gradcam_{sample_count:03d}_{Path(path).stem}.png"
                    plt.savefig(gradcam_dir / filename, dpi=200, bbox_inches='tight')
                    plt.close()
                    
                    sample_count += 1
                    
                except Exception as e:
                    print(f"Error generating GradCAM for sample {sample_count}: {e}")
                    continue
            
            if sample_count >= max_samples:
                break
        
        print(f"Generated {sample_count} GradCAM visualizations in {gradcam_dir}")
    
    def _print_evaluation_summary(self):
        """Print comprehensive evaluation summary"""
        metrics = self.results['metrics']
        calibration = self.results['calibration']
        threshold_metrics = self.results['threshold_metrics']
        
        print("\n" + "="*50)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*50)
        
        print("\n📊 CLASSIFICATION METRICS:")
        print(f"  Accuracy:      {metrics['accuracy']:.4f}")
        print(f"  Precision:     {metrics['precision']:.4f}")
        print(f"  Recall:        {metrics['recall']:.4f}")
        print(f"  F1 Score:      {metrics['f1']:.4f}")
        print(f"  Specificity:   {metrics['specificity']:.4f}")
        print(f"  Sensitivity:   {metrics['sensitivity']:.4f}")
        
        print(f"\n🎯 DISCRIMINATION METRICS:")
        print(f"  AUROC:         {metrics['auroc']:.4f}")
        print(f"  AUPRC:         {metrics['auprc']:.4f}")
        print(f"  Balanced Acc:  {metrics['balanced_accuracy']:.4f}")
        
        print(f"\n📏 CALIBRATION METRICS:")
        print(f"  Brier Score:   {metrics['brier_score']:.4f}")
        print(f"  ECE:           {calibration['ece']:.4f}")
        print(f"  MCE:           {calibration['mce']:.4f}")
        
        print(f"\n🎛️ OPTIMAL THRESHOLD:")
        print(f"  Threshold:     {threshold_metrics['threshold']:.3f}")
        print(f"  F1 at optimal: {threshold_metrics['f1']:.4f}")
        print(f"  Acc at optimal:{threshold_metrics['accuracy']:.4f}")
        
        print("\n" + "="*50)
    
    def save_results(self, save_path: str):
        """Save evaluation results to file"""
        results_path = Path(save_path)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        import json
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Save detailed classification report
        predictions = np.array(self.results['predictions'])
        labels = np.array(self.results['labels'])
        
        report = classification_report(
            labels, predictions, 
            target_names=['Normal', 'Pneumonia'],
            output_dict=True
        )
        
        report_path = results_path.parent / "classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Results saved to {results_path}")
        print(f"Classification report saved to {report_path}")

def compare_preprocessing_methods(model_paths: List[str], 
                                test_loader: DataLoader, 
                                config: Config,
                                device: torch.device) -> Dict:
    """Compare different preprocessing methods"""
    print("Comparing preprocessing methods...")
    
    results_comparison = {}
    
    for model_path in model_paths:
        # Load model
        model = PneumoniaClassifier(config.model)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Extract preprocessing method from path
        preprocessing_method = Path(model_path).parent.name
        
        # Evaluate
        evaluator = PneumoniaEvaluator(model, config, device)
        results = evaluator.evaluate(test_loader)
        
        results_comparison[preprocessing_method] = results['metrics']
    
    # Create comparison visualization
    metrics_df = pd.DataFrame(results_comparison).T
    
    plt.figure(figsize=(12, 8))
    metrics_to_plot = ['auroc', 'f1', 'accuracy', 'precision', 'recall']
    
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    for i, (method, values) in enumerate(results_comparison.items()):
        offset = (i - len(results_comparison) // 2) * width
        plt.bar(x + offset, [values[m] for m in metrics_to_plot], 
               width, label=method, alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Preprocessing Methods Comparison')
    plt.xticks(x, metrics_to_plot)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('preprocessing_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return results_comparison

if __name__ == "__main__":
    # Test evaluation
    from config import get_raw_preprocessing_config
    from data_handler import ChestXrayDataManager
    from model import create_model
    
    # Setup
    config = get_raw_preprocessing_config()
    config.update_paths(".")
    
    # Create test data
    data_manager = ChestXrayDataManager(config)
    _, _, test_loader = data_manager.setup_data()
    
    # Create model
    model = create_model(config.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create evaluator
    evaluator = PneumoniaEvaluator(model, config, device)
    
    # Test evaluation
    print("Testing evaluation...")
    results = evaluator.evaluate(test_loader, save_dir="test_evaluation_output")
    
    print("Evaluation test completed!")
    print(f"Test AUROC: {results['metrics']['auroc']:.4f}")
    print(f"Test F1: {results['metrics']['f1']:.4f}")
