"""
Model architecture module for pneumonia classification
Implements DenseNet-121 baseline with configurable options
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Dict, Any
import numpy as np

from config import ModelConfig

class PneumoniaClassifier(nn.Module):
    """DenseNet-121 based pneumonia classifier"""
    
    def __init__(self, config: ModelConfig):
        super(PneumoniaClassifier, self).__init__()
        self.config = config
        
        # Load backbone
        self.backbone = self._create_backbone()
        
        # Get feature dimension
        self.feature_dim = self._get_feature_dim()
        
        # Create classifier head
        self.classifier = self._create_classifier_head()
        
        # Initialize weights
        self._initialize_weights()
    
    def _create_backbone(self) -> nn.Module:
        """Create backbone network"""
        if self.config.backbone == "densenet121":
            backbone = models.densenet121(pretrained=self.config.pretrained)
            # Remove classifier layer
            backbone.classifier = nn.Identity()
            return backbone
        elif self.config.backbone == "densenet169":
            backbone = models.densenet169(pretrained=self.config.pretrained)
            backbone.classifier = nn.Identity()
            return backbone
        elif self.config.backbone == "densenet201":
            backbone = models.densenet201(pretrained=self.config.pretrained)
            backbone.classifier = nn.Identity()
            return backbone
        else:
            raise ValueError(f"Unsupported backbone: {self.config.backbone}")
    
    def _get_feature_dim(self) -> int:
        """Get feature dimension from backbone"""
        if "densenet121" in self.config.backbone:
            return 1024
        elif "densenet169" in self.config.backbone:
            return 1664
        elif "densenet201" in self.config.backbone:
            return 1920
        else:
            # Dynamically compute feature dimension
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                features = self.backbone(dummy_input)
                return features.shape[1]
    
    def _create_classifier_head(self) -> nn.Module:
        """Create classifier head"""
        layers = []
        
        # Global average pooling is handled in forward pass
        
        # Dropout
        if self.config.dropout_rate > 0:
            layers.append(nn.Dropout(self.config.dropout_rate))
        
        # Final classifier layer
        layers.append(nn.Linear(self.feature_dim, self.config.num_classes))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Extract features
        features = self.backbone(x)
        
        # Global average pooling if needed
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        # Classify
        logits = self.classifier(features)
        
        return logits
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification"""
        features = self.backbone(x)
        
        if len(features.shape) > 2:
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        return features
    
    def freeze_backbone(self, num_layers: int = None):
        """Freeze backbone layers"""
        if num_layers is None:
            # Freeze entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
        else:
            # Freeze first num_layers
            layer_count = 0
            for name, module in self.backbone.named_modules():
                if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)):
                    if layer_count < num_layers:
                        for param in module.parameters():
                            param.requires_grad = False
                    layer_count += 1
    
    def unfreeze_backbone(self):
        """Unfreeze all backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def get_trainable_parameters(self) -> Dict[str, int]:
        """Get number of trainable parameters"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        classifier_trainable = sum(p.numel() for p in self.classifier.parameters() if p.requires_grad)
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'backbone_total': backbone_params,
            'backbone_trainable': backbone_trainable,
            'classifier_total': classifier_params,
            'classifier_trainable': classifier_trainable
        }

class EnsembleClassifier(nn.Module):
    """Ensemble of multiple classifiers for improved performance"""
    
    def __init__(self, models: list, ensemble_method: str = "average"):
        super(EnsembleClassifier, self).__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ensemble"""
        outputs = []
        
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Combine outputs
        stacked_outputs = torch.stack(outputs, dim=0)
        
        if self.ensemble_method == "average":
            return torch.mean(stacked_outputs, dim=0)
        elif self.ensemble_method == "max":
            return torch.max(stacked_outputs, dim=0)[0]
        elif self.ensemble_method == "weighted":
            # Simple equal weighting for now
            weights = torch.ones(len(self.models)) / len(self.models)
            weights = weights.to(x.device).view(-1, 1, 1)
            return torch.sum(stacked_outputs * weights, dim=0)
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

class GradCAM:
    """Gradient-weighted Class Activation Mapping for interpretability"""
    
    def __init__(self, model: PneumoniaClassifier, target_layer: str):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks"""
        target_module = dict(self.model.named_modules())[self.target_layer]
        
        def forward_hook(module, input, output):
            self.activations = output
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
        
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> np.ndarray:
        """Generate class activation map"""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1)
        
        # Backward pass
        self.model.zero_grad()
        
        # Create one-hot encoding for target class
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1.0
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:])  # [H, W]
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam / cam.max() if cam.max() > 0 else cam
        
        return cam.detach().cpu().numpy()
    
    def visualize_cam(self, input_tensor: torch.Tensor, class_idx: int = None) -> tuple:
        """Generate and return CAM visualization"""
        import cv2
        import matplotlib.pyplot as plt
        
        # Generate CAM
        cam = self.generate_cam(input_tensor, class_idx)
        
        # Convert CAM to float32 before OpenCV operations to avoid CV_64F error
        cam = cam.astype(np.float32)
        
        # Resize CAM to input size
        input_size = input_tensor.shape[-2:]
        cam_resized = cv2.resize(cam, input_size)
        
        # Convert input tensor to numpy for visualization
        input_np = input_tensor[0].detach().cpu().numpy()
        
        # Denormalize input image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        input_np = input_np.transpose(1, 2, 0)
        input_np = input_np * std + mean
        input_np = np.clip(input_np, 0, 1)
        
        # Ensure input_np is float32 before OpenCV operations
        input_np = input_np.astype(np.float32)
        
        # Convert to grayscale if needed
        if input_np.shape[2] == 3:
            input_gray = cv2.cvtColor(input_np, cv2.COLOR_RGB2GRAY)
        else:
            input_gray = input_np[:, :, 0]
        
        return input_gray, cam_resized

def create_model(config: ModelConfig) -> PneumoniaClassifier:
    """Factory function to create model"""
    model = PneumoniaClassifier(config)
    
    # Apply freezing if specified
    if config.freeze_backbone:
        model.freeze_backbone()
    elif config.freeze_layers > 0:
        model.freeze_backbone(config.freeze_layers)
    
    return model

def load_checkpoint(model: PneumoniaClassifier, checkpoint_path: str) -> Dict[str, Any]:
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return checkpoint

def save_checkpoint(model: PneumoniaClassifier, 
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
                   epoch: int,
                   metrics: Dict[str, float],
                   checkpoint_path: str):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)

if __name__ == "__main__":
    # Test model creation
    from config import ModelConfig
    
    config = ModelConfig()
    model = create_model(config)
    
    # Print model info
    param_info = model.get_trainable_parameters()
    print("Model Parameter Information:")
    for key, value in param_info.items():
        print(f"  {key}: {value:,}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
        print(f"\nInput shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    # Test GradCAM
    print("\nTesting GradCAM...")
    gradcam = GradCAM(model, "backbone.features.denseblock4")
    
    # Single input for GradCAM
    single_input = dummy_input[:1]
    single_input.requires_grad_()
    
    cam = gradcam.generate_cam(single_input)
    print(f"CAM shape: {cam.shape}")
    print(f"CAM range: [{cam.min():.3f}, {cam.max():.3f}]")
