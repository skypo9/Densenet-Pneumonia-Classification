"""
Data handling module for pneumonia classification
Implements patient-level splits and multiple preprocessing pipelines
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from skimage import exposure, filters
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config, DataConfig

class PneumoniaDataset(Dataset):
    """Custom dataset for pneumonia classification with preprocessing options"""
    
    def __init__(self, 
                 image_paths: List[str], 
                 labels: List[int],
                 config: DataConfig,
                 transform: Optional[Callable] = None,
                 preprocessing_stats: Optional[Dict] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.config = config
        self.transform = transform
        self.preprocessing_stats = preprocessing_stats or {}
        
        # Set up preprocessing pipeline
        self.preprocessing_fn = self._get_preprocessing_function()
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Apply preprocessing
        image = self.preprocessing_fn(image)
        
        # Convert to 3-channel (RGB) for pretrained models
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Convert to PIL Image for transforms
        image = Image.fromarray(image.astype(np.uint8))
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        
        return image, label, image_path
    
    def _get_preprocessing_function(self):
        """Get preprocessing function based on config"""
        if self.config.preprocessing_type == "raw":
            return self._raw_preprocessing
        elif self.config.preprocessing_type == "histogram_matching":
            return self._histogram_matching
        elif self.config.preprocessing_type == "z_score":
            return self._zscore_normalization
        else:
            raise ValueError(f"Unknown preprocessing type: {self.config.preprocessing_type}")
    
    def _raw_preprocessing(self, image: np.ndarray) -> np.ndarray:
        """Raw preprocessing - just resize and basic normalization"""
        # Resize
        image = cv2.resize(image, self.config.image_size)
        
        # Normalize to [0, 255]
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        return image
    
    def _histogram_matching(self, image: np.ndarray) -> np.ndarray:
        """Histogram matching preprocessing"""
        # Resize
        image = cv2.resize(image, self.config.image_size)
        
        # Get reference histogram if available
        if 'reference_histogram' in self.preprocessing_stats:
            reference = self.preprocessing_stats['reference_histogram']
            # Ensure both images are uint8 for histogram matching
            image = image.astype(np.uint8)
            reference = reference.astype(np.uint8)
            # Match histogram to reference - updated for newer scikit-image versions
            image = exposure.match_histograms(image, reference, channel_axis=None)
            # Ensure result is uint8
            image = image.astype(np.uint8)
        else:
            # Apply histogram equalization
            image = cv2.equalizeHist(image.astype(np.uint8))
        
        # Normalize to [0, 255] and ensure uint8
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
        
        return image
    
    def _zscore_normalization(self, image: np.ndarray) -> np.ndarray:
        """Z-score normalization preprocessing"""
        # Resize
        image = cv2.resize(image, self.config.image_size)
        
        # Convert to float
        image = image.astype(np.float32)
        
        # Z-score normalization
        if 'global_mean' in self.preprocessing_stats and 'global_std' in self.preprocessing_stats:
            # Use global statistics
            mean = self.preprocessing_stats['global_mean']
            std = self.preprocessing_stats['global_std']
        else:
            # Use image statistics
            mean = image.mean()
            std = image.std()
        
        if std > 0:
            image = (image - mean) / std
        
        # Clip to range
        clip_min, clip_max = self.config.z_score_clip_range
        image = np.clip(image, clip_min, clip_max)
        
        # Normalize to [0, 255]
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        return image.astype(np.uint8)

class ChestXrayDataManager:
    """Data manager for chest X-ray pneumonia dataset"""
    
    def __init__(self, config: Config):
        self.config = config
        self.data_root = Path(config.data.data_root)
        self.preprocessing_stats = {}
        
        # Create data directory if it doesn't exist
        self.data_root.mkdir(parents=True, exist_ok=True)
    
    def setup_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Set up train, validation, and test dataloaders"""
        
        # Download or prepare dataset
        self._prepare_dataset()
        
        # Create patient-level splits
        train_data, val_data, test_data = self._create_patient_splits()
        
        # Compute preprocessing statistics
        self._compute_preprocessing_stats(train_data)
        
        # Create datasets
        train_dataset = self._create_dataset(train_data, is_training=True)
        val_dataset = self._create_dataset(val_data, is_training=False)
        test_dataset = self._create_dataset(test_data, is_training=False)
        
        # Create dataloaders
        train_loader = self._create_dataloader(train_dataset, shuffle=True)
        val_loader = self._create_dataloader(val_dataset, shuffle=False)
        test_loader = self._create_dataloader(test_dataset, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def _prepare_dataset(self):
        """Prepare the chest X-ray dataset"""
        # Check for multiple possible dataset structures
        possible_paths = [
            self.data_root / self.config.data.dataset_name,  # Original path
            self.data_root / "chest-xray-pneumonia",          # Kaggle format
            self.data_root / "chest_xray",                    # Alternative format
            self.data_root                                    # Data directly in root
        ]
        
        dataset_path = None
        for path in possible_paths:
            if self._check_dataset_structure(path):
                dataset_path = path
                self.dataset_path = dataset_path
                print(f"✅ Found valid dataset at: {dataset_path}")
                break
        
        if dataset_path is None:
            print(f"❌ No valid dataset found in any of these locations:")
            for path in possible_paths:
                print(f"  - {path}")
            print("\n📥 Download options:")
            print("1. Kaggle Chest X-Ray Pneumonia Dataset:")
            print("   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
            print("2. COVID-19 Radiography Database:")
            print("   https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database")
            print("3. RSNA Pneumonia Detection Challenge:")
            print("   https://www.kaggle.com/c/rsna-pneumonia-detection-challenge")
            
            print("\n📁 Expected structure (option 1 - flat structure):")
            print("data/chest_xray_pneumonia/")
            print("  ├── NORMAL/")
            print("  └── PNEUMONIA/")
            
            print("\n📁 Expected structure (option 2 - Kaggle format):")
            print("data/chest-xray-pneumonia/")
            print("  ├── train/")
            print("  │   ├── NORMAL/")
            print("  │   └── PNEUMONIA/")
            print("  ├── val/")
            print("  │   ├── NORMAL/")
            print("  │   └── PNEUMONIA/")
            print("  └── test/")
            print("      ├── NORMAL/")
            print("      └── PNEUMONIA/")
            
            # Create dummy dataset for development
            dummy_path = self.data_root / self.config.data.dataset_name
            self._create_dummy_dataset(dummy_path)
            self.dataset_path = dummy_path
    
    def _check_dataset_structure(self, path: Path) -> bool:
        """Check if a path contains a valid dataset structure"""
        if not path.exists():
            return False
        
        # Check for flat structure (NORMAL/ and PNEUMONIA/ directly)
        flat_normal = path / "NORMAL"
        flat_pneumonia = path / "PNEUMONIA"
        
        if flat_normal.exists() and flat_pneumonia.exists():
            normal_count = len(list(flat_normal.glob("*.jpeg")) + list(flat_normal.glob("*.jpg")) + list(flat_normal.glob("*.png")))
            pneumonia_count = len(list(flat_pneumonia.glob("*.jpeg")) + list(flat_pneumonia.glob("*.jpg")) + list(flat_pneumonia.glob("*.png")))
            
            if normal_count > 0 and pneumonia_count > 0:
                print(f"Found flat structure: {normal_count} normal, {pneumonia_count} pneumonia images")
                return True
        
        # Check for split structure (train/val/test with NORMAL/ and PNEUMONIA/)
        splits = ["train", "val", "test"]
        has_splits = all((path / split).exists() for split in splits)
        
        if has_splits:
            total_normal = 0
            total_pneumonia = 0
            
            for split in splits:
                split_normal = path / split / "NORMAL"
                split_pneumonia = path / split / "PNEUMONIA"
                
                if split_normal.exists() and split_pneumonia.exists():
                    normal_count = len(list(split_normal.glob("*.jpeg")) + list(split_normal.glob("*.jpg")) + list(split_normal.glob("*.png")))
                    pneumonia_count = len(list(split_pneumonia.glob("*.jpeg")) + list(split_pneumonia.glob("*.jpg")) + list(split_pneumonia.glob("*.png")))
                    
                    total_normal += normal_count
                    total_pneumonia += pneumonia_count
                    print(f"Found {split} split: {normal_count} normal, {pneumonia_count} pneumonia")
            
            if total_normal > 0 and total_pneumonia > 0:
                print(f"Found split structure: {total_normal} total normal, {total_pneumonia} total pneumonia images")
                return True
        
        return False
    
    def _create_dummy_dataset(self, dataset_path: Path):
        """Create a dummy dataset for development and testing"""
        print("Creating dummy dataset for development...")
        
        # Create directories
        (dataset_path / "NORMAL").mkdir(parents=True, exist_ok=True)
        (dataset_path / "PNEUMONIA").mkdir(parents=True, exist_ok=True)
        
        # Generate dummy X-ray images
        np.random.seed(42)
        
        # Normal X-rays (clearer, less noisy) - simulate multiple patients
        for i in range(100):
            patient_id = f"patient{i//10:02d}"  # 10 patients, 10 images each
            image = self._generate_dummy_xray(is_pneumonia=False)
            cv2.imwrite(str(dataset_path / "NORMAL" / f"{patient_id}_normal_{i:03d}.png"), image)
        
        # Pneumonia X-rays (more opaque, cloudy) - simulate multiple patients  
        for i in range(100):
            patient_id = f"patient{(i//10)+10:02d}"  # Different patient IDs (10-19)
            image = self._generate_dummy_xray(is_pneumonia=True)
            cv2.imwrite(str(dataset_path / "PNEUMONIA" / f"{patient_id}_pneumonia_{i:03d}.png"), image)
        
        print(f"Dummy dataset created with 200 images at {dataset_path}")
    
    def _generate_dummy_xray(self, is_pneumonia: bool = False) -> np.ndarray:
        """Generate a dummy X-ray image"""
        # Base chest X-ray shape
        image = np.random.normal(120, 30, (512, 512)).astype(np.uint8)
        
        # Add chest cavity structure
        center_x, center_y = 256, 280
        
        # Lung areas (darker)
        lung_left = cv2.ellipse(np.zeros_like(image), (center_x - 80, center_y), 
                               (60, 120), 0, 0, 360, 50, -1)
        lung_right = cv2.ellipse(np.zeros_like(image), (center_x + 80, center_y), 
                                (60, 120), 0, 0, 360, 50, -1)
        
        image = image - lung_left - lung_right
        
        # Heart shadow
        heart = cv2.ellipse(np.zeros_like(image), (center_x - 20, center_y + 40), 
                           (40, 60), 0, 0, 360, 30, -1)
        image = image + heart
        
        if is_pneumonia:
            # Add consolidation/infiltrates (brighter areas in lungs)
            for _ in range(3):
                x = np.random.randint(center_x - 100, center_x + 100)
                y = np.random.randint(center_y - 80, center_y + 80)
                size = np.random.randint(20, 40)
                consolidation = cv2.circle(np.zeros_like(image), (x, y), size, 60, -1)
                image = image + consolidation
        
        # Add noise
        noise = np.random.normal(0, 10, image.shape)
        image = image + noise
        
        # Clip and normalize
        image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    def _create_patient_splits(self) -> Tuple[List, List, List]:
        """Create patient-level train/validation/test splits"""
        
        # Check if we have pre-split data (Kaggle format) or need to create splits
        if self._has_presplit_data():
            return self._load_presplit_data()
        else:
            return self._create_custom_splits()
    
    def _has_presplit_data(self) -> bool:
        """Check if dataset already has train/val/test splits"""
        splits = ["train", "val", "test"]
        return all((self.dataset_path / split / "NORMAL").exists() and 
                  (self.dataset_path / split / "PNEUMONIA").exists() 
                  for split in splits)
    
    def _load_presplit_data(self) -> Tuple[List, List, List]:
        """Load data from pre-existing train/val/test splits"""
        print("📂 Using pre-existing train/val/test splits...")
        
        def load_split_data(split_name: str) -> List:
            data = []
            split_path = self.dataset_path / split_name
            
            # Load normal cases
            normal_dir = split_path / "NORMAL"
            for img_path in normal_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    patient_id = self._extract_patient_id(img_path.name)
                    data.append({
                        'image_path': str(img_path),
                        'label': 0,  # Normal
                        'patient_id': patient_id
                    })
            
            # Load pneumonia cases
            pneumonia_dir = split_path / "PNEUMONIA"
            for img_path in pneumonia_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    patient_id = self._extract_patient_id(img_path.name)
                    data.append({
                        'image_path': str(img_path),
                        'label': 1,  # Pneumonia
                        'patient_id': patient_id
                    })
            
            print(f"  {split_name}: {len([d for d in data if d['label'] == 0])} normal, "
                  f"{len([d for d in data if d['label'] == 1])} pneumonia")
            
            return data
        
        train_data = load_split_data("train")
        val_data = load_split_data("val")
        test_data = load_split_data("test")
        
        return train_data, val_data, test_data
    
    def _create_custom_splits(self) -> Tuple[List, List, List]:
        """Create custom patient-level splits from flat structure"""
        print("📂 Creating custom train/val/test splits...")
        
        # Collect all images with labels
        data = []
        
        # Normal cases
        normal_dir = self.dataset_path / "NORMAL"
        if normal_dir.exists():
            for img_path in normal_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Extract patient ID from filename
                    patient_id = self._extract_patient_id(img_path.name)
                    data.append({
                        'image_path': str(img_path),
                        'label': 0,  # Normal
                        'patient_id': patient_id
                    })
        
        # Pneumonia cases
        pneumonia_dir = self.dataset_path / "PNEUMONIA"
        if pneumonia_dir.exists():
            for img_path in pneumonia_dir.glob("*"):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    # Extract patient ID from filename
                    patient_id = self._extract_patient_id(img_path.name)
                    data.append({
                        'image_path': str(img_path),
                        'label': 1,  # Pneumonia
                        'patient_id': patient_id
                    })
        
        if len(data) == 0:
            raise ValueError("No valid images found in dataset!")
        
        print(f"Found {len(data)} total images")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Patient-level splitting
        patients = df['patient_id'].unique()
        
        # Stratify by class distribution at patient level
        patient_labels = df.groupby('patient_id')['label'].max()  # Patient has pneumonia if any image has it
        
        # First split: train vs (val + test)
        train_patients, temp_patients = train_test_split(
            patients, 
            test_size=(self.config.data.val_ratio + self.config.data.test_ratio),
            stratify=patient_labels,
            random_state=self.config.experiment.random_seed
        )
        
        # Second split: val vs test
        val_size = self.config.data.val_ratio / (self.config.data.val_ratio + self.config.data.test_ratio)
        val_patients, test_patients = train_test_split(
            temp_patients,
            test_size=1-val_size,
            stratify=patient_labels[temp_patients],
            random_state=self.config.experiment.random_seed
        )
        
        # Create final splits
        train_data = df[df['patient_id'].isin(train_patients)].to_dict('records')
        val_data = df[df['patient_id'].isin(val_patients)].to_dict('records')
        test_data = df[df['patient_id'].isin(test_patients)].to_dict('records')
        
        print(f"Data splits created:")
        print(f"  Train: {len(train_data)} images from {len(train_patients)} patients")
        print(f"  Val: {len(val_data)} images from {len(val_patients)} patients")
        print(f"  Test: {len(test_data)} images from {len(test_patients)} patients")
        
        return train_data, val_data, test_data
    
    def _extract_patient_id(self, filename: str) -> str:
        """Extract patient ID from filename"""
        # For dummy data, use filename prefix
        # In real dataset, implement proper patient ID extraction
        return filename.split('_')[0]
    
    def _compute_preprocessing_stats(self, train_data: List[Dict]):
        """Compute statistics needed for preprocessing"""
        print("Computing preprocessing statistics...")
        
        if self.config.data.preprocessing_type == "histogram_matching":
            # Compute reference histogram from training data
            reference_images = []
            for item in train_data[:50]:  # Use subset for efficiency
                img = cv2.imread(item['image_path'], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, self.config.data.image_size)
                    reference_images.append(img)
            
            if reference_images:
                # Use median image as reference
                reference_stack = np.stack(reference_images)
                reference = np.median(reference_stack, axis=0)
                self.preprocessing_stats['reference_histogram'] = reference
        
        elif self.config.data.preprocessing_type == "z_score":
            # Compute global mean and std
            pixel_values = []
            for item in train_data:
                img = cv2.imread(item['image_path'], cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, self.config.data.image_size)
                    pixel_values.extend(img.flatten())
            
            if pixel_values:
                self.preprocessing_stats['global_mean'] = np.mean(pixel_values)
                self.preprocessing_stats['global_std'] = np.std(pixel_values)
    
    def _create_dataset(self, data: List[Dict], is_training: bool) -> PneumoniaDataset:
        """Create dataset with appropriate transforms"""
        image_paths = [item['image_path'] for item in data]
        labels = [item['label'] for item in data]
        
        # Create transforms
        transform = self._get_transforms(is_training)
        
        return PneumoniaDataset(
            image_paths=image_paths,
            labels=labels,
            config=self.config.data,
            transform=transform,
            preprocessing_stats=self.preprocessing_stats
        )
    
    def _get_transforms(self, is_training: bool) -> transforms.Compose:
        """Get image transforms for training or validation"""
        transform_list = []
        
        if is_training and self.config.data.use_augmentation:
            # Training augmentations
            transform_list.extend([
                transforms.RandomRotation(self.config.data.rotation_range),
                transforms.RandomHorizontalFlip(p=self.config.data.horizontal_flip_prob),
                transforms.ColorJitter(
                    brightness=self.config.data.brightness_factor,
                    contrast=self.config.data.contrast_factor
                ),
            ])
        
        # Common transforms
        transform_list.extend([
            transforms.Resize(self.config.data.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.data.normalize_mean,
                std=self.config.data.normalize_std
            )
        ])
        
        return transforms.Compose(transform_list)
    
    def _create_dataloader(self, dataset: PneumoniaDataset, shuffle: bool) -> DataLoader:
        """Create dataloader with specified configuration"""
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.experiment.num_workers,
            pin_memory=self.config.experiment.pin_memory,
            drop_last=shuffle  # Drop last batch for training to avoid issues with batch norm
        )
    
    def get_class_weights(self, train_loader: DataLoader) -> torch.Tensor:
        """Compute class weights for balanced loss"""
        label_counts = torch.zeros(2)
        
        for _, labels, _ in train_loader:
            for label in labels:
                label_counts[int(label.item())] += 1
        
        # Compute inverse frequency weights
        total_samples = label_counts.sum()
        class_weights = total_samples / (2 * label_counts)
        
        print(f"Class distribution: Normal={label_counts[0]:.0f}, Pneumonia={label_counts[1]:.0f}")
        print(f"Class weights: Normal={class_weights[0]:.3f}, Pneumonia={class_weights[1]:.3f}")
        
        return class_weights
    
    def visualize_preprocessing(self, num_samples: int = 4):
        """Visualize different preprocessing methods"""
        # Get some sample images
        dataset_path = self.data_root / self.config.data.dataset_name
        normal_images = list((dataset_path / "NORMAL").glob("*.png"))[:num_samples//2]
        pneumonia_images = list((dataset_path / "PNEUMONIA").glob("*.png"))[:num_samples//2]
        
        sample_images = normal_images + pneumonia_images
        sample_labels = [0] * len(normal_images) + [1] * len(pneumonia_images)
        
        # Create figure
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        preprocessing_types = ["raw", "histogram_matching", "z_score"]
        
        for i, (img_path, label) in enumerate(zip(sample_images, sample_labels)):
            # Load original image
            original = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            original = cv2.resize(original, self.config.data.image_size)
            
            # Show original
            axes[i, 0].imshow(original, cmap='gray')
            axes[i, 0].set_title(f"Original ({'Pneumonia' if label else 'Normal'})")
            axes[i, 0].axis('off')
            
            # Show different preprocessing
            for j, prep_type in enumerate(preprocessing_types):
                # Temporarily change preprocessing type
                old_prep_type = self.config.data.preprocessing_type
                self.config.data.preprocessing_type = prep_type
                
                # Create temporary dataset
                temp_dataset = PneumoniaDataset(
                    [str(img_path)], [label], self.config.data,
                    transform=None, preprocessing_stats=self.preprocessing_stats
                )
                
                # Get preprocessed image
                processed_img, _, _ = temp_dataset[0]
                if isinstance(processed_img, torch.Tensor):
                    processed_img = processed_img.numpy()
                
                # Show processed image
                if len(processed_img.shape) == 3:
                    processed_img = processed_img[0]  # Take first channel
                
                axes[i, j+1].imshow(processed_img, cmap='gray')
                axes[i, j+1].set_title(f"{prep_type.replace('_', ' ').title()}")
                axes[i, j+1].axis('off')
                
                # Restore original preprocessing type
                self.config.data.preprocessing_type = old_prep_type
        
        plt.tight_layout()
        plt.savefig('preprocessing_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    # Test data manager
    from config import get_raw_preprocessing_config
    
    config = get_raw_preprocessing_config()
    config.update_paths(".")
    
    data_manager = ChestXrayDataManager(config)
    
    # Create preprocessing visualization
    print("Creating preprocessing visualization...")
    data_manager.visualize_preprocessing(num_samples=2)
    
    # Setup data
    print("Setting up data loaders...")
    train_loader, val_loader, test_loader = data_manager.setup_data()
    
    print(f"Data loaders created:")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # Test a batch
    for images, labels, paths in train_loader:
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Label distribution: {labels.sum().item()}/{len(labels)} positive")
        break
