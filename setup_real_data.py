"""
Real Dataset Setup Helper
Helps download and verify chest X-ray datasets for pneumonia classification
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
import shutil
from typing import Optional
import argparse

def check_data_structure(data_dir: str) -> bool:
    """Check if data directory has correct structure"""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return False
    
    # Check for expected structure
    expected_structure = {
        'train': ['NORMAL', 'PNEUMONIA'],
        'val': ['NORMAL', 'PNEUMONIA'], 
        'test': ['NORMAL', 'PNEUMONIA']
    }
    
    print(f"📁 Checking data structure in: {data_dir}")
    
    total_images = 0
    for split, classes in expected_structure.items():
        split_path = data_path / split
        if not split_path.exists():
            print(f"❌ Missing split directory: {split}")
            return False
            
        for class_name in classes:
            class_path = split_path / class_name
            if not class_path.exists():
                print(f"❌ Missing class directory: {split}/{class_name}")
                return False
            
            # Count images
            image_files = list(class_path.glob('*.jpeg')) + list(class_path.glob('*.jpg')) + list(class_path.glob('*.png'))
            count = len(image_files)
            total_images += count
            print(f"  {split}/{class_name}: {count} images")
    
    print(f"✅ Data structure verified! Total images: {total_images}")
    return True

def download_sample_dataset():
    """Download a small sample dataset for testing"""
    print("🔄 Creating sample dataset structure...")
    
    # Create directory structure
    base_dir = Path("data/chest-xray-pneumonia")
    
    for split in ['train', 'val', 'test']:
        for class_name in ['NORMAL', 'PNEUMONIA']:
            (base_dir / split / class_name).mkdir(parents=True, exist_ok=True)
    
    print("✅ Sample directory structure created!")
    print("\n📝 Next steps:")
    print("1. Download real chest X-ray dataset from:")
    print("   https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("2. Extract the downloaded zip file")
    print("3. Copy images to the created directory structure")
    print("4. Run: python setup_real_data.py --verify")

def setup_kaggle_api():
    """Help user set up Kaggle API for dataset download"""
    print("🔧 Kaggle API Setup Instructions:")
    print("\n1. Create Kaggle account: https://www.kaggle.com/")
    print("2. Go to Account settings: https://www.kaggle.com/account")
    print("3. Create API token (download kaggle.json)")
    print("4. Install Kaggle API:")
    print("   pip install kaggle")
    print("5. Place kaggle.json in:")
    print("   Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
    print("   Linux/Mac: ~/.kaggle/kaggle.json")
    print("6. Download dataset:")
    print("   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
    print("   unzip chest-xray-pneumonia.zip -d data/")

def create_test_config():
    """Create optimized config for real data testing"""
    config_content = """# Real Data Test Configuration
data:
  data_root: "data/chest-xray-pneumonia"
  preprocessing_type: "histogram_matching"
  image_size: [224, 224]
  use_augmentation: true
  
model:
  backbone: "densenet121"
  pretrained: true
  dropout_rate: 0.3
  
training:
  epochs: 15
  batch_size: 16
  learning_rate: 1.0e-4
  weight_decay: 1.0e-3
  class_weights: [1.0, 1.5]  # Balance for pneumonia dataset
  early_stopping_patience: 8
  
evaluation:
  generate_gradcam: true
  num_gradcam_samples: 20
  threshold_metric: "f1"
  
experiment:
  experiment_name: "real_data_test"
  description: "Test with real chest X-ray data"
"""
    
    config_path = Path("configs/real_data_config.yaml")
    config_path.parent.mkdir(exist_ok=True)
    
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Real data config created: {config_path}")
    return config_path

def verify_installation():
    """Verify that all required packages are installed"""
    print("🔍 Verifying installation...")
    
    required_packages = [
        'torch', 'torchvision', 'opencv-python', 'Pillow', 
        'albumentations', 'numpy', 'scipy', 'scikit-learn',
        'pandas', 'matplotlib', 'seaborn', 'pyyaml', 
        'tensorboard', 'tqdm', 'scikit-image'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {missing_packages}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✅ All packages installed!")
        return True

def run_quick_test():
    """Run a quick test with real data"""
    print("🚀 Running quick test with real data...")
    
    # Check if data exists
    if not check_data_structure("data/chest-xray-pneumonia"):
        print("❌ Real data not found. Please download first.")
        return False
    
    # Create config
    config_path = create_test_config()
    
    # Run test
    cmd = f'python main.py --config {config_path} --mode single --epochs 3 --batch_size 8'
    print(f"Running: {cmd}")
    
    import subprocess
    try:
        result = subprocess.run(cmd.split(), capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Quick test successful!")
            return True
        else:
            print(f"❌ Test failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Real Dataset Setup Helper")
    parser.add_argument("--verify", action="store_true", help="Verify data structure")
    parser.add_argument("--setup-kaggle", action="store_true", help="Show Kaggle API setup instructions")
    parser.add_argument("--create-sample", action="store_true", help="Create sample directory structure")
    parser.add_argument("--create-config", action="store_true", help="Create real data config")
    parser.add_argument("--check-install", action="store_true", help="Check package installation")
    parser.add_argument("--quick-test", action="store_true", help="Run quick test with real data")
    parser.add_argument("--data-dir", default="data/chest-xray-pneumonia", help="Data directory path")
    
    args = parser.parse_args()
    
    if args.verify:
        check_data_structure(args.data_dir)
    elif args.setup_kaggle:
        setup_kaggle_api()
    elif args.create_sample:
        download_sample_dataset()
    elif args.create_config:
        create_test_config()
    elif args.check_install:
        verify_installation()
    elif args.quick_test:
        run_quick_test()
    else:
        # Show help menu
        print("🏥 Real Dataset Setup Helper")
        print("\nAvailable commands:")
        print("  --verify          Check if data is properly structured")
        print("  --setup-kaggle    Show Kaggle API setup instructions")
        print("  --create-sample   Create sample directory structure")
        print("  --create-config   Create optimized config for real data")
        print("  --check-install   Verify all packages are installed")
        print("  --quick-test      Run quick test with real data")
        print("\nExample usage:")
        print("  python setup_real_data.py --verify")
        print("  python setup_real_data.py --quick-test")

if __name__ == "__main__":
    main()
