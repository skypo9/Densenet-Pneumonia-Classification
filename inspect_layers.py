#!/usr/bin/env python3

import sys
sys.path.append('src')
from model import create_model
from config import get_raw_preprocessing_config

def inspect_model_layers():
    config = get_raw_preprocessing_config()
    print("Model backbone:", config.model.backbone)
    model = create_model(config.model)
    
    print("Available layers in DenseNet-121:")
    print("=" * 50)
    
    for name, module in model.named_modules():
        print(f"{name}: {type(module).__name__}")
        
    print("\n" + "=" * 50)
    print("Looking for denseblock layers:")
    for name, module in model.named_modules():
        if 'denseblock' in name or 'dense' in name.lower():
            print(f"  {name}: {type(module).__name__}")

if __name__ == "__main__":
    inspect_model_layers()
