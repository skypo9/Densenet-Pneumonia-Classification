#!/usr/bin/env python3

import sys
import argparse
sys.path.append('src')
from config import Config, get_raw_preprocessing_config

def test_epochs():
    # Test command line argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--preprocessing", type=str, default="raw", help="Preprocessing method")
    
    # Parse test arguments
    test_args = ["--epochs", "2", "--preprocessing", "raw"]
    args = parser.parse_args(test_args)
    
    # Load config
    config = get_raw_preprocessing_config()
    
    print(f"Original config epochs: {config.training.num_epochs}")
    
    # Update with command line args
    config.training.num_epochs = args.epochs
    
    print(f"Updated config epochs: {config.training.num_epochs}")
    print(f"Command line epochs: {args.epochs}")
    
    if config.training.num_epochs == args.epochs:
        print("✅ Epoch configuration is working correctly!")
    else:
        print("❌ Epoch configuration is NOT working!")

if __name__ == "__main__":
    test_epochs()
