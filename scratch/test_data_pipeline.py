import yaml
import sys
import os
import torch
from collections import Counter

# Add the project root to the python path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset import get_cifar10_datasets, get_cifar10_dataloaders, get_class_mapping

def run_tests():
    print("--- Starting Data Pipeline Tests ---")
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    data_root = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    try:
        # Test mapping
        class_map = get_class_mapping(config)
        print(f"1. Class mapping successfully retrieved: {class_map}")
        
        # Test datasets creation and length
        train_ds, val_ds, test_ds = get_cifar10_datasets(config, data_root=data_root)
        print(f"2. Number of training samples: {len(train_ds)}")
        print(f"3. Number of validation samples: {len(val_ds)}")
        print(f"4. Number of test samples: {len(test_ds)}")
        
        if len(train_ds) != 45000:
            print(f"ERROR: Expected 45000 train samples, got {len(train_ds)}")
        if len(val_ds) != 5000:
            print(f"ERROR: Expected 5000 val samples, got {len(val_ds)}")
        if len(test_ds) != 10000:
            print(f"ERROR: Expected 10000 test samples, got {len(test_ds)}")

        # Test DataLoaders
        train_loader, val_loader, test_loader = get_cifar10_dataloaders(config, data_root=data_root)
        
        # Retrieve one batch
        images, labels = next(iter(train_loader))
        print(f"5. One batch image tensor shape: {images.shape}")
        print(f"6. One batch label tensor shape: {labels.shape}")
        
        # Verify finite values
        if not torch.isfinite(images).all():
            print("ERROR: Images contain NaN or Inf values")
        else:
            print("7. All image values are finite")
            
        # Check normalization range approx [-1, 1]
        print(f"   Image value range: [{images.min().item():.4f}, {images.max().item():.4f}]")
            
        # Verify labels are valid
        invalid_labels = [l.item() for l in labels if l.item() < 0 or l.item() > 9]
        if invalid_labels:
            print(f"ERROR: Invalid labels found: {invalid_labels}")
        else:
            print("8. All labels are valid (in range 0-9)")
            
        # Print class distribution for the batch
        label_counts = Counter(labels.tolist())
        dist = {class_map[k]: v for k, v in label_counts.items()}
        print(f"9. Class distribution for this batch: {dist}")
        
        print("\n=> ALL VALIDATION CHECKS PASSED SUCCESSFULLY")
        
    except Exception as e:
        print(f"\n=> ERROR ENCOUNTERED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    run_tests()
