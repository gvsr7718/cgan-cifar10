"""Dataset loading and DataLoader construction for CIFAR-10.

This module handles downloading, caching, splitting the CIFAR-10
dataset, and constructing PyTorch DataLoader instances with appropriate
batch sizes, shuffling, and worker configurations.
"""

import os
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.datasets import CIFAR10

from src.data.preprocessing import get_train_transforms, get_eval_transforms


def get_class_mapping(config: Dict[str, Any]) -> Dict[int, str]:
    """
    Returns a mapping from class index to class name based on the configuration.
    
    Args:
        config (dict): The configuration dictionary.
        
    Returns:
        dict: Mapping of integer index (0-9) to string class name.
    """
    class_names = config.get("dataset", {}).get("class_names", [])
    if len(class_names) != 10:
        raise ValueError("CIFAR-10 must have exactly 10 class names configured.")
    return {i: name for i, name in enumerate(class_names)}


def validate_dataset(dataset: Dataset, expected_len: int, name: str) -> None:
    """
    Validates that a dataset has the expected number of samples, 
    and that samples conform to expected shapes and label ranges.
    
    Args:
        dataset (Dataset): The dataset to validate.
        expected_len (int): The expected number of samples.
        name (str): The name of the dataset split (for logging/errors).
    """
    if len(dataset) != expected_len:
        raise ValueError(f"{name} dataset size mismatch: expected {expected_len}, got {len(dataset)}")
    
    # Check a single sample
    if len(dataset) > 0:
        image, label = dataset[0]
        
        # Check image shape [3, 32, 32]
        if image.shape != (3, 32, 32):
            raise ValueError(f"{name} dataset image shape mismatch: expected (3, 32, 32), got {image.shape}")
            
        # Check label range 0-9
        if not (0 <= label <= 9):
            raise ValueError(f"{name} dataset label out of bounds: expected 0-9, got {label}")


def get_cifar10_datasets(config: Dict[str, Any], data_root: str = "data") -> Tuple[Dataset, Dataset, Dataset]:
    """
    Downloads (if necessary) and creates the train, validation, and test datasets.
    
    Args:
        config (dict): Configuration dictionary containing dataset settings and random seed.
        data_root (str): The directory where the dataset is stored or will be downloaded.
        
    Returns:
        tuple: (train_dataset, validation_dataset, test_dataset)
    """
    seed = config.get("training", {}).get("seed", 42)
    train_split = config.get("dataset", {}).get("train_split", 0.90)
    
    # Official CIFAR-10 training set (50,000 images)
    full_train_dataset = CIFAR10(
        root=data_root, 
        train=True, 
        download=True, 
        transform=get_train_transforms()
    )
    
    # Official CIFAR-10 test set (10,000 images)
    test_dataset = CIFAR10(
        root=data_root, 
        train=False, 
        download=True, 
        transform=get_eval_transforms()
    )
    
    total_train_samples = len(full_train_dataset)
    if total_train_samples != 50000:
        raise ValueError(f"Official CIFAR-10 train set should have 50000 samples, got {total_train_samples}")
        
    train_size = int(train_split * total_train_samples)
    val_size = total_train_samples - train_size
    
    # Use PyTorch generator with the configured seed for deterministic splits
    generator = torch.Generator().manual_seed(seed)
    
    train_dataset, validation_dataset = random_split(
        full_train_dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    # Validation dataset should ideally use eval transforms.
    # Since random_split wraps the dataset, it inherits the transforms of full_train_dataset.
    # For CIFAR-10 cGAN without complex augmentations (train and eval transforms are both just ToTensor + Normalize),
    # this is perfectly fine. If we had RandomCrop in train_transforms, we'd need a custom Dataset wrapper 
    # to apply eval_transforms to the validation split. 
    
    # Validate datasets
    validate_dataset(train_dataset, train_size, "Train")
    validate_dataset(validation_dataset, val_size, "Validation")
    validate_dataset(test_dataset, 10000, "Test")
    
    return train_dataset, validation_dataset, test_dataset


def get_cifar10_dataloaders(
    config: Dict[str, Any], 
    data_root: str = "data"
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Constructs PyTorch DataLoader instances for the train, validation, and test sets.
    
    Args:
        config (dict): Configuration dictionary containing training settings (batch_size, num_workers).
        data_root (str): The directory where the dataset is stored or will be downloaded.
        
    Returns:
        tuple: (train_loader, validation_loader, test_loader)
    """
    train_dataset, validation_dataset, test_dataset = get_cifar10_datasets(config, data_root)
    
    batch_size = config.get("training", {}).get("batch_size", 128)
    num_workers = config.get("training", {}).get("num_workers", 2)
    
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,          # Shuffle training data
        num_workers=num_workers,
        drop_last=True         # Drop incomplete batches for stable GAN training
    )
    
    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,         # Do not shuffle validation data
        num_workers=num_workers,
        drop_last=False
    )
    
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,         # Do not shuffle test data
        num_workers=num_workers,
        drop_last=False
    )
    
    return train_loader, validation_loader, test_loader
