"""Data preprocessing and augmentation pipelines.

This module defines torchvision transformation pipelines for training
and evaluation, including normalization to [-1, 1] for GAN training stability.
"""

import torchvision.transforms as transforms
from typing import Tuple

def get_base_transforms() -> transforms.Compose:
    """
    Get the standard transformations for CIFAR-10 cGAN training and evaluation.
    
    Transforms applied:
    - ToTensor: Converts PIL Image or numpy.ndarray (H x W x C) in the range
      [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    - Normalize: Normalizes each channel with mean=0.5 and std=0.5, resulting
      in a range of [-1.0, 1.0] which is suitable for Generator networks with Tanh output.
      
    Returns:
        torchvision.transforms.Compose: The composition of transforms.
    """
    # CIFAR-10 images are natively 32x32.
    # Mean and Std of (0.5, 0.5, 0.5) normalize data to [-1, 1] range.
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

def get_train_transforms() -> transforms.Compose:
    """
    Get transformations for training data.
    
    Currently returns the base transformations. Data augmentation (e.g., RandomHorizontalFlip)
    can be added here if needed, but keeping it simple for baseline cGAN.
    """
    return get_base_transforms()

def get_eval_transforms() -> transforms.Compose:
    """
    Get transformations for validation and test data.
    """
    return get_base_transforms()

def get_denormalize_transform() -> transforms.Compose:
    """
    Get a transform to revert the [-1, 1] normalization back to [0, 1]
    for visualization or saving images.
    """
    # To denormalize, we reverse the operation: x_denorm = (x * std) + mean
    # Since mean=0.5, std=0.5, we have x_denorm = x * 0.5 + 0.5
    # This is equivalent to Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
    # However, it's safer to implement it explicitly or just use a lambda.
    return transforms.Lambda(lambda t: (t * 0.5) + 0.5)
