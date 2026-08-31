"""Adversarial loss functions for cGAN training.

This module provides standard Binary Cross-Entropy with Logits losses
for training the Generator and Discriminator of a conditional GAN.

The Discriminator returns raw logits, so BCEWithLogitsLoss is used
instead of applying Sigmoid inside the model.
"""

import torch
import torch.nn as nn


def discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    real_label: float = 1.0,
    fake_label: float = 0.0,
) -> torch.Tensor:
    """Calculate the Discriminator adversarial loss.

    Args:
        real_logits: Discriminator logits for real images.
            Shape: [batch_size, 1].
        fake_logits: Discriminator logits for generated images.
            Shape: [batch_size, 1].
        real_label: Target value for real images. Default: 1.0.
        fake_label: Target value for fake images. Default: 0.0.

    Returns:
        Scalar Discriminator loss.
    """
    criterion = nn.BCEWithLogitsLoss()

    real_targets = torch.full_like(real_logits, real_label)
    fake_targets = torch.full_like(fake_logits, fake_label)

    real_loss = criterion(real_logits, real_targets)
    fake_loss = criterion(fake_logits, fake_targets)

    return real_loss + fake_loss


def generator_loss(
    fake_logits: torch.Tensor,
    real_label: float = 1.0,
) -> torch.Tensor:
    """Calculate the Generator adversarial loss.

    The Generator wants the Discriminator to classify generated images
    as real, so fake images receive a target of 1.0.

    Args:
        fake_logits: Discriminator logits for generated images.
            Shape: [batch_size, 1].
        real_label: Target value the Generator wants. Default: 1.0.

    Returns:
        Scalar Generator loss.
    """
    criterion = nn.BCEWithLogitsLoss()

    targets = torch.full_like(fake_logits, real_label)

    return criterion(fake_logits, targets)