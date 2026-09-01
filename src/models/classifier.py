"""CIFAR-10 classifier for class-consistency evaluation.

This module provides a lightweight CNN classifier that can be trained
independently on CIFAR-10 and later used to evaluate whether generated
cGAN images match their intended class labels.
"""

from typing import Tuple

import torch
import torch.nn as nn


class CIFAR10Classifier(nn.Module):
    """CNN classifier for CIFAR-10 images.

    Args:
        image_channels: Number of input channels. Default: 3.
        num_classes: Number of CIFAR-10 classes. Default: 10.
    """

    def __init__(
        self,
        image_channels: int = 3,
        num_classes: int = 10,
    ) -> None:
        super().__init__()

        self.image_channels = image_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(image_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 16x16 -> 8x8
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 8x8 -> 4x4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Classify a batch of CIFAR-10 images.

        Args:
            images: Tensor with shape [batch_size, 3, 32, 32].

        Returns:
            Class logits with shape [batch_size, num_classes].
        """
        features = self.features(images)
        return self.classifier(features)


def predict(
    classifier: CIFAR10Classifier,
    images: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Predict class labels and confidence scores.

    Args:
        classifier: CIFAR-10 classifier.
        images: Input images with shape [B, 3, 32, 32].

    Returns:
        A tuple containing:
            predictions: Predicted class indices [B].
            confidence: Maximum softmax probability [B].
    """
    classifier.eval()

    with torch.no_grad():
        logits = classifier(images)
        probabilities = torch.softmax(logits, dim=1)
        confidence, predictions = probabilities.max(dim=1)

    return predictions, confidence


def classifier_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Calculate standard cross-entropy classification loss."""
    return nn.functional.cross_entropy(logits, labels)