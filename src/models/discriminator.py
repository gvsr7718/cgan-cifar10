"""Conditional Discriminator architecture for CIFAR-10 image validation.

This module defines the Discriminator network for a Conditional GAN (cGAN) that
evaluates whether an input 32x32x3 RGB image, conditioned on a class label, is
real or generated. The architecture follows a DCGAN-style design with strided
convolutions, BatchNorm, and LeakyReLU activations. Class conditioning is achieved
by concatenating a learned label embedding with flattened image features before the
final classification layer.

Architecture Overview:
    [Image (3, 32, 32)]
        → Conv2d → (64, 16, 16)  + LeakyReLU(0.2)
        → Conv2d → (128, 8, 8)   + BatchNorm + LeakyReLU(0.2)
        → Conv2d → (256, 4, 4)   + BatchNorm + LeakyReLU(0.2)
        → Flatten → (4096,)

    [Class label y]
        → Embedding → (50,)

    [4096-dim image features] + [50-dim label embedding] = [4146-dim combined]
        → Linear → scalar logit (no Sigmoid — intended for BCEWithLogitsLoss)
"""

import torch
import torch.nn as nn


class Discriminator(nn.Module):
    """Conditional Discriminator network for real/fake classification of CIFAR-10 images.

    The Discriminator accepts two inputs:
        - An image tensor of shape [batch_size, image_channels, image_size, image_size]
          containing either real CIFAR-10 images or synthetic images produced by the
          Generator. Pixel values are expected in [-1, 1] (matching the project's
          normalization scheme).
        - An integer class label y in {0, 1, ..., num_classes-1} identifying the
          CIFAR-10 category that the image is claimed to belong to. The label is mapped
          to a dense vector via a learned nn.Embedding layer.

    The image is processed through a convolutional feature extractor that progressively
    downsamples spatial dimensions while increasing channel depth. The resulting feature
    vector is concatenated with the class embedding and passed through a linear layer
    to produce a single scalar logit representing the real/fake prediction.

    No Sigmoid activation is applied to the output — the raw logit is returned for
    use with BCEWithLogitsLoss, which is numerically more stable.

    Args:
        image_channels (int): Number of input image channels (3 for RGB). Default: 3.
        num_classes (int): Number of conditional class categories. Default: 10.
        embedding_dim (int): Dimensionality of the learned class embedding. Default: 50.
        image_size (int): Spatial dimension of the input image (must be 32). Default: 32.
    """

    def __init__(
        self,
        image_channels: int = 3,
        num_classes: int = 10,
        embedding_dim: int = 50,
        image_size: int = 32,
    ) -> None:
        super().__init__()

        if image_size != 32:
            raise ValueError(
                f"Discriminator architecture requires image_size=32, got {image_size}. "
                "The three Conv2d layers with kernel_size=4, stride=2, padding=1 "
                "downsample 32 → 16 → 8 → 4. Other sizes are not supported."
            )

        self.image_channels = image_channels
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.image_size = image_size

        # Convolutional feature extractor
        # Progressively downsamples spatial dimensions: 32 → 16 → 8 → 4
        self.conv_blocks = nn.Sequential(
            # Block 1: (3, 32, 32) → (64, 16, 16)
            # No BatchNorm on the first layer (DCGAN convention)
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 2: (64, 16, 16) → (128, 8, 8)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # Block 3: (128, 8, 8) → (256, 4, 4)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Learned class embedding: maps integer label to a dense vector
        self.label_embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=embedding_dim,
        )

        # Final classification layer
        # Input: flattened image features (256 * 4 * 4 = 4096) + label embedding (50) = 4146
        feature_dim = 256 * 4 * 4  # 4096
        combined_dim = feature_dim + embedding_dim  # 4146

        self.classifier = nn.Linear(combined_dim, 1)

        # Initialize weights following DCGAN best practices
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights using DCGAN-recommended initialization.

        - Conv2d layers: Normal distribution with mean=0 and std=0.02.
        - BatchNorm layers: Weight ~ N(1.0, 0.02), bias = 0.
        - Linear layers: Normal distribution with mean=0 and std=0.02.
        - Embedding layers: Normal distribution with mean=0 and std=0.02.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Classify an image as real or fake, conditioned on a class label.

        Args:
            images (torch.Tensor): Input images (real or generated).
                Shape: [batch_size, image_channels, image_size, image_size].
                Values expected in [-1, 1].
            labels (torch.Tensor): Integer class indices for conditioning.
                Shape: [batch_size]. Values must be in {0, 1, ..., num_classes - 1}.

        Returns:
            torch.Tensor: Raw logits (no Sigmoid) for real/fake classification.
                Shape: [batch_size, 1].
        """
        # Extract image features through convolutional blocks: [B, 256, 4, 4]
        features = self.conv_blocks(images)

        # Flatten spatial dimensions: [B, 256 * 4 * 4] = [B, 4096]
        features = features.view(features.size(0), -1)

        # Map integer labels to learned dense embeddings: [B, embedding_dim]
        label_emb = self.label_embedding(labels)

        # Concatenate image features and class embedding: [B, 4096 + 50] = [B, 4146]
        combined = torch.cat([features, label_emb], dim=1)

        # Produce scalar logit: [B, 1]
        logit = self.classifier(combined)

        return logit
