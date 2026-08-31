"""Conditional Generator architecture for CIFAR-10 image synthesis.

This module defines the Generator network for a Conditional GAN (cGAN) that maps
a latent noise vector z concatenated with a learned class embedding to a synthetic
32x32x3 RGB image. The architecture follows a DCGAN-style design with transposed
convolutions, batch normalization, and ReLU activations, producing output in [-1, 1]
via a Tanh final activation — compatible with the project's CIFAR-10 preprocessing
normalization scheme (mean=0.5, std=0.5 per channel).

Architecture Overview:
    [z (100-dim)] + [embedding(y) (50-dim)] = [150-dim combined vector]
        → Linear → Reshape to (256, 4, 4)
        → ConvTranspose2d → (128, 8, 8)   + BatchNorm + ReLU
        → ConvTranspose2d → (64, 16, 16)  + BatchNorm + ReLU
        → ConvTranspose2d → (3, 32, 32)   + Tanh
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """Conditional Generator network for class-conditioned CIFAR-10 image synthesis.

    The Generator accepts two inputs:
        - A latent noise vector z sampled from a standard normal distribution N(0, I).
          This vector encodes the stochastic variation in generated images (e.g., pose,
          texture details, background) and has shape [batch_size, latent_dim].
        - An integer class label y in {0, 1, ..., num_classes-1} identifying the target
          CIFAR-10 category. The label is mapped to a dense vector via a learned
          nn.Embedding layer, producing a class embedding of shape [batch_size, embedding_dim].

    The noise vector and class embedding are concatenated into a single conditional
    input vector of dimension (latent_dim + embedding_dim), which is then projected
    and reshaped through a series of transposed convolutional blocks to produce an
    output image tensor of shape [batch_size, image_channels, image_size, image_size].

    The output values lie in [-1, 1] due to the Tanh final activation.

    Args:
        latent_dim (int): Dimensionality of the input noise vector z. Default: 100.
        num_classes (int): Number of conditional class categories. Default: 10.
        embedding_dim (int): Dimensionality of the learned class embedding. Default: 50.
        image_channels (int): Number of output image channels (3 for RGB). Default: 3.
        image_size (int): Spatial dimension of the output image (height = width). Default: 32.
    """

    def __init__(
        self,
        latent_dim: int = 100,
        num_classes: int = 10,
        embedding_dim: int = 50,
        image_channels: int = 3,
        image_size: int = 32,
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.image_channels = image_channels
        self.image_size = image_size

        # Learned class embedding: maps integer label to a dense vector
        self.label_embedding = nn.Embedding(
            num_embeddings=num_classes,
            embedding_dim=embedding_dim,
        )

        combined_dim = latent_dim + embedding_dim  # 100 + 50 = 150

        # Project the combined latent vector to a spatial feature map of shape (256, 4, 4)
        self.projection = nn.Sequential(
            nn.Linear(combined_dim, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(inplace=True),
        )

        # Transposed convolutional upsampling blocks
        self.deconv_blocks = nn.Sequential(
            # Block 1: (256, 4, 4) → (128, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Block 2: (128, 8, 8) → (64, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: (64, 16, 16) → (3, 32, 32)
            nn.ConvTranspose2d(64, image_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )

        # Initialize weights following DCGAN best practices
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize model weights using DCGAN-recommended initialization.

        - ConvTranspose2d layers: Normal distribution with mean=0 and std=0.02.
        - BatchNorm layers: Weight ~ N(1.0, 0.02), bias = 0.
        - Linear layers: Normal distribution with mean=0 and std=0.02.
        - Embedding layers: Normal distribution with mean=0 and std=0.02.
        """
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.normal_(m.weight, mean=1.0, std=0.02)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Generate class-conditioned synthetic images from noise and labels.

        Args:
            noise (torch.Tensor): Latent noise vectors sampled from N(0, I).
                Shape: [batch_size, latent_dim].
            labels (torch.Tensor): Integer class indices for conditioning.
                Shape: [batch_size]. Values must be in {0, 1, ..., num_classes - 1}.

        Returns:
            torch.Tensor: Generated RGB images with pixel values in [-1, 1].
                Shape: [batch_size, image_channels, image_size, image_size].
        """
        # Map integer labels to learned dense embeddings: [batch_size, embedding_dim]
        label_emb = self.label_embedding(labels)

        # Concatenate noise and class embedding: [batch_size, latent_dim + embedding_dim]
        combined = torch.cat([noise, label_emb], dim=1)

        # Project to spatial feature map and reshape: [batch_size, 256, 4, 4]
        x = self.projection(combined)
        x = x.view(-1, 256, 4, 4)

        # Upsample through transposed convolution blocks: [batch_size, 3, 32, 32]
        x = self.deconv_blocks(x)

        return x
