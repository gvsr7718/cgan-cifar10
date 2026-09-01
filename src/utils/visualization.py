"""Visualization tools for synthetic images and training metrics.

Provides utilities for:
- Class-conditional generated image grids.
- Generator vs. Discriminator loss curves.
- FID progression curves.
"""

import os
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

from src.utils.image_utils import denormalize_tensor


def plot_class_conditional_grid(
    generator,
    num_classes: int = 10,
    samples_per_class: int = 10,
    save_path: str = "results/samples/class_conditional_grid.png",
) -> None:
    """Generate and save a grid of class-conditioned images.

    Rows correspond to classes and columns correspond to different
    random latent vectors.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be greater than zero.")

    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be greater than zero.")

    device = next(generator.parameters()).device
    latent_dim = generator.latent_dim

    was_training = generator.training
    generator.eval()

    with torch.no_grad():
        labels = torch.arange(
            num_classes,
            device=device,
        ).repeat_interleave(samples_per_class)

        noise = torch.randn(
            num_classes * samples_per_class,
            latent_dim,
            device=device,
        )

        images = generator(noise, labels)

    if was_training:
        generator.train()

    images = denormalize_tensor(images).cpu()

    grid = make_grid(
        images,
        nrow=samples_per_class,
        padding=2,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.figure(figsize=(12, 12))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_training_losses(
    g_losses: Sequence[float],
    d_losses: Sequence[float],
    save_path: str = "results/graphs/training_losses.png",
) -> None:
    """Plot and save Generator and Discriminator losses."""
    if len(g_losses) == 0 or len(d_losses) == 0:
        raise ValueError("Loss histories cannot be empty.")

    if len(g_losses) != len(d_losses):
        raise ValueError(
            "Generator and Discriminator loss histories must have "
            "the same length."
        )

    epochs = range(1, len(g_losses) + 1)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, g_losses, label="Generator Loss")
    plt.plot(epochs, d_losses, label="Discriminator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("cGAN Training Losses")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_fid_curve(
    epochs: Sequence[int],
    fid_scores: Sequence[float],
    save_path: str = "results/graphs/fid_curve.png",
) -> None:
    """Plot and save FID scores across evaluation epochs."""
    if len(epochs) == 0 or len(fid_scores) == 0:
        raise ValueError("Epochs and FID histories cannot be empty.")

    if len(epochs) != len(fid_scores):
        raise ValueError(
            "Epoch and FID histories must have the same length."
        )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, fid_scores, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("FID Score")
    plt.title("FID Progression")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()