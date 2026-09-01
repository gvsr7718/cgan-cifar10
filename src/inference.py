"""Inference and conditional sample generation interface.

This module provides utilities for loading a trained cGAN Generator,
generating class-conditioned CIFAR-10 images, and exporting generated
samples from the command line.
"""

import argparse
import os
from typing import Optional

import torch

from src.models.generator import Generator
from src.training.checkpoint import load_checkpoint
from src.utils.image_utils import save_image_grid


def load_trained_generator(
    checkpoint_path: str,
    config: Optional[dict] = None,
    device: Optional[torch.device] = None,
) -> Generator:
    """Load a trained Generator from a checkpoint.

    Args:
        checkpoint_path: Path to a .pth/.pt checkpoint.
        config: Optional model configuration dictionary.
        device: Device on which to load the model.

    Returns:
        Generator in evaluation mode.
    """
    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model_config = config or {}

    generator = Generator(
        latent_dim=model_config.get("latent_dim", 100),
        num_classes=model_config.get("num_classes", 10),
        embedding_dim=model_config.get("embedding_dim", 50),
        image_channels=model_config.get("image_channels", 3),
        image_size=model_config.get("image_size", 32),
    ).to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # Support checkpoints containing the full training state.
    if "generator_state_dict" in checkpoint:
        state_dict = checkpoint["generator_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        # Also support a raw Generator state_dict.
        state_dict = checkpoint

    generator.load_state_dict(state_dict)
    generator.eval()

    return generator


@torch.no_grad()
def generate_samples(
    generator: Generator,
    class_idx: int,
    num_samples: int,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generate class-conditioned synthetic images.

    Args:
        generator: Trained Generator model.
        class_idx: CIFAR-10 class index from 0 to 9.
        num_samples: Number of images to generate.
        device: Device for generation.

    Returns:
        Tensor of generated images with shape
        [num_samples, 3, 32, 32], values in [-1, 1].
    """
    if not 0 <= class_idx < generator.num_classes:
        raise ValueError(
            f"class_idx must be in [0, {generator.num_classes - 1}], "
            f"got {class_idx}"
        )

    if num_samples <= 0:
        raise ValueError("num_samples must be greater than zero.")

    if device is None:
        device = next(generator.parameters()).device

    noise = torch.randn(
        num_samples,
        generator.latent_dim,
        device=device,
    )

    labels = torch.full(
        (num_samples,),
        class_idx,
        dtype=torch.long,
        device=device,
    )

    generator.eval()

    images = generator(noise, labels)

    return images


def main() -> None:
    """Command-line interface for conditional image generation."""

    parser = argparse.ArgumentParser(
        description="Generate class-conditioned CIFAR-10 images using a trained cGAN."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the trained Generator checkpoint.",
    )

    parser.add_argument(
        "--class-idx",
        type=int,
        required=True,
        help="CIFAR-10 class index (0-9).",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of images to generate.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/samples/generated_grid.png",
        help="Output path for the generated image grid.",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    generator = load_trained_generator(
        checkpoint_path=args.checkpoint,
        device=device,
    )

    images = generate_samples(
        generator=generator,
        class_idx=args.class_idx,
        num_samples=args.num_samples,
        device=device,
    )

    os.makedirs(
        os.path.dirname(args.output) or ".",
        exist_ok=True,
    )

    save_image_grid(
        images,
        args.output,
        nrow=min(10, args.num_samples),
    )

    print(f"Generated {args.num_samples} images.")
    print(f"Class: {args.class_idx}")
    print(f"Device: {device}")
    print(f"Saved grid to: {args.output}")


if __name__ == "__main__":
    main()