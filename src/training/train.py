"""Training loop for the Conditional GAN (cGAN).

This module provides a reusable Trainer class that alternates Discriminator
and Generator updates using the adversarial losses defined in losses.py.
"""

import os
import sys
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.training.losses import discriminator_loss, generator_loss
from src.training.checkpoint import save_checkpoint


class Trainer:
    """Trainer for a conditional GAN."""

    def __init__(
        self,
        generator: Generator,
        discriminator: Discriminator,
        train_loader: DataLoader,
        device: torch.device,
        lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        checkpoint_dir: str = "checkpoints",
    ) -> None:
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.train_loader = train_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        self.g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=lr,
            betas=(beta1, beta2),
        )

        self.d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=lr,
            betas=(beta1, beta2),
        )

        self.current_epoch = 0

    def train_discriminator(
        self,
        real_images: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one Discriminator update."""

        batch_size = real_images.size(0)

        real_images = real_images.to(self.device)
        labels = labels.to(self.device)

        # Generate fake images without allowing gradients to update G.
        noise = torch.randn(
            batch_size,
            self.generator.latent_dim,
            device=self.device,
        )

        with torch.no_grad():
            fake_images = self.generator(noise, labels)

        self.d_optimizer.zero_grad(set_to_none=True)

        real_logits = self.discriminator(real_images, labels)
        fake_logits = self.discriminator(fake_images, labels)

        loss = discriminator_loss(
            real_logits,
            fake_logits,
        )

        loss.backward()
        self.d_optimizer.step()

        return loss.detach()

    def train_generator(
        self,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Perform one Generator update."""

        batch_size = labels.size(0)

        labels = labels.to(self.device)

        noise = torch.randn(
            batch_size,
            self.generator.latent_dim,
            device=self.device,
        )

        self.g_optimizer.zero_grad(set_to_none=True)

        fake_images = self.generator(noise, labels)
        fake_logits = self.discriminator(fake_images, labels)

        loss = generator_loss(fake_logits)

        loss.backward()
        self.g_optimizer.step()

        return loss.detach()

    def train_epoch(self) -> Dict[str, float]:
        """Train both networks for one complete epoch."""

        self.generator.train()
        self.discriminator.train()

        total_d_loss = 0.0
        total_g_loss = 0.0
        num_batches = 0

        for real_images, labels in self.train_loader:
            d_loss = self.train_discriminator(
                real_images,
                labels,
            )

            g_loss = self.train_generator(labels)

            total_d_loss += d_loss.item()
            total_g_loss += g_loss.item()
            num_batches += 1

        if num_batches == 0:
            raise ValueError("Training DataLoader is empty.")

        self.current_epoch += 1

        return {
            "d_loss": total_d_loss / num_batches,
            "g_loss": total_g_loss / num_batches,
        }

    def train(
        self,
        num_epochs: int,
        save_every: int = 1,
    ) -> None:
        """Run the complete training procedure."""

        if num_epochs <= 0:
            raise ValueError("num_epochs must be greater than zero.")

        for epoch in range(num_epochs):
            metrics = self.train_epoch()

            print(
                f"Epoch [{epoch + 1}/{num_epochs}] "
                f"D Loss: {metrics['d_loss']:.4f} "
                f"G Loss: {metrics['g_loss']:.4f}"
            )

            if save_every > 0 and (epoch + 1) % save_every == 0:
                checkpoint_state = {
                    "epoch": self.current_epoch,
                    "generator_state_dict": self.generator.state_dict(),
                    "discriminator_state_dict": self.discriminator.state_dict(),
                    "g_optimizer_state_dict": self.g_optimizer.state_dict(),
                    "d_optimizer_state_dict": self.d_optimizer.state_dict(),
                    "g_loss": metrics["g_loss"],
                    "d_loss": metrics["d_loss"],
                }

                save_checkpoint(
                    state=checkpoint_state,
                    is_best=False,
                    checkpoint_dir=self.checkpoint_dir,
                    filename=f"checkpoint_epoch_{epoch + 1}.pth",
                )


def create_trainer(
    train_loader: DataLoader,
    device: Optional[torch.device] = None,
    checkpoint_dir: str = "checkpoints",
) -> Trainer:
    """Create a Trainer using the project's default model configuration."""

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    generator = Generator()
    discriminator = Discriminator()

    return Trainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    print(
        "train.py provides the Trainer class. "
        "Use a DataLoader and call Trainer.train()."
    )