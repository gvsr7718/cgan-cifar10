"""Smoke-test for the Trainer class in src/training/train.py."""

import os
import sys
import copy

import torch
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.generator import Generator
from src.models.discriminator import Discriminator
from src.training.train import Trainer


def run_tests():
    print("=" * 70)
    print("  Trainer — Smoke Test")
    print("=" * 70)

    all_passed = True
    device = torch.device("cpu")

    # 1. Create a small synthetic dataset
    print("\n" + "-" * 70)
    print("TEST: Dataset and Model Initialization")
    print("-" * 70)

    num_samples = 8
    batch_size = 4
    
    # Random images in range [-1, 1]
    synthetic_images = torch.rand((num_samples, 3, 32, 32)) * 2 - 1
    # Random labels in [0, 9]
    synthetic_labels = torch.randint(0, 10, (num_samples,))

    dataset = TensorDataset(synthetic_images, synthetic_labels)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize models
    generator = Generator()
    discriminator = Discriminator()

    # Save initial parameters to check if they change
    init_g_params = [p.clone().detach() for p in generator.parameters()]
    init_d_params = [p.clone().detach() for p in discriminator.parameters()]

    trainer = Trainer(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        device=device,
        checkpoint_dir="checkpoints_test" # temporary directory
    )

    print("  Models and Trainer instantiated: PASS")

    # 2. Run one train_epoch()
    print("\n" + "-" * 70)
    print("TEST: Run train_epoch()")
    print("-" * 70)

    metrics = trainer.train_epoch()
    d_loss = metrics.get("d_loss")
    g_loss = metrics.get("g_loss")

    print(f"  D Loss: {d_loss}")
    print(f"  G Loss: {g_loss}")

    # 3. Verify returned losses
    d_loss_ok = isinstance(d_loss, float) and not torch.isnan(torch.tensor(d_loss))
    g_loss_ok = isinstance(g_loss, float) and not torch.isnan(torch.tensor(g_loss))

    print(f"  D loss is finite scalar: {'PASS' if d_loss_ok else 'FAIL'}")
    print(f"  G loss is finite scalar: {'PASS' if g_loss_ok else 'FAIL'}")

    if not (d_loss_ok and g_loss_ok):
        all_passed = False

    # 4. Verify trainer.current_epoch == 1
    epoch_ok = trainer.current_epoch == 1
    print(f"  Trainer current_epoch == 1: {'PASS' if epoch_ok else 'FAIL'}")
    
    if not epoch_ok:
        all_passed = False

    # 5. Verify parameters changed
    print("\n" + "-" * 70)
    print("TEST: Parameter Updates")
    print("-" * 70)

    g_params_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(init_g_params, generator.parameters()))
    d_params_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(init_d_params, discriminator.parameters()))

    print(f"  Generator parameters changed: {'PASS' if g_params_changed else 'FAIL'}")
    print(f"  Discriminator parameters changed: {'PASS' if d_params_changed else 'FAIL'}")

    if not (g_params_changed and d_params_changed):
        all_passed = False

    # Summary
    print("\n" + "=" * 70)
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
