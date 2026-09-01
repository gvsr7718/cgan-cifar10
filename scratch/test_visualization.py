"""Smoke tests for visualization utilities."""

import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.generator import Generator
from src.utils.visualization import (
    plot_class_conditional_grid,
    plot_training_losses,
    plot_fid_curve,
)


def run_tests():
    print("=" * 70)
    print("  Visualization Utilities — Smoke Test")
    print("=" * 70)

    all_passed = True

    # ---------------------------------------------------------------
    # 1. Model initialization
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Generator Initialization")
    print("-" * 70)

    device = torch.device("cpu")
    generator = Generator().to(device)

    print("  Generator created: PASS")

    # ---------------------------------------------------------------
    # 2. Class-conditional grid
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Class-Conditional Grid")
    print("-" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        grid_path = os.path.join(temp_dir, "grid.png")

        plot_class_conditional_grid(
            generator=generator,
            num_classes=10,
            samples_per_class=2,
            save_path=grid_path,
        )

        grid_ok = os.path.isfile(grid_path) and os.path.getsize(grid_path) > 0

        print(
            f"  Grid image created: "
            f"{'PASS' if grid_ok else 'FAIL'}"
        )

        if not grid_ok:
            all_passed = False

    # ---------------------------------------------------------------
    # 3. Training loss curve
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Training Loss Plot")
    print("-" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        loss_path = os.path.join(temp_dir, "losses.png")

        g_losses = [2.0, 1.5, 1.2, 0.9]
        d_losses = [1.8, 1.4, 1.1, 0.8]

        plot_training_losses(
            g_losses=g_losses,
            d_losses=d_losses,
            save_path=loss_path,
        )

        loss_ok = os.path.isfile(loss_path) and os.path.getsize(loss_path) > 0

        print(
            f"  Loss plot created: "
            f"{'PASS' if loss_ok else 'FAIL'}"
        )

        if not loss_ok:
            all_passed = False

    # ---------------------------------------------------------------
    # 4. FID curve
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: FID Curve")
    print("-" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        fid_path = os.path.join(temp_dir, "fid.png")

        epochs = [10, 20, 30, 40]
        fid_scores = [150.0, 120.0, 95.0, 80.0]

        plot_fid_curve(
            epochs=epochs,
            fid_scores=fid_scores,
            save_path=fid_path,
        )

        fid_ok = os.path.isfile(fid_path) and os.path.getsize(fid_path) > 0

        print(
            f"  FID plot created: "
            f"{'PASS' if fid_ok else 'FAIL'}"
        )

        if not fid_ok:
            all_passed = False

    # ---------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------
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