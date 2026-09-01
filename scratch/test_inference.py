"""Smoke tests for cGAN inference utilities."""

import os
import sys

import torch

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from src.models.generator import Generator
from src.inference import generate_samples


def run_tests():
    print("=" * 70)
    print("  Inference Utilities — Smoke Test")
    print("=" * 70)

    all_passed = True
    device = torch.device("cpu")

    print("\n" + "-" * 70)
    print("TEST: Generator Initialization")
    print("-" * 70)

    generator = Generator().to(device)
    print("  Generator created: PASS")

    print("\n" + "-" * 70)
    print("TEST: Generate Class-Conditioned Samples")
    print("-" * 70)

    images = generate_samples(
        generator=generator,
        class_idx=3,
        num_samples=8,
        device=device,
    )

    print(f"  Output shape: {list(images.shape)}")

    shape_ok = images.shape == (8, 3, 32, 32)
    finite_ok = torch.isfinite(images).all().item()
    range_ok = images.min().item() >= -1.0 and images.max().item() <= 1.0

    print(
        f"  Correct output shape: "
        f"{'PASS' if shape_ok else 'FAIL'}"
    )
    print(
        f"  All values finite: "
        f"{'PASS' if finite_ok else 'FAIL'}"
    )
    print(
        f"  Values in [-1, 1]: "
        f"{'PASS' if range_ok else 'FAIL'}"
    )

    if not (shape_ok and finite_ok and range_ok):
        all_passed = False

    print("\n" + "-" * 70)
    print("TEST: Class Conditioning")
    print("-" * 70)

    images_class_0 = generate_samples(
        generator=generator,
        class_idx=0,
        num_samples=4,
        device=device,
    )

    images_class_9 = generate_samples(
        generator=generator,
        class_idx=9,
        num_samples=4,
        device=device,
    )

    class_outputs_ok = (
        images_class_0.shape == images_class_9.shape
        and torch.isfinite(images_class_0).all()
        and torch.isfinite(images_class_9).all()
    )

    print(
        f"  Different classes generate valid outputs: "
        f"{'PASS' if class_outputs_ok else 'FAIL'}"
    )

    if not class_outputs_ok:
        all_passed = False

    print("\n" + "-" * 70)
    print("TEST: Invalid Arguments")
    print("-" * 70)

    invalid_class_ok = False

    try:
        generate_samples(
            generator=generator,
            class_idx=10,
            num_samples=2,
            device=device,
        )
    except ValueError:
        invalid_class_ok = True

    print(
        f"  Invalid class rejected: "
        f"{'PASS' if invalid_class_ok else 'FAIL'}"
    )

    if not invalid_class_ok:
        all_passed = False

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