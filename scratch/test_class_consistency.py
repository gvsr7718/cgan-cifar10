"""Smoke tests for cGAN class consistency evaluation."""

import os
import sys

import torch

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from src.models.generator import Generator
from src.models.classifier import CIFAR10Classifier
from src.evaluation.class_consistency import (
    compute_class_consistency_score,
    compute_per_class_consistency,
    compute_confusion_matrix,
)


def run_tests():
    print("=" * 70)
    print("  Class Consistency — Smoke Test")
    print("=" * 70)

    all_passed = True

    device = torch.device("cpu")

    # Use small numbers to keep the smoke test fast.
    num_samples_per_class = 2

    # ---------------------------------------------------------------
    # Model initialization
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Model Initialization")
    print("-" * 70)

    generator = Generator()
    classifier = CIFAR10Classifier()

    print("  Generator created: PASS")
    print("  Classifier created: PASS")

    # ---------------------------------------------------------------
    # Overall consistency
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Overall Class Consistency")
    print("-" * 70)

    score = compute_class_consistency_score(
        generator=generator,
        classifier=classifier,
        num_samples_per_class=num_samples_per_class,
        device=device,
    )

    score_ok = (
        isinstance(score, float)
        and 0.0 <= score <= 1.0
        and torch.isfinite(torch.tensor(score)).item()
    )

    print(f"  Overall consistency: {score:.4f}")
    print(
        f"  Score in [0, 1]: "
        f"{'PASS' if score_ok else 'FAIL'}"
    )

    if not score_ok:
        all_passed = False

    # ---------------------------------------------------------------
    # Per-class consistency
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Per-Class Consistency")
    print("-" * 70)

    per_class = compute_per_class_consistency(
        generator=generator,
        classifier=classifier,
        num_samples_per_class=num_samples_per_class,
        device=device,
    )

    per_class_ok = (
        len(per_class) == 10
        and all(0.0 <= value <= 1.0 for value in per_class.values())
    )

    print(f"  Number of classes: {len(per_class)}")
    print(
        f"  Ten class scores returned: "
        f"{'PASS' if per_class_ok else 'FAIL'}"
    )

    for class_idx, value in per_class.items():
        print(f"    Class {class_idx}: {value:.4f}")

    if not per_class_ok:
        all_passed = False

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Confusion Matrix")
    print("-" * 70)

    confusion = compute_confusion_matrix(
        generator=generator,
        classifier=classifier,
        num_samples_per_class=num_samples_per_class,
        device=device,
    )

    confusion_shape_ok = confusion.shape == (10, 10)
    confusion_integer_ok = confusion.dtype == torch.long
    confusion_total_ok = (
        confusion.sum().item()
        == 10 * num_samples_per_class
    )

    print(f"  Matrix shape: {list(confusion.shape)}")
    print(
        f"  Shape is [10, 10]: "
        f"{'PASS' if confusion_shape_ok else 'FAIL'}"
    )
    print(
        f"  Integer counts: "
        f"{'PASS' if confusion_integer_ok else 'FAIL'}"
    )
    print(
        f"  Total predictions correct: "
        f"{'PASS' if confusion_total_ok else 'FAIL'}"
    )

    if not (
        confusion_shape_ok
        and confusion_integer_ok
        and confusion_total_ok
    ):
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