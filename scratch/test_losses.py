"""Validation tests for cGAN adversarial loss functions."""

import os
import sys

import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.losses import discriminator_loss, generator_loss


def run_tests():
    print("=" * 70)
    print("  cGAN Loss Functions — Validation Tests")
    print("=" * 70)

    all_passed = True

    # ---------------------------------------------------------------
    # Test 1: Discriminator loss should be low when predictions
    # correctly classify real as real and fake as fake.
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Discriminator loss")
    print("-" * 70)

    good_real_logits = torch.full((8, 1), 10.0)
    good_fake_logits = torch.full((8, 1), -10.0)

    d_loss_good = discriminator_loss(
        good_real_logits,
        good_fake_logits,
    )

    print(f"  Correct predictions loss: {d_loss_good.item():.6f}")

    good_d_ok = d_loss_good.item() < 0.01
    print(
        f"  Loss is near zero: {good_d_ok} "
        f"{'PASS' if good_d_ok else 'FAIL'}"
    )

    if not good_d_ok:
        all_passed = False

    # ---------------------------------------------------------------
    # Test 2: Discriminator loss should be high when predictions
    # are completely wrong.
    # ---------------------------------------------------------------
    bad_real_logits = torch.full((8, 1), -10.0)
    bad_fake_logits = torch.full((8, 1), 10.0)

    d_loss_bad = discriminator_loss(
        bad_real_logits,
        bad_fake_logits,
    )

    print(f"  Completely wrong predictions loss: {d_loss_bad.item():.6f}")

    bad_d_ok = d_loss_bad.item() > 10.0
    print(
        f"  Loss is high: {bad_d_ok} "
        f"{'PASS' if bad_d_ok else 'FAIL'}"
    )

    if not bad_d_ok:
        all_passed = False

    # ---------------------------------------------------------------
    # Test 3: Generator loss should be low when the Discriminator
    # believes fake images are real.
    # ---------------------------------------------------------------
    good_fake_logits = torch.full((8, 1), 10.0)

    g_loss_good = generator_loss(good_fake_logits)

    print(
        f"\n  Generator loss with successful deception: "
        f"{g_loss_good.item():.6f}"
    )

    good_g_ok = g_loss_good.item() < 0.01
    print(
        f"  Loss is near zero: {good_g_ok} "
        f"{'PASS' if good_g_ok else 'FAIL'}"
    )

    if not good_g_ok:
        all_passed = False

    # ---------------------------------------------------------------
    # Test 4: Generator loss should be high when the Discriminator
    # correctly identifies fake images.
    # ---------------------------------------------------------------
    bad_fake_logits = torch.full((8, 1), -10.0)

    g_loss_bad = generator_loss(bad_fake_logits)

    print(
        f"  Generator loss when detected as fake: "
        f"{g_loss_bad.item():.6f}"
    )

    bad_g_ok = g_loss_bad.item() > 10.0
    print(
        f"  Loss is high: {bad_g_ok} "
        f"{'PASS' if bad_g_ok else 'FAIL'}"
    )

    if not bad_g_ok:
        all_passed = False

    # ---------------------------------------------------------------
    # Test 5: Losses must be scalar finite tensors.
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Loss output validity")
    print("-" * 70)

    scalar_ok = d_loss_good.ndim == 0 and g_loss_good.ndim == 0
    finite_ok = (
        torch.isfinite(d_loss_good).item()
        and torch.isfinite(g_loss_good).item()
    )

    print(
        f"  Discriminator loss is scalar: "
        f"{d_loss_good.ndim == 0} "
        f"{'PASS' if d_loss_good.ndim == 0 else 'FAIL'}"
    )

    print(
        f"  Generator loss is scalar: "
        f"{g_loss_good.ndim == 0} "
        f"{'PASS' if g_loss_good.ndim == 0 else 'FAIL'}"
    )

    print(
        f"  Losses are finite: {finite_ok} "
        f"{'PASS' if finite_ok else 'FAIL'}"
    )

    if not scalar_ok or not finite_ok:
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