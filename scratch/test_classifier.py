"""Smoke tests for the CIFAR-10 classifier."""

import os
import sys

import torch

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from src.models.classifier import (
    CIFAR10Classifier,
    classifier_loss,
    predict,
)


def run_tests():
    print("=" * 70)
    print("  CIFAR-10 Classifier — Smoke Test")
    print("=" * 70)

    all_passed = True

    device = torch.device("cpu")
    batch_size = 8

    # ---------------------------------------------------------------
    # Model creation
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Model Initialization")
    print("-" * 70)

    model = CIFAR10Classifier().to(device)

    print("  Classifier created: PASS")

    # ---------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Forward Pass")
    print("-" * 70)

    images = torch.randn(
        batch_size,
        3,
        32,
        32,
        device=device,
    )

    labels = torch.randint(
        0,
        10,
        (batch_size,),
        device=device,
    )

    logits = model(images)

    shape_ok = logits.shape == (batch_size, 10)
    finite_ok = torch.isfinite(logits).all().item()

    print(f"  Output shape: {list(logits.shape)}")
    print(
        f"  Output shape correct: "
        f"{'PASS' if shape_ok else 'FAIL'}"
    )
    print(
        f"  All logits finite: "
        f"{'PASS' if finite_ok else 'FAIL'}"
    )

    if not (shape_ok and finite_ok):
        all_passed = False

    # ---------------------------------------------------------------
    # Loss
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Classification Loss")
    print("-" * 70)

    loss = classifier_loss(logits, labels)

    loss_ok = (
        loss.ndim == 0
        and torch.isfinite(loss).item()
        and loss.item() > 0
    )

    print(f"  Loss: {loss.item():.6f}")
    print(
        f"  Valid scalar loss: "
        f"{'PASS' if loss_ok else 'FAIL'}"
    )

    if not loss_ok:
        all_passed = False

    # ---------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Prediction Helper")
    print("-" * 70)

    predictions, confidence = predict(model, images)

    prediction_shape_ok = predictions.shape == (batch_size,)
    confidence_shape_ok = confidence.shape == (batch_size,)
    prediction_range_ok = bool(
        ((predictions >= 0) & (predictions < 10)).all()
    )
    confidence_range_ok = bool(
        ((confidence >= 0) & (confidence <= 1)).all()
    )

    print(
        f"  Prediction shape correct: "
        f"{'PASS' if prediction_shape_ok else 'FAIL'}"
    )
    print(
        f"  Confidence shape correct: "
        f"{'PASS' if confidence_shape_ok else 'FAIL'}"
    )
    print(
        f"  Predictions in [0, 9]: "
        f"{'PASS' if prediction_range_ok else 'FAIL'}"
    )
    print(
        f"  Confidence in [0, 1]: "
        f"{'PASS' if confidence_range_ok else 'FAIL'}"
    )

    if not (
        prediction_shape_ok
        and confidence_shape_ok
        and prediction_range_ok
        and confidence_range_ok
    ):
        all_passed = False

    # ---------------------------------------------------------------
    # Backpropagation
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Backpropagation")
    print("-" * 70)

    model.train()

    logits = model(images)
    loss = classifier_loss(logits, labels)
    loss.backward()

    gradients_exist = any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"  Gradients computed: "
        f"{'PASS' if gradients_exist else 'FAIL'}"
    )

    if not gradients_exist:
        all_passed = False

    # ---------------------------------------------------------------
    # Parameter count
    # ---------------------------------------------------------------
    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"  Trainable parameters: {parameter_count:,}")

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