"""Quick verification script for the Conditional Discriminator."""

import os
import sys
import yaml
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.discriminator import Discriminator

def run_tests():
    print("=" * 70)
    print("  Conditional Discriminator — Verification")
    print("=" * 70)

    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    image_channels = config["model"]["image_channels"]
    num_classes = config["model"]["num_classes"]
    image_size = config["model"]["image_size"]
    embedding_dim = 50

    disc = Discriminator(
        image_channels=image_channels,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        image_size=image_size,
    )
    disc.eval()

    total_params = sum(p.numel() for p in disc.parameters() if p.requires_grad)
    print(f"\nDiscriminator architecture:")
    print(disc)
    print(f"\nTotal trainable parameters: {total_params:,}")

    all_passed = True
    batch_size = 8

    # --- CPU forward pass ---
    print("\n" + "-" * 70)
    print("TEST: CPU forward pass")
    print("-" * 70)

    torch.manual_seed(0)
    images = torch.randn(batch_size, image_channels, image_size, image_size)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])

    with torch.no_grad():
        output = disc(images, labels)

    expected_shape = torch.Size([batch_size, 1])
    shape_ok = output.shape == expected_shape
    finite_ok = torch.isfinite(output).all().item()
    print(f"  Output shape: {output.shape}  (expected {expected_shape})  {'PASS' if shape_ok else 'FAIL'}")
    print(f"  All values finite: {finite_ok}  {'PASS' if finite_ok else 'FAIL'}")
    print(f"  Output values: {output.squeeze().tolist()}")
    if not (shape_ok and finite_ok):
        all_passed = False

    # --- Label conditioning ---
    print("\n" + "-" * 70)
    print("TEST: Label conditioning (same image, different labels)")
    print("-" * 70)

    torch.manual_seed(42)
    fixed_images = torch.randn(batch_size, image_channels, image_size, image_size)
    labels_a = torch.zeros(batch_size, dtype=torch.long)
    labels_b = torch.full((batch_size,), 5, dtype=torch.long)

    with torch.no_grad():
        out_a = disc(fixed_images, labels_a)
        out_b = disc(fixed_images, labels_b)

    max_diff = (out_a - out_b).abs().max().item()
    label_cond_ok = max_diff > 1e-6
    print(f"  Outputs differ: {label_cond_ok}  max_diff={max_diff:.6f}  {'PASS' if label_cond_ok else 'FAIL'}")
    if not label_cond_ok:
        all_passed = False

    # --- Image sensitivity ---
    print("\n" + "-" * 70)
    print("TEST: Image sensitivity (different images, same labels)")
    print("-" * 70)

    fixed_labels = torch.full((batch_size,), 3, dtype=torch.long)
    torch.manual_seed(100)
    imgs_1 = torch.randn(batch_size, image_channels, image_size, image_size)
    torch.manual_seed(200)
    imgs_2 = torch.randn(batch_size, image_channels, image_size, image_size)

    with torch.no_grad():
        out_1 = disc(imgs_1, fixed_labels)
        out_2 = disc(imgs_2, fixed_labels)

    img_max_diff = (out_1 - out_2).abs().max().item()
    img_sens_ok = img_max_diff > 1e-6
    print(f"  Outputs differ: {img_sens_ok}  max_diff={img_max_diff:.6f}  {'PASS' if img_sens_ok else 'FAIL'}")
    if not img_sens_ok:
        all_passed = False

    # --- Invalid image_size ---
    print("\n" + "-" * 70)
    print("TEST: Invalid image_size rejection")
    print("-" * 70)
    try:
        Discriminator(image_size=64)
        print("  ValueError NOT raised  FAIL")
        all_passed = False
    except ValueError:
        print("  ValueError raised correctly  PASS")

    # --- Summary ---
    print("\n" + "=" * 70)
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 70)
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    run_tests()
