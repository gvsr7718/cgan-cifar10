"""Test script for the Conditional Generator.

Validates the Generator architecture, forward pass, output shape, value range,
class conditioning effect, and noise sensitivity on CPU and optionally GPU.
"""

import os
import sys
import yaml
import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.generator import Generator


def run_tests():
    print("=" * 70)
    print("  Conditional Generator — Validation Tests")
    print("=" * 70)

    # ── 1. Load configuration ──────────────────────────────────────────────
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    latent_dim = config["model"]["latent_dim"]       # 100
    num_classes = config["model"]["num_classes"]      # 10
    image_channels = config["model"]["image_channels"]  # 3
    image_size = config["model"]["image_size"]        # 32
    embedding_dim = 50  # Recommended default

    print(f"\nConfiguration loaded:")
    print(f"  latent_dim      = {latent_dim}")
    print(f"  num_classes     = {num_classes}")
    print(f"  embedding_dim   = {embedding_dim}")
    print(f"  image_channels  = {image_channels}")
    print(f"  image_size      = {image_size}")
    print(f"  combined_dim    = {latent_dim + embedding_dim}")

    all_passed = True
    batch_size = 8

    # ── 2. Create Generator ────────────────────────────────────────────────
    gen = Generator(
        latent_dim=latent_dim,
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        image_channels=image_channels,
        image_size=image_size,
    )
    gen.eval()  # Set to eval mode for deterministic BatchNorm behaviour

    total_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
    print(f"\nGenerator architecture:")
    print(gen)
    print(f"\nTotal trainable parameters: {total_params:,}")

    # ── 3. CPU forward pass ────────────────────────────────────────────────
    print("\n" + "-" * 70)
    print("TEST: CPU forward pass")
    print("-" * 70)

    torch.manual_seed(0)
    noise = torch.randn(batch_size, latent_dim)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])  # One label per CIFAR-10 class (first 8)

    with torch.no_grad():
        output = gen(noise, labels)

    # Verify shape
    expected_shape = torch.Size([batch_size, image_channels, image_size, image_size])
    shape_ok = output.shape == expected_shape
    print(f"  Output shape: {output.shape}  (expected {expected_shape})  {'PASS' if shape_ok else 'FAIL'}")
    if not shape_ok:
        all_passed = False

    # Verify finite values
    finite_ok = torch.isfinite(output).all().item()
    print(f"  All values finite: {finite_ok}  {'PASS' if finite_ok else 'FAIL'}")
    if not finite_ok:
        all_passed = False

    # Verify range [-1, 1] with tolerance
    tolerance = 1e-6
    min_val = output.min().item()
    max_val = output.max().item()
    range_ok = (min_val >= -1.0 - tolerance) and (max_val <= 1.0 + tolerance)
    print(f"  Output min: {min_val:.6f}  max: {max_val:.6f}  in [-1, 1]: {'PASS' if range_ok else 'FAIL'}")
    if not range_ok:
        all_passed = False

    cpu_passed = shape_ok and finite_ok and range_ok
    print(f"  CPU test result: {'PASS' if cpu_passed else 'FAIL'}")

    # ── 4. GPU forward pass (if available) ─────────────────────────────────
    print("\n" + "-" * 70)
    print("TEST: GPU forward pass")
    print("-" * 70)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        gen_gpu = gen.to(device)
        noise_gpu = noise.to(device)
        labels_gpu = labels.to(device)

        with torch.no_grad():
            output_gpu = gen_gpu(noise_gpu, labels_gpu)

        gpu_shape_ok = output_gpu.shape == expected_shape
        gpu_finite_ok = torch.isfinite(output_gpu).all().item()
        gpu_min = output_gpu.min().item()
        gpu_max = output_gpu.max().item()
        gpu_range_ok = (gpu_min >= -1.0 - tolerance) and (gpu_max <= 1.0 + tolerance)
        gpu_passed = gpu_shape_ok and gpu_finite_ok and gpu_range_ok

        print(f"  Output shape: {output_gpu.shape}  {'PASS' if gpu_shape_ok else 'FAIL'}")
        print(f"  All values finite: {gpu_finite_ok}  {'PASS' if gpu_finite_ok else 'FAIL'}")
        print(f"  Output min: {gpu_min:.6f}  max: {gpu_max:.6f}  in [-1, 1]: {'PASS' if gpu_range_ok else 'FAIL'}")
        print(f"  GPU test result: {'PASS' if gpu_passed else 'FAIL'}")

        if not gpu_passed:
            all_passed = False

        # Move back to CPU for remaining tests
        gen = gen.cpu()
    else:
        print("  CUDA not available — GPU test skipped")

    # ── 5. Label conditioning test ─────────────────────────────────────────
    print("\n" + "-" * 70)
    print("TEST: Label conditioning (same noise, different labels)")
    print("-" * 70)

    gen.eval()
    torch.manual_seed(42)
    fixed_noise = torch.randn(batch_size, latent_dim)

    labels_a = torch.zeros(batch_size, dtype=torch.long)     # All class 0 (airplane)
    labels_b = torch.ones(batch_size, dtype=torch.long) * 5  # All class 5 (dog)

    with torch.no_grad():
        out_a = gen(fixed_noise, labels_a)
        out_b = gen(fixed_noise, labels_b)

    max_diff = (out_a - out_b).abs().max().item()
    outputs_differ = max_diff > 1e-6
    print(f"  Outputs differ with different labels: {outputs_differ}  {'PASS' if outputs_differ else 'FAIL'}")
    print(f"  Max absolute difference: {max_diff:.6f}")
    if not outputs_differ:
        all_passed = False

    # ── 6. Noise sensitivity test ──────────────────────────────────────────
    print("\n" + "-" * 70)
    print("TEST: Noise sensitivity (same labels, different noise)")
    print("-" * 70)

    fixed_labels = torch.full((batch_size,), 3, dtype=torch.long)  # All class 3 (cat)

    torch.manual_seed(100)
    noise_1 = torch.randn(batch_size, latent_dim)
    torch.manual_seed(200)
    noise_2 = torch.randn(batch_size, latent_dim)

    with torch.no_grad():
        out_1 = gen(noise_1, fixed_labels)
        out_2 = gen(noise_2, fixed_labels)

    noise_max_diff = (out_1 - out_2).abs().max().item()
    noise_outputs_differ = noise_max_diff > 1e-6
    print(f"  Outputs differ with different noise: {noise_outputs_differ}  {'PASS' if noise_outputs_differ else 'FAIL'}")
    print(f"  Max absolute difference: {noise_max_diff:.6f}")
    if not noise_outputs_differ:
        all_passed = False

    # ── Summary ────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    run_tests()
