"""Smoke test for FID calculation module."""

import os
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.evaluation.fid import (
    calculate_frechet_distance,
    compute_fid_score,
)
from src.models.generator import Generator


def run_tests():
    print("=" * 70)
    print("  FID Calculation — Smoke Test")
    print("=" * 70)

    all_passed = True
    device = torch.device("cpu")

    # ---------------------------------------------------------------
    # Test calculate_frechet_distance
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: calculate_frechet_distance (identical Gaussians)")
    print("-" * 70)
    
    import numpy as np
    
    mu1 = np.zeros(2048)
    sigma1 = np.eye(2048)
    
    # Distance between identical distributions should be 0
    dist = calculate_frechet_distance(mu1, sigma1, mu1, sigma1)
    
    dist_ok = np.isclose(dist, 0.0, atol=1e-5)
    
    print(f"  Distance: {dist:.6f}")
    print(f"  Distance is 0 for identical distributions: {'PASS' if dist_ok else 'FAIL'}")
    
    if not dist_ok:
        all_passed = False

    # ---------------------------------------------------------------
    # Test compute_fid_score
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: compute_fid_score")
    print("-" * 70)

    # Use a small number to avoid memory issues and keep test fast
    num_samples = 4
    
    generator = Generator()
    
    # Synthetic real images
    # Shape: [num_samples, 3, 32, 32], range: [-1, 1]
    real_images = torch.rand((num_samples, 3, 32, 32)) * 2 - 1
    real_labels = torch.randint(0, 10, (num_samples,))
    
    dataset = TensorDataset(real_images, real_labels)
    dataloader = DataLoader(dataset, batch_size=2)
    
    fid_score = compute_fid_score(
        generator=generator,
        real_dataloader=dataloader,
        num_images=num_samples,
        device=device
    )
    
    score_ok = isinstance(fid_score, float) and np.isfinite(fid_score) and fid_score >= 0.0
    
    print(f"  FID Score: {fid_score:.4f}")
    print(f"  Valid scalar FID score: {'PASS' if score_ok else 'FAIL'}")
    
    if not score_ok:
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
