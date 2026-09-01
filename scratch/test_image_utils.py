"""Smoke tests for image utilities."""

import os
import sys
import tempfile
import numpy as np

import torch
from PIL import Image

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
)

from src.utils.image_utils import (
    denormalize_tensor,
    tensor_to_pil,
    save_image_grid,
)


def run_tests():
    print("=" * 70)
    print("  Image Utilities — Smoke Test")
    print("=" * 70)

    all_passed = True

    # ---------------------------------------------------------------
    # Test denormalize_tensor
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: denormalize_tensor")
    print("-" * 70)

    test_tensor = torch.tensor([-1.0, 0.0, 1.0, 2.0, -2.0])
    expected = torch.tensor([0.0, 0.5, 1.0, 1.0, 0.0])
    
    result = denormalize_tensor(test_tensor)
    
    denorm_ok = torch.allclose(result, expected)
    print(f"  Denormalization and clamping: {'PASS' if denorm_ok else 'FAIL'}")
    
    if not denorm_ok:
        all_passed = False

    # ---------------------------------------------------------------
    # Test tensor_to_pil
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: tensor_to_pil")
    print("-" * 70)

    # Tensor in range [-1, 1]
    img_tensor = torch.ones(3, 32, 32) * -1.0 # Should map to 0 (black)
    img_tensor[:, 0, 0] = 1.0 # Should map to 255 (white)
    
    pil_img = tensor_to_pil(img_tensor)
    
    pil_type_ok = isinstance(pil_img, Image.Image)
    pil_mode_ok = pil_img.mode == "RGB"
    pil_size_ok = pil_img.size == (32, 32)
    
    # Check values
    arr = np.array(pil_img)
    val_white_ok = np.all(arr[0, 0] == 255)
    val_black_ok = np.all(arr[1, 1] == 0)
    
    print(f"  Returns PIL Image: {'PASS' if pil_type_ok else 'FAIL'}")
    print(f"  Correct mode (RGB): {'PASS' if pil_mode_ok else 'FAIL'}")
    print(f"  Correct size: {'PASS' if pil_size_ok else 'FAIL'}")
    print(f"  Values scaled correctly: {'PASS' if (val_white_ok and val_black_ok) else 'FAIL'}")

    if not (pil_type_ok and pil_mode_ok and pil_size_ok and val_white_ok and val_black_ok):
        all_passed = False

    # ---------------------------------------------------------------
    # Test save_image_grid
    # ---------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: save_image_grid")
    print("-" * 70)

    batch_tensor = torch.rand(12, 3, 32, 32) * 2.0 - 1.0 # Random in [-1, 1]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_grid.png")
        save_image_grid(batch_tensor, save_path, nrow=4)
        
        file_exists = os.path.isfile(save_path)
        print(f"  File created successfully: {'PASS' if file_exists else 'FAIL'}")
        
        if not file_exists:
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
