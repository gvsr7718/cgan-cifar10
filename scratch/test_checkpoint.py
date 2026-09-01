"""Validation tests for checkpoint save/load utilities."""

import os
import sys
import tempfile

import torch

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.generator import Generator
from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    extract_state_dict,
)


def run_tests():
    print("=" * 70)
    print("  Checkpoint Utilities — Validation Tests")
    print("=" * 70)

    all_passed = True

    # ------------------------------------------------------------
    # 1. Create model and optimizer
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Create model and optimizer")
    print("-" * 70)

    model = Generator()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    print("  Generator created: PASS")
    print("  Optimizer created: PASS")

    # ------------------------------------------------------------
    # 2. Extract state dict
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Extract clean state dict")
    print("-" * 70)

    state_dict = extract_state_dict(model)

    state_ok = (
        isinstance(state_dict, dict)
        and len(state_dict) > 0
        and all(isinstance(v, torch.Tensor) for v in state_dict.values())
    )

    print(f"  State dict contains {len(state_dict)} tensors")
    print(f"  Valid state dict: {'PASS' if state_ok else 'FAIL'}")

    if not state_ok:
        all_passed = False

    # ------------------------------------------------------------
    # 3. Save checkpoint
    # ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("TEST: Save checkpoint")
    print("-" * 70)

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_state = {
            "epoch": 5,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": 0.42,
        }

        checkpoint_path = save_checkpoint(
            state=checkpoint_state,
            is_best=True,
            checkpoint_dir=temp_dir,
            filename="checkpoint.pth",
        )

        save_ok = os.path.isfile(checkpoint_path)
        best_path = os.path.join(temp_dir, "best_model.pth")
        best_ok = os.path.isfile(best_path)

        print(f"  Checkpoint created: {'PASS' if save_ok else 'FAIL'}")
        print(f"  Best checkpoint created: {'PASS' if best_ok else 'FAIL'}")

        if not save_ok or not best_ok:
            all_passed = False

        # --------------------------------------------------------
        # 4. Load checkpoint
        # --------------------------------------------------------
        print("\n" + "-" * 70)
        print("TEST: Load checkpoint")
        print("-" * 70)

        new_model = Generator()
        new_optimizer = torch.optim.Adam(
            new_model.parameters(),
            lr=0.0002,
        )

        loaded = load_checkpoint(
            checkpoint_path=checkpoint_path,
            model=new_model,
            optimizer=new_optimizer,
        )

        load_ok = (
            loaded["epoch"] == 5
            and abs(loaded["loss"] - 0.42) < 1e-8
        )

        print(f"  Epoch restored: {loaded['epoch']}  {'PASS' if loaded['epoch'] == 5 else 'FAIL'}")
        print(f"  Loss restored: {loaded['loss']}  {'PASS' if abs(loaded['loss'] - 0.42) < 1e-8 else 'FAIL'}")
        print(f"  Checkpoint metadata restored: {'PASS' if load_ok else 'FAIL'}")

        if not load_ok:
            all_passed = False

        # --------------------------------------------------------
        # 5. Verify model weights match
        # --------------------------------------------------------
        print("\n" + "-" * 70)
        print("TEST: Verify restored model weights")
        print("-" * 70)

        weights_match = all(
            torch.equal(
                model.state_dict()[key],
                new_model.state_dict()[key],
            )
            for key in model.state_dict()
        )

        print(
            f"  Model weights restored correctly: "
            f"{'PASS' if weights_match else 'FAIL'}"
        )

        if not weights_match:
            all_passed = False

        # --------------------------------------------------------
        # 6. Missing checkpoint handling
        # --------------------------------------------------------
        print("\n" + "-" * 70)
        print("TEST: Missing checkpoint handling")
        print("-" * 70)

        missing_ok = False

        try:
            load_checkpoint(
                checkpoint_path=os.path.join(temp_dir, "does_not_exist.pth"),
                model=new_model,
            )
        except FileNotFoundError:
            missing_ok = True

        print(
            f"  FileNotFoundError raised correctly: "
            f"{'PASS' if missing_ok else 'FAIL'}"
        )

        if not missing_ok:
            all_passed = False

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------
    print("\n" + "=" * 70)

    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")

    print("=" * 70)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    run_tests()