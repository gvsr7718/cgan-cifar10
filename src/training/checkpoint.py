"""Model checkpointing and state persistence utilities.

Provides helpers for saving and loading Generator/Discriminator model
weights, optimizer states, epoch counters, and training metadata.
"""

import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    checkpoint_dir: str,
    filename: str = "checkpoint.pth",
) -> str:
    """Save a training checkpoint.

    Args:
        state: Dictionary containing model/optimizer states and metadata.
        is_best: Whether this checkpoint represents the best model so far.
        checkpoint_dir: Directory where checkpoints should be saved.
        filename: Name of the regular checkpoint file.

    Returns:
        Path to the saved checkpoint.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)

    if is_best:
        best_path = os.path.join(checkpoint_dir, "best_model.pth")
        torch.save(state, best_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Load a checkpoint into a model and optionally an optimizer.

    Args:
        checkpoint_path: Path to the checkpoint file.
        model: Model whose weights should be restored.
        optimizer: Optional optimizer whose state should be restored.
        device: Device on which the checkpoint should be loaded.

    Returns:
        The complete checkpoint dictionary.

    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        KeyError: If the checkpoint does not contain model state.
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device if device is not None else "cpu",
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def extract_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    """Extract a clean CPU state dictionary from a model.

    Useful for deployment or inference exports.

    Args:
        model: PyTorch model.

    Returns:
        Dictionary containing detached CPU tensors.
    """
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }