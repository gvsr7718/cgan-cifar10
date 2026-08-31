# Dataset Directory (`data/`)

This directory is designated for local storage of the **CIFAR-10** dataset.

## Dataset Overview
- **Name**: CIFAR-10 (Canadian Institute For Advanced Research)
- **Content**: 60,000 $32 \times 32$ color images across 10 mutually exclusive classes.
- **Class Mapping**:
  0. airplane
  1. automobile
  2. bird
  3. cat
  4. deer
  5. dog
  6. frog
  7. horse
  8. ship
  9. truck

## Data Splits
The dataset is cleanly separated into three non-overlapping subsets:
- **Training Set (45,000 images)**: Randomly sampled 90% of the official training set. Used to train the cGAN Generator and Discriminator.
- **Validation Set (5,000 images)**: Remaining 10% of the official training set. Used for hyperparameter tuning and early stopping.
- **Test Set (10,000 images)**: The official CIFAR-10 test set. Strictly reserved for final quantitative evaluation (FID and Class Consistency).

*Note: The train/validation split is deterministic, controlled by the `seed` parameter in `configs/config.yaml`.*

## Preprocessing and Normalization
To ensure stability during GAN training, especially when using a `Tanh` activation in the Generator output, all images undergo the following preprocessing pipeline:
1. **ToTensor**: Converts PIL images (shape $H \times W \times C$, values in $[0, 255]$) to PyTorch FloatTensors (shape $C \times H \times W$, values in $[0.0, 1.0]$).
2. **Normalize**: Normalizes each channel consistently with `mean=(0.5, 0.5, 0.5)` and `std=(0.5, 0.5, 0.5)`. This scales the pixel values from $[0.0, 1.0]$ to $[-1.0, 1.0]$.
   * Equation: `value = (value - mean) / std`
   * Denormalization for visualization uses: `value = (value * 0.5) + 0.5`

## DataLoader Configuration
The data pipeline constructs PyTorch `DataLoader` instances with the following properties (configurable via `configs/config.yaml`):
- **Batch Size**: 128 (default)
- **Workers**: 2 (default) for background data loading
- **Shuffling**: Training data is shuffled every epoch. Validation and test sets are *not* shuffled.
- **Drop Last**: The training dataloader drops the last incomplete batch (`drop_last=True`) to maintain consistent batch sizes for the Discriminator.

## Data Acquisition & Ingestion
The dataset is downloaded and extracted automatically when the data pipeline functions in `src/data/dataset.py` are explicitly called. It uses `torchvision.datasets.CIFAR10(root='data', download=True)`.

## Version Control Note
Raw batch archives and downloaded binary files (e.g., `cifar-10-batches-py/`, `*.tar.gz`) are excluded from Git tracking via `.gitignore` to keep the repository lightweight.
