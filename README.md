# Conditional Image Generation using Generative Adversarial Networks (cGAN) with FID-Based Evaluation

## 1. Project Description
This project investigates class-conditional synthetic image synthesis on the **CIFAR-10** benchmark dataset using a **Conditional Generative Adversarial Network (cGAN)** architecture implemented in **PyTorch**. Unlike unconditional GANs, the cGAN conditions both the Generator and Discriminator on class labels ($y \in \{0, 1, \dots, 9\}$), enabling directed generation of specific object categories. The model's fidelity, diversity, and class alignment are evaluated using both quantitative benchmarks (**Fréchet Inception Distance (FID)** and **Class Consistency Scores**) and qualitative grid visualizations. An interactive **Streamlit** web application serves as a demonstration interface for real-time conditional sampling.

---

## 2. Main Objectives
- **Targeted Image Synthesis**: Build and train a conditional GAN capable of generating coherent $32 \times 32 \times 3$ RGB images corresponding to user-specified CIFAR-10 classes.
- **Quantitative Quality Assessment**: Implement and evaluate generative performance using the **Fréchet Inception Distance (FID)** to compare feature distributions of real and generated samples.
- **Class Alignment Verification**: Measure the semantic accuracy of conditioned generations using an independent pre-trained CIFAR-10 classifier to compute a **Class Consistency Score**.
- **Modular & Reproducible Framework**: Establish an end-to-end machine learning engineering workflow with centralized YAML configuration, reproducible random seeding, clean model checkpointing, and separate training/evaluation pipelines.
- **Interactive Interface**: Deploy an interactive Streamlit application allowing users to generate and explore synthetic images across classes in real-time.

---

## 3. Planned Architecture

The system utilizes a class-conditional adversarial framework based on deep convolutional networks:

```
[Latent Vector z ~ N(0, I)] ──┐
                              ├──> [Embedding / Concat] ──> [Generator Network] ──────┐
[Class Label y] ──────────────┤                                                       │
                              │                                                       v
                              │                                           [Synthetic Image G(z, y)]
                              │                                                       │
                              ├──> [Embedding / Concat] ──> [Discriminator Network] <─┘ or [Real Image x]
                                                                        │
                                                                        v
                                                          [Real / Fake Validity Score]
```

### Components
1. **Conditional Generator ($G$)**:
   - **Inputs**: Latent noise vector $z \in \mathbb{R}^{d_{latent}}$ (e.g., $d=100$) sampled from $\mathcal{N}(0, I)$, concatenated with a learned dense class embedding vector $e(y)$.
   - **Layers**: Transposed convolutional layers (`ConvTranspose2d`) with Batch Normalization and ReLU activations, mapping low-dimensional representations to spatial feature maps.
   - **Output**: $32 \times 32 \times 3$ RGB image tensor scaled to $[-1, 1]$ via a `Tanh` activation function.

2. **Conditional Discriminator ($D$)**:
   - **Inputs**: $32 \times 32 \times 3$ image (real or generated) conditioned on class label $y$ (via spatial embedding projection or channel concatenation).
   - **Layers**: Strided convolutional layers (`Conv2d`) with LeakyReLU activations and normalization layers (e.g., Spectral Normalization or Batch Normalization).
   - **Output**: Validity score indicating probability that the input image is a genuine sample of class $y$.

3. **Loss Formulation**:
   - Minimax adversarial objective with Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) and soft label smoothing for stabilized training dynamics.

4. **Auxiliary Classifier (Evaluation)**:
   - A pre-trained CIFAR-10 classification model used during evaluation to classify generated images $G(z, y)$ and verify that the intended class $y$ is accurately represented.

---

## 4. Project Structure

```
cgan-cifar10/
│
├── README.md                          # Project documentation and specifications
├── requirements.txt                   # Python environment dependencies
├── .gitignore                         # Git exclusion rules for artifacts, data, and cache
│
├── configs/
│   └── config.yaml                    # Centralized hyperparameter and experiment configuration
│
├── data/
│   └── README.md                      # Dataset acquisition and directory documentation
│
├── src/
│   ├── __init__.py                    # Core package initializer
│   │
│   ├── data/
│   │   ├── __init__.py                # Data subpackage initializer
│   │   ├── dataset.py                 # CIFAR-10 dataset loading and DataLoader construction
│   │   └── preprocessing.py           # Augmentation, scaling, and normalization pipelines
│   │
│   ├── models/
│   │   ├── __init__.py                # Models subpackage initializer
│   │   ├── generator.py               # Conditional Generator network architecture
│   │   ├── discriminator.py           # Conditional Discriminator network architecture
│   │   └── classifier.py              # Pre-trained / evaluation classifier architecture
│   │
│   ├── training/
│   │   ├── __init__.py                # Training subpackage initializer
│   │   ├── train.py                   # cGAN training loop orchestrator
│   │   ├── losses.py                  # Adversarial loss functions and regularization
│   │   └── checkpoint.py              # Model checkpoint saving and restoration
│   │
│   ├── evaluation/
│   │   ├── __init__.py                # Evaluation subpackage initializer
│   │   ├── fid.py                     # Fréchet Inception Distance calculation pipeline
│   │   ├── class_consistency.py       # Classifier-based semantic consistency evaluation
│   │   └── metrics.py                 # Metric logging, tracking, and summary reporting
│   │
│   ├── utils/
│   │   ├── __init__.py                # Utilities subpackage initializer
│   │   ├── seed.py                    # Deterministic random seed management
│   │   ├── visualization.py           # Image grids, loss curves, and metric plots
│   │   └── image_utils.py             # Tensor transformations, unnormalization, and disk I/O
│   │
│   └── inference.py                   # High-level API and CLI for checkpoint image generation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      # EDA, class balance, and dataset visualization
│   ├── 02_train_cgan.ipynb            # Interactive GPU training notebook (Colab-ready)
│   └── 03_evaluate_model.ipynb        # Quantitative evaluation, metric plots, and analysis
│
├── checkpoints/
│   └── .gitkeep                       # Directory for saved model weights (.pt / .pth)
│
├── results/
│   ├── samples/
│   │   └── .gitkeep                   # Generated synthetic image grids across training
│   ├── graphs/
│   │   └── .gitkeep                   # Loss curves and FID progression charts
│   └── metrics/
│       └── .gitkeep                   # Quantitative evaluation JSON reports and logs
│
└── app/
    └── streamlit_app.py               # Interactive Streamlit web demo application
```

---

## 5. Technology Stack

| Category | Technology / Library | Purpose |
| :--- | :--- | :--- |
| **Language** | Python 3.10+ | Primary programming language |
| **Deep Learning Framework** | PyTorch (`torch`) | Tensor computation, neural network modules, autograd |
| **Computer Vision** | Torchvision (`torchvision`) | CIFAR-10 dataset ingestion, image transforms |
| **Numerical Processing** | NumPy (`numpy`), SciPy (`scipy`) | Array operations and covariance matrix computations |
| **Visualization** | Matplotlib (`matplotlib`), Pillow (`PIL`) | Plotting loss curves, metric trajectories, image grid rendering |
| **Evaluation Metrics** | `pytorch-fid`, `torchmetrics` | Fréchet Inception Distance and evaluation metrics |
| **Configuration** | PyYAML (`pyyaml`) | Declarative experiment and hyperparameter configs |
| **Progress & Utilities** | `tqdm` | Training progress visualization in terminal and notebooks |
| **Web Interface** | Streamlit (`streamlit`) | Interactive web application for real-time sampling |
| **Compute Infrastructure** | Google Colab (GPU) / Local CUDA | Hardware acceleration for model training |
| **Version Control** | Git / GitHub | Code management and version control |

---

## 6. Planned Evaluation Metrics

1. **Fréchet Inception Distance (FID)**:
   - Quantifies the statistical distance between feature representations of real CIFAR-10 test images and generated images extracted from an intermediate layer of an Inception network:
     $$\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)$$
   - Lower values indicate higher image fidelity and diverse distribution matching.

2. **Class Consistency Score**:
   - Assesses conditional alignment by passing synthetic images $G(z, y)$ through a trained CIFAR-10 classifier:
     $$\text{Consistency} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left(C(G(z_i, y_i)) = y_i\right)$$
   - Measures whether the generated images accurately reflect the target class semantics.

3. **Qualitative Evaluation**:
   - $10 \times 10$ image grid visualizations displaying sample generations for all 10 CIFAR-10 classes (`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`).
   - Latent space linear interpolation between noise vectors and class embeddings to evaluate continuity and avoid mode collapse.

---

## 7. Development Workflow

The development follows a structured, modular approach:
1. **Scaffolding & Architecture Design**: Establish repository structure, package boundaries, and configuration schema.
2. **Dataset & Preprocessing Pipeline**: Build robust data loaders with standard CIFAR-10 normalization and augmentation.
3. **Model Construction**: Implement the conditional Generator, Discriminator, and auxiliary classifier architectures with unit tests on tensor shapes.
4. **Adversarial Training**: Train the cGAN using balanced optimization, periodic sample checkpointing, and loss tracking.
5. **Quantitative Evaluation**: Compute FID benchmarks and class consistency scores against the CIFAR-10 test set.
6. **Application Deployment**: Build the interactive Streamlit interface for accessible model inference and exploration.

---

## 8. Google Colab GPU Training Workflow

To facilitate training on high-performance accelerators without local GPU hardware constraints:

1. **Environment Setup**:
   - Open `notebooks/02_train_cgan.ipynb` in Google Colab.
   - Select a GPU runtime (Runtime > Change runtime type > T4 GPU / V100 GPU).
2. **Repository Synchronization**:
   - Clone the Git repository or upload the project files to the Colab environment.
   - Install dependencies: `!pip install -r requirements.txt`.
3. **Training Execution**:
   - Execute training via the notebook or via CLI: `!python src/training/train.py --config configs/config.yaml`.
4. **Artifact Persistence**:
   - Checkpoints (`checkpoints/best_generator.pt`) and evaluation outputs (`results/`) are saved periodically and can be synced with Google Drive or downloaded to the local repository.
