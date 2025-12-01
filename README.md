# MambaOCR: Efficient Scene Text Recognition

This project implements a modern Optical Character Recognition (OCR) system using a **CNN-Mamba-CTC** architecture. It leverages **Mamba (State Space Models)** for efficient sequence modeling, replacing traditional RNNs/LSTMs, combined with a ResNet backbone for feature extraction.

## 🏗️ Architecture Flow

The model follows a standard CRNN-like pipeline but substitutes the recurrent layers with Mamba blocks for better scalability and speed.

## 🚀 Features

*   **Backbone**: ResNet-based feature extractor.
*   **Encoder**: Bidirectional Mamba blocks for long-range sequence dependency.
*   **Decoder**: Connectionist Temporal Classification (CTC) for alignment-free training.
*   **Training**: Mixed Precision (AMP) support, OneCycleLR scheduler.
*   **Augmentations**: Albumentations pipeline (Rotation, Noise, Blur).

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone <repo_url>
    cd ocr_project
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: `mamba-ssm` requires a GPU with CUDA support.*

## 📂 Project Structure

```
ocr_project/
├── configs/        # Configuration parameters
├── data/           # Dataset loading and augmentations
├── models/         # Model architecture (CNN, Mamba, OCR)
├── train.py        # Training script
├── infer.py        # Inference script
├── utils.py        # Decoders, metrics, logging
└── requirements.txt
```

## 🏃 Usage

### Training

1.  Prepare your dataset and update `train.py` with your data paths.
2.  Run training:
    ```bash
    python train.py
    ```

### Inference

To run inference on a single image:

```bash
python infer.py
```

(Ensure you have a trained checkpoint in `checkpoints/` or update the path in `infer.py`)

## 📊 Configuration

Modify `configs/config.py` to adjust hyperparameters:

*   `img_height`, `img_width`: Input image dimensions.
*   `vocab`: Character set.
*   `batch_size`, `learning_rate`, `epochs`: Training settings.
*   `mamba_d_model`, `mamba_layers`: Model size.
