# MambaOCR: Efficient Scene Text Recognition with LoRA

This project implements a state-of-the-art Optical Character Recognition (OCR) system combining **ResNet34**, **Mamba (State Space Models)**, and **CTC**. It features **Adapter-based Fine-Tuning** for efficient training of the vision backbone, allowing for high performance with controlled parameter updates.


### Key Components Flow:
1.  **Image Preprocessing**: Images are resized to a fixed height (32px) while maintaining aspect ratio, then padded to a max width.
2.  **Feature Extraction (Vision)**:
    *   **Backbone**: A pre-trained **ResNet34** extracts visual features.
    *   **Adapter Injection**: **Bottleneck Adapters** are injected into the convolutional layers.
    *   **Freeze Strategy**: The bulk of the ResNet backbone is **frozen**. Training is restricted to:
        *   The inserted Adapters
        *   The final convolutional block (`layer4`)
        *   The last projection layer
    *   **Stride Patching**: Standard CNNs downsample both height and width. Strides are patched to downsample Height (to 1) but **preserve Width**, ensuring a long enough sequence for text recognition.
3.  **Sequence Modeling (Language)**:
    *   The 2D feature map `[B, C, 1, W]` is converted to a 1D sequence `[B, W, C]`.
    *   **Mamba Blocks** process this sequence to model long-range dependencies between characters.
4.  **Decoding**:
    *   A **CTC (Connectionist Temporal Classification)** head maps the sequence to character probabilities.
    *   The final output is decoded into text, handling repeated characters and blanks automatically.

## Features

*   **Efficient Architecture**: Replaces heavy RNNs/LSTMs with **Mamba**, offering linear scaling with sequence length.
*   **Smart Fine-Tuning**: Uses **Adapters** and partial unfreezing (Layer4) to adapt the ImageNet-trained backbone to OCR tasks without destroying pre-trained knowledge.
*   **Robust Vision**: Uses **ResNet34** (ImageNet pre-trained) as a powerful feature extractor.
*   **Mixed Precision**: Full support for FP16 (AMP) training.
*   **Production Ready**: Includes inference scripts, checkpoint management, and modular configuration.

## 📂 Project Structure

```
ocr_project/
├── configs/
│   └── config.py       # Central configuration (Hyperparams, Paths, LoRA settings)
├── data/
│   ├── dataset.py      # Custom Dataset & DataLoader logic
│   └── ...
├── models/
│   ├── cnn_backbone.py # ResNet34 with Adapter implementation
│   ├── mamba_encoder.py# Mamba block definitions
│   └── ocr_model.py    # Main MambaOCR assembly
├── train.py            # Training loop with freezing strategy & AMP
├── infer.py            # Inference script for testing images
├── utils.py            # Metrics (CER/WER), Decoders, Logging
└── requirements.txt    # Dependencies
```

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


### 1. Configuration
Edit `configs/config.py`  set paths and parameters:
```python
self.data_dir = "path/to/your/data"
self.use_lora = True 
self.batch_size = 32
```

### 2. Training
Start the training pipeline. The script will automatically freeze the backbone and enable LoRA gradients.
```bash
python train.py
```

### 3. Inference
Run inference on a single image or a folder:
```bash
python infer.py
```
*(Ensure you point to a valid checkpoint in `infer.py`)*

## Data Generation
This project includes a powerful synthetic data generator `create_dummy_data.py` useful for training and sanity checking.
*   **Features**: Random rotation, Gaussian blur, noise injection, and variable fonts/backgrounds.
*   **Usage**: Great for verifying code functionality and learning basic text patterns. For production-grade accuracy, mixing in real-world datasets (MJSynth, etc.) is recommended.

## Performance & Training details
*   **Frozen Params**: Most of ResNet34 Base.
*   **Trainable Params**: ResNet Adapters, ResNet Layer4, Mamba (with LoRA), Final Classifier.
*   **Loss Tracking**: The training loop tracks partial loss and Validation CER. Expect CER to drop to ~0.07 on synthetic data.
