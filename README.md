# MambaOCR: Efficient Scene Text Recognition with LoRA

This project implements a state-of-the-art Optical Character Recognition (OCR) system combining **ConvNeXt**, **Mamba (State Space Models)**, and **CTC**. It features **Low-Rank Adaptation (LoRA)** for efficient fine-tuning of the vision backbone, allowing for high performance with minimal trainable parameters.


### Key Components Flow:
1.  **Image Preprocessing**: Images are resized to a fixed height (32px) while maintaining aspect ratio, then padded to a max width.
2.  **Feature Extraction (Vision)**:
    *   **Backbone**: A pre-trained **ConvNeXt-Tiny** extracts visual features.
    *   **LoRA Injection**: Instead of training the full backbone, we inject **Low-Rank Adapters (LoRA)** into the convolutional layers. The base weights are **frozen**, and only the small rank-8 adapters are trained.
    *   **Stride Patching**: Standard CNNs downsample both height and width. We patch the strides to downsample Height (to 1) but **preserve Width**, ensuring we have a long enough sequence for text recognition.
3.  **Sequence Modeling (Language)**:
    *   The 2D feature map `[B, C, 1, W]` is converted to a 1D sequence `[B, W, C]`.
    *   **Mamba Blocks** process this sequence to model long-range dependencies between characters.
4.  **Decoding**:
    *   A **CTC (Connectionist Temporal Classification)** head maps the sequence to character probabilities.
    *   The final output is decoded into text, handling repeated characters and blanks automatically.

## 🚀 Features

*   **⚡ Efficient Architecture**: Replaces heavy RNNs/LSTMs with **Mamba**, offering linear scaling with sequence length.
*   **🧠 LoRA Fine-Tuning**: Implements custom **LoRAConv2d** layers. Only ~1-5% of parameters are trainable, drastically reducing memory usage and preventing catastrophic forgetting.
*   **👀 Robust Vision**: Uses **ConvNeXt-Tiny** (ImageNet pre-trained) as a powerful feature extractor.
*   **🔄 Mixed Precision**: Full support for FP16 (AMP) training.
*   **🛠️ Production Ready**: Includes inference scripts, checkpoint management, and modular configuration.

## 📂 Project Structure

```
ocr_project/
├── configs/
│   └── config.py       # Central configuration (Hyperparams, Paths, LoRA settings)
├── data/
│   ├── dataset.py      # Custom Dataset & DataLoader logic
│   └── ...
├── models/
│   ├── cnn_backbone.py # ConvNeXt with custom LoRAConv2d implementation
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

## 🏃 Usage

### 1. Configuration
Edit `configs/config.py` to set your paths and parameters:
```python
self.data_dir = "path/to/your/data"
self.use_lora = True  # Enable/Disable LoRA
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

## 📊 Performance & LoRA
By using LoRA, we achieve comparable accuracy to full fine-tuning but with significantly faster convergence and lower VRAM requirements.
*   **Frozen Params**: ConvNeXt Base, Mamba Base.
*   **Trainable Params**: LoRA Adapters (Rank 8), Final Classifier.
