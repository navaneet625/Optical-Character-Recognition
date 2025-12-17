# MambaOCR: Scene Text Recognition with LoRA

This project implements an OCR system using ResNet34, Mamba, and CTC. It uses adapter-based fine-tuning to train efficiently on the vision backbone.

### How it works
1. **Preprocessing**: Images are resized to 32px height, preserving aspect ratio, and padded.
2. **Vision Backbone**: Uses a pre-trained ResNet34 with injected adapters. freezes most of the network and only train the adapters, partial layer4, and projections.
3. **Mamba Encoder**: Models the sequence of features to understand character context.
4. **Decoding**: Uses CTC to convert the sequence into text.

## Features
- **Mamba**: Uses state space models instead of RNNs for better scaling.
- **Adapters**: Fine-tunes the backbone efficiently without breaking pre-trained weights.
- **ResNet34**: Strong baseline feature extractor.
- **FP16**: Supports mixed precision training.

## Project Structure
- `configs/`: Configuration settings.
- `data/`: Dataset loading and processing.
- `models/`: Model architecture (ResNet + Mamba).
- `train.py`: Main training script.
- `infer.py`: Inference script.
- `utils.py`: Metrics and decoding helpers.
- `gen_data.py`: Synthetic data generator.

## Installation
Clone the repo and install dependencies:
```bash
git clone <repo_url>
cd ocr_project
pip install -r requirements.txt
```
*Note: `mamba-ssm` needs a GPU.*

## Usage

### Configuration
Check `configs/config.py` to set your data paths and batch size.

### Training
Run the training script (it handles freezing automatically):
```bash
python train.py
```

### Inference
Test on images:
```bash
python infer.py
```

## Data Generation
Use `gen_data.py` to create synthetic training data with random fonts and noise.