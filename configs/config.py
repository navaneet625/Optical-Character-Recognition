import torch
import os

class Config:
    def __init__(self):
        self.data_dir = "/kaggle/input/ocr-synthetic-dataset"
        self.images_dir = os.path.join(self.data_dir, "images")
        self.labels_file = os.path.join(self.data_dir, "labels.txt")
        self.checkpoint_dir = "checkpoints"
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_mamba_ocr.pth")
        
        # Create output directories if local
        if not self.data_dir.startswith("/kaggle"):
             os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Digits + Lowercase + Uppercase
        self.vocab = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.blank_idx = 0 
        
        self.img_height = 32
        self.img_width = 320
        # ImageNet Statistics
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # ResNet Backbone
        self.cnn_out = 512      # ResNet34 output
        self.adapter_dim = 64   
        
        # Mamba Encoder
        self.mamba_pretrained = "state-spaces/mamba-130m-hf"
        self.mamba_layers = 4
        self.use_lora = True
        self.lora_rank = 64   
        self.mamba_dropout = 0.1

        self.batch_size = 16
        self.epochs = 5
        self.learning_rate = 1e-4
        self.weight_decay = 1e-2
        self.gradient_clip_val = 1.0
        self.num_workers = 4
  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mixed_precision = True 
