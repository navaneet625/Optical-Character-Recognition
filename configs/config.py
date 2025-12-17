import torch
import os

class Config:
    def __init__(self):
        # Data Paths
        # self.data_dir = "/kaggle/working/data"
        # self.images_dir = os.path.join(self.data_dir, "images")
        # self.labels_file = os.path.join(self.data_dir, "labels.txt")
        
        # self.checkpoint_dir = "/kaggle/working/checkpoints"
        # self.best_model_path = os.path.join(self.checkpoint_dir, "best_mamba_ocr.pth")
        
        # if not os.path.exists(self.checkpoint_dir):
        #      os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.data_dir = "data"  
        
        self.images_dir = os.path.join(self.data_dir, "images")
        self.labels_file = os.path.join(self.data_dir, "labels.txt")
        self.checkpoint_dir = "checkpoints"
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_mamba_ocr.pth")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Model & Vocab
        # STRICT VOCAB: 0-9 + A-Z (36 Char)
        self.vocab = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        self.blank_idx = 0 
        
        self.img_height = 32
        self.img_width = 320
        
        # ImageNet Stats
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        # Architecture
        self.cnn_out = 512
        self.adapter_dim = 64
        self.mamba_pretrained = "state-spaces/mamba-130m-hf"
        self.mamba_layers = 6
        self.use_lora = True
        self.lora_rank = 64
        self.mamba_dropout = 0.1

        self.mamba_dropout = 0.1

        # Training Hyperparameters
        self.batch_size = 256 
        self.epochs = 1 
        self.learning_rate = 3e-4   
        self.weight_decay = 1e-2
        self.gradient_clip_val = 1.0
        
        self.num_workers = 4
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.mixed_precision = True 