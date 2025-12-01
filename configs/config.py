import torch
import os

class Config:
    def __init__(self):
        # --------------------------
        # 1. Data Configuration
        # --------------------------
        # The characters the model can read. 
        # Note: If dataset has uppercase, add A-Z here!
        self.vocab = "0123456789abcdefghijklmnopqrstuvwxyz" 
        
        # CTC Special Tokens
        self.blank_idx = 0 
        
        # Image Dimensions
        self.img_height = 32   # Fixed (ResNet standard)
        self.img_width = 320   # Maximum width (images are resized proportionally)

        # Paths (Update these to your Kaggle Input paths)
        # Example: "/kaggle/input/mjsynth-dataset/mnt/ramdisk/max/90kDICT32px"
        self.data_dir = "data/" 
        self.images_dir = os.path.join(self.data_dir, "images")
        self.labels_file = os.path.join(self.data_dir, "labels", "labels.csv")
        self.checkpoint_dir = "checkpoints/" # Directory to save models 

        # --------------------------
        # 2. Model Configuration
        # --------------------------
        # ResNet Backbone
        self.cnn_out = 512
        self.adapter_dim = 32   # Bottleneck size for CNN Adapters

        # Mamba Encoder
        self.mamba_pretrained = "state-spaces/mamba-130m-hf"
        self.mamba_d_model = 768 # Fixed dim of the 130m model
        self.mamba_layers = 4    # Truncate to 4 layers (Fast & Light)
        self.use_lora = True     # Enable Low-Rank Adaptation

        # --------------------------
        # 3. Training Configuration
        # --------------------------
        # Batch Size
        # 2x T4 GPUs (16GB each) + Mixed Precision can handle 128-256 easily.
        self.batch_size = 1 
        
        # Learning Rate Strategy
        self.learning_rate = 1e-3  # Base LR for Classifier & Projectors
        
        # Duration
        self.epochs = 30           # 30-50 epochs is standard for fine-tuning
        
        # Optimization
        self.weight_decay = 1e-2
        self.gradient_clip_val = 1.0 # Critical for Mamba stability
        
        # System
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 4         # Critical for high GPU utilization
        self.mixed_precision = True  # Enable fp16 (AMP)