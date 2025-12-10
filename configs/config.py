import torch
import os

class Config:
    def __init__(self):

        self.vocab = "0123456789abcdefghijklmnopqrstuvwxyz" 
  
        self.blank_idx = 0 
        
        # Image Dimensions
        self.img_height = 32
        self.img_width = 320 

        # Paths (Verify these exist!)
        self.data_dir = "data/" 
        self.images_dir = os.path.join(self.data_dir, "images")
        self.labels_file = os.path.join(self.data_dir, "labels", "labels.csv")
        self.checkpoint_dir = "checkpoints/"
        os.makedirs(self.checkpoint_dir, exist_ok=True) 

        # --------------------------
        # 2. Model Configuration
        # --------------------------
        # ConvNeXt Backbone
        self.cnn_out = 768       # Must match Mamba's hidden size for efficiency
        self.adapter_dim = 32    # Size of LoRA/Adapter bottleneck

        # Mamba Encoder
        self.mamba_pretrained = "state-spaces/mamba-130m-hf"
        self.mamba_d_model = 768 # The 130m model is fixed at 768 dim
        self.mamba_layers = 4    
        self.use_lora = True     

        # --------------------------
        # 3. Training Configuration
        # --------------------------
        # Batch Size
        # CRITICAL FIX: Changed 1 -> 32
        # Batch size 1 makes training unstable (noisy gradients) and extremely slow.
        # On a Colab T4 GPU, you can easily handle 32 or 64.
        self.batch_size = 32 
        
        # Learning Rate Strategy
        self.learning_rate = 5e-4  # Slightly lowered for stability with Mamba
        
        # Duration
        self.epochs = 1
        
        # Optimization
        self.weight_decay = 1e-2
        self.gradient_clip_val = 1.0 
        
        # System
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_workers = 2   
        self.mixed_precision = True