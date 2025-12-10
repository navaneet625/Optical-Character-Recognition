import torch
import torch.nn as nn
from .cnn_backbone import ConvNeXtFeatureExtractor
from .mamba_encoder import MambaEncoder

class MambaOCR(nn.Module):
    def __init__(self, vocab_size, cnn_out=768, n_layers=4, adapter_dim=8):
        super().__init__()
        
        # Vision Backbone
        self.cnn = ConvNeXtFeatureExtractor(output_channel=cnn_out, lora_r=adapter_dim)
        
        # Sequence Encoder
        self.encoder = MambaEncoder(input_dim=cnn_out, n_layers=n_layers, use_lora=True, lora_rank=adapter_dim)
        
        # Classifier
        mamba_dim = self.encoder.config.hidden_size
        self.classifier = nn.Linear(mamba_dim, vocab_size)

    def forward(self, x):
        # 1. Extract Features [B, 768, H, W]
        features = self.cnn(x)
        
        # 2. Force Height to 1 (Fixes the crash if H=2)
        # [B, 768, H, W] -> [B, 768, 1, W]
        features = nn.functional.adaptive_avg_pool2d(features, (1, None))
        
        # 3. Prepare for Mamba
        # Squeeze H: [B, 768, 1, W] -> [B, 768, W]
        features = features.squeeze(2)
        
        # Permute for Sequence: [B, 768, W] -> [B, W, 768]
        features = features.permute(0, 2, 1)

        # 4. Encoder & Classifier
        enc_out = self.encoder(features)
        logits = self.classifier(enc_out)

        return logits