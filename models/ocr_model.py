import torch
import torch.nn as nn
from .cnn_backbone import ResNetFeatureExtractor
from .mamba_encoder import MambaEncoder

class MambaOCR(nn.Module):
    def __init__(self, vocab_size, cnn_out=512, n_layers=4):
        super().__init__()
        
        # 1. Vision Backbone
        self.cnn = ResNetFeatureExtractor(output_channel=cnn_out, adapter_dim=32)
        
        # 2. Sequence Encoder
        self.encoder = MambaEncoder(input_dim=cnn_out, n_layers=n_layers, use_lora=True)
        
        # 3. Classifier
        mamba_dim = self.encoder.config.hidden_size
        self.classifier = nn.Linear(mamba_dim, vocab_size)

    def forward(self, x):
        # [B, 1, H, W] -> [B, 512, 1, W']
        features = self.cnn(x)
        
        # [B, 512, 1, W'] -> [B, W', 512]
        features = features.squeeze(2).permute(0, 2, 1)

        # Forward Mamba (No mask needed for standard Mamba-CTC training)
        enc_out = self.encoder(features)

        # [B, W', Vocab]
        logits = self.classifier(enc_out)

        return logits