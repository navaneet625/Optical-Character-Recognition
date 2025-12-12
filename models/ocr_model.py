import torch
import torch.nn as nn
from .cnn_backbone import ResNetFeatureExtractor
from .mamba_encoder import MambaEncoder

class MambaOCR(nn.Module):
    def __init__(self, vocab_size, cnn_out=512, n_layers=4, adapter_dim=32, lora_rank=None):
        super().__init__()

        self.cnn = ResNetFeatureExtractor(
            output_channel=cnn_out,
            adapter_dim=adapter_dim
        )

        self.norm = nn.LayerNorm(cnn_out)

        # Use explicit lora_rank if provided, else default to adapter_dim
        real_lora_rank = lora_rank if lora_rank is not None else adapter_dim

        self.encoder = MambaEncoder(
            input_dim=cnn_out,
            n_layers=n_layers,
            use_lora=True,
            lora_rank=real_lora_rank
        )

        hidden_dim = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        # CNN: [B, 3, H, W] → [B, C, H', W']
        features = self.cnn(x)

        # Adaptive pool height → 1 (fixes mismatch for odd heights)
        features = nn.functional.adaptive_avg_pool2d(features, (1, None))

        # Remove height: [B, C, 1, W] → [B, C, W]
        features = features.squeeze(2)

        # Permute for sequence: [B, C, W] → [B, W, C]
        features = features.permute(0, 2, 1)

        # LayerNorm stabilization
        features = self.norm(features)

        # Mamba Encoder
        enc = self.encoder(features)

        # Final classifier: [B, W, vocab_size]
        logits = self.classifier(enc)

        return logits
