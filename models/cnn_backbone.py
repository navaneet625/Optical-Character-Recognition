import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import math

class LoRAConv2d(nn.Module):
    def __init__(self, base_layer: nn.Conv2d, rank: int = 8, alpha: int = 16):
        super().__init__()
        if not isinstance(base_layer, nn.Conv2d):
            raise ValueError(f"LoRAConv2d expects nn.Conv2d, got {type(base_layer)}")

        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / max(1, rank)

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # LoRA Branch
        # We assume the base layer stride handles the geometry. 
        # We create a 1x1 LoRA adapter that matches the stride if necessary.
        if self.base_layer.stride != (1, 1) and self.base_layer.stride != 1:
            # If base downsamples, LoRA must downsample too.
            # We use a 1x1 conv with the same stride to be efficient.
            self.lora_A = nn.Conv2d(
                base_layer.in_channels, rank, 
                kernel_size=1, stride=base_layer.stride, padding=0, bias=False
            )
        else:
            self.lora_A = nn.Conv2d(
                base_layer.in_channels, rank, 
                kernel_size=1, stride=1, padding=0, bias=False
            )

        self.lora_B = nn.Conv2d(
            rank, base_layer.out_channels, 
            kernel_size=1, stride=1, padding=0, bias=False
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.base_layer(x) + (self.lora_B(self.lora_A(x)) * self.scaling)

def apply_lora_to_model(module: nn.Module, rank: int = 8, alpha: int = 16):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            if not isinstance(child, LoRAConv2d):
                # Only apply LoRA to 1x1 point-wise convolutions (Standard for ConvNeXt)
                # This saves parameters and is more stable.
                if child.kernel_size == (1, 1) or child.kernel_size == 1:
                    lora_wrapper = LoRAConv2d(child, rank=rank, alpha=alpha)
                    setattr(module, name, lora_wrapper)
        else:
            apply_lora_to_model(child, rank=rank, alpha=alpha)

class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self, output_channel=512, lora_r=8):
        super().__init__()
        
        print(f"Loading ConvNeXt-Tiny with LoRA (rank={lora_r})...")
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        base_model = convnext_tiny(weights=weights)
        self.backbone = base_model.features
        
        # -------------------------------------------------------------
        # CORRECT PATCHING STRATEGY (Preserve Pretrained Weights)
        # -------------------------------------------------------------
        # 1. STEM (Layer 0): Stride 4. KEEP IT. 
        #    Output Width: 320 -> 80. (Good enough, and we keep weights!)
        
        # 2. Downsample Layer 1 (Layer 2): Stride 2. 
        #    We modify this IN-PLACE to preserve weights.
        if hasattr(self.backbone[2][1], 'stride'):
            # Change stride to (2, 1) to keep width 80
            self.backbone[2][1].stride = (2, 1) 

        # 3. Downsample Layer 2 (Layer 4): Stride 2.
        if hasattr(self.backbone[4][1], 'stride'):
            # Change stride to (2, 1)
            self.backbone[4][1].stride = (2, 1)

        # 4. Downsample Layer 3 (Layer 6): Stride 2.
        if hasattr(self.backbone[6][1], 'stride'):
            # Change stride to (2, 1) or even (1, 1) to be safe
            self.backbone[6][1].stride = (2, 1)

        # Result:
        # Width: 320 -> Stem(80) -> D1(80) -> D2(80) -> D3(80).
        # Final Width = 80 steps. (Plenty for CTC).
        # Height: 32 -> Stem(8) -> D1(4) -> D2(2) -> D3(1).
        # Final Height = 1. (Perfect).
        
        # -------------------------------------------------------------

        # Apply LoRA only to the 1x1 layers
        apply_lora_to_model(self.backbone, rank=lora_r, alpha=lora_r*2)
        
        # Project to target dimension (ConvNeXt Tiny = 768)
        self.last_conv = nn.Conv2d(768, output_channel, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)
        x = self.last_conv(x)
        return x