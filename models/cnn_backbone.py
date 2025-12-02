import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
import math

# ------------------------------
# Robust LoRA Wrapper
# ------------------------------
class LoRAConv2d(nn.Module):
    def __init__(self, base_layer: nn.Conv2d, rank: int = 8, alpha: int = 16, use_spatial: bool = False):
        super().__init__()
        
        # Validate input
        if not isinstance(base_layer, nn.Conv2d):
            raise ValueError(f"LoRAConv2d expects nn.Conv2d, got {type(base_layer)}")

        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(1, rank)

        # Freeze base layer
        for p in self.base_layer.parameters():
            p.requires_grad = False

        # Extract Geometry
        in_ch = base_layer.in_channels
        out_ch = base_layer.out_channels
        kernel_size = base_layer.kernel_size
        stride = base_layer.stride
        padding = base_layer.padding
        dilation = base_layer.dilation
        groups = base_layer.groups

        # --- LoRA Branch Design ---
        # If stride > 1, we MUST handle it in lora_A to match dimensions.
        # We use a 1x1 kernel with stride if possible, or the original kernel if needed.
        
        if stride != (1, 1) or stride != 1:
            # If downsampling, we force lora_A to match the spatial reduction
            # Using the original kernel size is safest to match padding logic
            self.lora_A = nn.Conv2d(
                in_ch, rank, 
                kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, 
                groups=1, bias=False
            )
        else:
            # Standard 1x1 adapter for non-downsampling layers (Efficient)
            self.lora_A = nn.Conv2d(
                in_ch, rank, 
                kernel_size=1, stride=1, padding=0, 
                bias=False
            )

        # lora_B projects back to output channels (Always 1x1, stride 1)
        self.lora_B = nn.Conv2d(
            rank, out_ch, 
            kernel_size=1, stride=1, padding=0, 
            bias=False
        )

        # Init
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        # 1. Base Path (Frozen)
        base_out = self.base_layer(x)
        
        # 2. LoRA Path
        # lora_A handles stride/downsampling
        a_out = self.lora_A(x)
        # lora_B projects channel dimensions
        lora_out = self.lora_B(a_out) * self.scaling
        
        # Safety check for dimensions (Debug only)
        # if base_out.shape != lora_out.shape:
        #     print(f"Shape Mismatch! Base: {base_out.shape}, LoRA: {lora_out.shape}")
        
        return base_out + lora_out

# -----------------------------------------
# Recursion Helper
# -----------------------------------------
def apply_lora_to_model(module: nn.Module, rank: int = 8, alpha: int = 16):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Conv2d):
            # Check if it's already wrapped
            if not isinstance(child, LoRAConv2d):
                # Wrap it
                lora_wrapper = LoRAConv2d(child, rank=rank, alpha=alpha)
                setattr(module, name, lora_wrapper)
        else:
            apply_lora_to_model(child, rank=rank, alpha=alpha)

# -----------------------------------------
# Feature Extractor
# -----------------------------------------
class ConvNeXtFeatureExtractor(nn.Module):
    def __init__(self, output_channel=512, lora_r=8):
        super().__init__()
        
        print(f"🏗️ Loading ConvNeXt-Tiny with LoRA (r={lora_r})...")
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        base_model = convnext_tiny(weights=weights)
        self.backbone = base_model.features
        
        # 1. Patch Strides for OCR (Preserve Width)
        # Target the downsampling layers in stages 2 and 3
        # ConvNeXt structure: [0]Stem, [1]Stage1, [2]Down1, [3]Stage2, [4]Down2, ...
        # We modify indices 4 and 6 (Downsample layers)
        
        # Index 4 (Downsample 2->3)
        if hasattr(self.backbone[4][1], 'stride'):
             self.backbone[4][1].stride = (2, 1) # Down H only
             
        # Index 6 (Downsample 3->4)
        if hasattr(self.backbone[6][1], 'stride'):
             self.backbone[6][1].stride = (2, 1) # Down H only

        # 2. Apply LoRA
        apply_lora_to_model(self.backbone, rank=lora_r, alpha=lora_r*2)
        
        # 3. Final Projection
        self.last_conv = nn.Conv2d(768, output_channel, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = self.backbone(x)
        x = self.last_conv(x)
        return x