import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# -------------------------
# 1. Adapter Module (Fixed)
# -------------------------
class ConvAdapter(nn.Module):
    def __init__(self, channels, bottleneck_dim=32):
        super().__init__()
        
        # Down-projection (Random Init is fine)
        self.down = nn.Conv2d(channels, bottleneck_dim, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        
        # Up-projection
        self.up = nn.Conv2d(bottleneck_dim, channels, kernel_size=1, bias=False)
        
        # --- CRITICAL FIX: Zero-Initialization ---
        # This ensures the adapter output is 0 at the start.
        # The model behaves exactly like the pretrained ResNet initially.
        nn.init.zeros_(self.up.weight)
        if self.up.bias is not None:
            nn.init.zeros_(self.up.bias)

    def forward(self, x):
        # Residual connection is internal to the adapter
        return x + self.up(self.act(self.down(x)))

# -------------------------
# 2. Patching Logic (Verified)
# -------------------------
def add_adapter_to_resnet_block(block, bottleneck_dim=32):
    """
    Modifies a torchvision BasicBlock in-place to insert an adapter.
    Location: Between Relu1 and Conv2.
    """
    # 1. Create the adapter
    # We use block.conv1.out_channels to match dimensions dynamically
    channels = block.conv1.out_channels
    block.adapter = ConvAdapter(channels, bottleneck_dim)

    # 2. Define the new forward pass
    def forward_with_adapter(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # >>> Adapter Injection <<<
        out = self.adapter(out) 

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    # 3. Bind the method to the instance (Monkey Patching)
    # This replaces the .forward() method ONLY for this specific block instance
    block.forward = forward_with_adapter.__get__(block, block.__class__)

# -------------------------
# 3. Full Model Wrapper
# -------------------------
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_channel=512, adapter_dim=32):
        super().__init__()

        # A. Load Pretrained
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.resnet = resnet18(weights=weights)

        # B. Inject Adapters
        # We iterate through all modules, finding BasicBlocks
        for name, module in self.resnet.named_modules():
            if module.__class__.__name__ == "BasicBlock":
                # Scale adapter size relative to channels (optional, but good practice)
                # Or keep fixed 32 as you did.
                add_adapter_to_resnet_block(module, bottleneck_dim=adapter_dim)

        # C. OCR Modifications (Preserve Width)
        # Layer 3 & 4 strides
        self.resnet.layer3[0].conv1.stride = (2, 1)
        self.resnet.layer3[0].downsample[0].stride = (2, 1)
        self.resnet.layer4[0].conv1.stride = (2, 1)
        self.resnet.layer4[0].downsample[0].stride = (2, 1)

        # D. Feature Pyramid extraction (Remove FC)
        self.backbone = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4
        )

        # E. Final Projection
        self.last_conv = nn.Conv2d(512, output_channel, kernel_size=1)

    def forward(self, x):
        # x: [B, 1, 32, W] (Grayscale)
        x = x.repeat(1, 3, 1, 1) 
        x = self.backbone(x)
        x = self.last_conv(x)
        return x