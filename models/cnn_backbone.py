import torch
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights

class ConvAdapter(nn.Module):
    def __init__(self, channels, bottleneck_dim=32):
        super().__init__()
        self.down = nn.Conv2d(channels, bottleneck_dim, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.up = nn.Conv2d(bottleneck_dim, channels, kernel_size=1, bias=False)

        # Zero-init for stable training
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))

class AdapterWrappedBlock(nn.Module):
    def __init__(self, original_block, adapter_dim=32):
        super().__init__()

        # copy original block structure
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        self.relu = original_block.relu

        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2

        self.downsample = original_block.downsample
        self.stride = original_block.stride

        # Adapter inserted after conv1
        channels = original_block.conv1.out_channels
        self.adapter = ConvAdapter(channels, bottleneck_dim=adapter_dim)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # adapter applied
        out = self.adapter(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNetFeatureExtractor(nn.Module):
    def __init__(self, output_channel=512, adapter_dim=32):
        super().__init__()

        print(f"Loading ResNet-34 adapter_dim={adapter_dim}...")

        # Load pretrained weights
        weights = ResNet34_Weights.IMAGENET1K_V1
        self.resnet = resnet34(weights=weights)

        # Inject adapters into each block
        for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
            layer = getattr(self.resnet, layer_name)
            for i, block in enumerate(layer):
                layer[i] = AdapterWrappedBlock(block, adapter_dim)

        print("Adapters inserted successfully.")

        # OCR stride tweaks
        self.resnet.layer3[0].conv1.stride = (2, 1)
        if self.resnet.layer3[0].downsample is not None:
            self.resnet.layer3[0].downsample[0].stride = (2, 1)

        self.resnet.layer4[0].conv1.stride = (2, 1)
        if self.resnet.layer4[0].downsample is not None:
            self.resnet.layer4[0].downsample[0].stride = (2, 1)

        # Backbone feature extractor
        self.backbone = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
            self.resnet.maxpool,
            self.resnet.layer1,
            self.resnet.layer2,
            self.resnet.layer3,
            self.resnet.layer4,
        )

        # Final projection to cnn_out (default 512)
        self.last_conv = nn.Conv2d(512, output_channel, kernel_size=1)

    def forward(self, x):
        # x is [B, 3, H, W] guaranteed by dataset
        x = self.backbone(x)
        x = self.last_conv(x)
        return x
