"""
CBAM: Convolutional Block Attention Module (2018, Woo et al.)
=============================================================

NEW TECHNOLOGY added to improve deepfake detection accuracy.

PURPOSE:
    CBAM helps the model FOCUS on the most important parts of the image.
    Instead of treating every pixel equally, CBAM learns:
    1. WHICH feature channels matter (Channel Attention)
    2. WHERE in the image to look (Spatial Attention)

    For deepfake detection, this means:
    - Channel attention: "Pay more attention to edge/noise channels"
    - Spatial attention: "Focus on face boundaries where blending artifacts appear"

WHY IT HELPS:
    Standard CNNs process every spatial region equally. But deepfake artifacts
    are highly localized — they appear at face boundaries, around eyes/mouth,
    or in specific frequency bands. CBAM lets the network zoom into these
    regions automatically.

PROVEN RESULTS:
    - Used in "Attentive ResNet101" deepfake detectors achieving >96% accuracy
    - Adds negligible computational overhead (<1% extra params)
    - Compatible with any CNN backbone (EfficientNet, ConvNeXt, etc.)

REFERENCE:
    Woo, S., Park, J., Lee, J.Y. and Kweon, I.S., 2018.
    "CBAM: Convolutional Block Attention Module." ECCV 2018.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """
    Channel Attention: Learn WHICH feature channels are important.

    How it works:
        1. Take a feature map [B, C, H, W]
        2. Apply Global Avg Pooling AND Global Max Pooling → [B, C, 1, 1] each
        3. Pass both through a shared MLP (bottleneck: C → C/ratio → C)
        4. Add them together and apply Sigmoid → [B, C, 1, 1]
        5. Multiply with input → channels with high attention get amplified

    Think of it as: "Which of the 768 feature detectors found something useful?"
    """

    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # Global Max Pooling

        # Shared MLP bottleneck: compress channels then expand
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),  # Compress
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),  # Expand back
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))  # [B, C, 1, 1]
        max_out = self.fc(self.max_pool(x))  # [B, C, 1, 1]
        out = self.sigmoid(avg_out + max_out)  # [B, C, 1, 1]
        return x * out  # Scale each channel by its attention weight


class SpatialAttention(nn.Module):
    """
    Spatial Attention: Learn WHERE in the image to focus.

    How it works:
        1. Take a feature map [B, C, H, W]
        2. Compute channel-wise Average and Max → [B, 2, H, W]
        3. Apply a 7x7 conv → [B, 1, H, W]
        4. Sigmoid → attention map showing which SPATIAL regions matter
        5. Multiply with input → important regions get amplified

    Think of it as: "Where in the image are the suspicious artifacts?"
    For deepfakes: highlights face boundaries, eye regions, blending seams.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2  # Same padding to preserve spatial dims
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        combined = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attention = self.sigmoid(self.conv(combined))  # [B, 1, H, W]
        return x * attention  # Amplify important spatial regions


class CBAM(nn.Module):
    """
    Complete CBAM module: Channel Attention → Spatial Attention.

    Sequentially refines features by first recalibrating channels,
    then focusing on important spatial locations.

    Usage:
        cbam = CBAM(in_planes=768)  # Match your feature map channels
        features = backbone(x)       # [B, 768, 7, 7]
        refined = cbam(features)     # [B, 768, 7, 7] — attention-weighted

    Parameters:
        in_planes: number of input channels (must match feature map channels)
        ratio: channel reduction ratio for the bottleneck MLP (default=16)
        kernel_size: spatial attention conv kernel size (default=7)
    """

    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)  # Refine channels: "which features matter?"
        x = self.spatial_attention(x)  # Refine spatial: "where to look?"
        return x
