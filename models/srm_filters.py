"""
SRM (Steganalysis Rich Model) Noise Filters
============================================

PURPOSE:
    Extract the INVISIBLE NOISE PATTERN from images.
    Real cameras leave consistent, predictable noise.
    Deepfake generators create synthetic noise that doesn't match.

ANALOGY:
    Imagine you have a painting and you want to check if it's authentic.
    Instead of looking at what the painting shows (a landscape, a face),
    you look at the CANVAS TEXTURE and BRUSHSTROKE PATTERNS underneath.
    That's what SRM filters do — they ignore the content and analyze the texture.

HOW SRM FILTERS WORK:
    Each filter is a small 5x5 grid of numbers (called a "kernel").
    When you slide this grid across an image, it:
    1. Multiplies each pixel by the corresponding number
    2. Sums everything up
    3. The result highlights certain patterns while cancelling others

    The key insight: ALL SRM kernels have coefficients that sum to 0.
    This means they CANCEL OUT smooth areas and PRESERVE only edges/noise.

    Example (simplified 3x3 version):
        [ 0, -1,  0]     This kernel detects vertical edges
        [ 0,  2,  0]     Smooth areas → output ≈ 0
        [ 0, -1,  0]     Edges → output ≠ 0

    We use 9 different kernels, each sensitive to different noise patterns:
    - Some detect horizontal edges
    - Some detect diagonal edges
    - Some detect corner patterns
    - Some detect complex noise signatures

    Real images have natural noise from the camera sensor.
    Fake images have SYNTHETIC noise from the generation algorithm.
    These filters make that difference visible.

ACADEMIC REFERENCE:
    Based on "Rich Models for Steganalysis of Digital Images"
    by Fridrich & Kodovsky (IEEE 2012)
    Originally designed for steganography detection, repurposed for deepfakes.

INPUTS:
    images: [batch_size, 3, H, W] — RGB images (any resolution)

OUTPUTS:
    residuals: [batch_size, 9, H, W] — 9 noise residual maps
        Each of the 9 maps highlights different noise patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ═══════════════════════════════════════════════════════════════
#  The 9 SRM Filter Kernels (Fixed math — not learned)
# ═══════════════════════════════════════════════════════════════

def get_srm_kernels():
    """
    Return the 9 standard SRM high-pass filter kernels.

    Each kernel is a 5x5 grid of numbers. When convolved with an image,
    it extracts specific types of noise/edge patterns.

    Returns:
        numpy array of shape [9, 5, 5]
    """

    # Kernel 1: 1st-order horizontal edge detector
    # Detects changes from left to right
    #  0  0  0  0  0
    #  0  0  0  0  0
    #  0  1 -1  0  0   ← subtracts right pixel from left pixel
    #  0  0  0  0  0
    #  0  0  0  0  0
    k1 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)

    # Kernel 2: 1st-order vertical edge detector
    # Same as k1 but detects top-to-bottom changes
    k2 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, -1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)

    # Kernel 3: 2nd-order horizontal detector (like Laplacian)
    # Sensitive to curvature (concave vs convex patterns)
    k3 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 1, -2, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)

    # Kernel 4: 2nd-order vertical detector
    k4 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, -2, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)

    # Kernel 5: 3rd-order SQUARE filter (5x5 edge-aware)
    # More complex pattern that catches subtle manipulation artifacts
    k5 = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 2, -1, 0],
        [0, 2, -4, 2, 0],
        [0, -1, 2, -1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32) / 4.0  # Normalize so values don't explode

    # Kernel 6: Cross-shaped 3rd-order filter
    k6 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, -4, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32) / 4.0

    # Kernel 7: Full 5x5 high-pass filter
    # Catches noise patterns at a larger scale
    k7 = np.array([
        [-1, 2, -2, 2, -1],
        [2, -6, 8, -6, 2],
        [-2, 8, -12, 8, -2],
        [2, -6, 8, -6, 2],
        [-1, 2, -2, 2, -1],
    ], dtype=np.float32) / 12.0

    # Kernel 8: Diagonal edge detector (top-left to bottom-right)
    k8 = np.array([
        [0, 0, 0, 0, 0],
        [0, -1, 0, 0, 0],
        [0, 0, 2, 0, 0],
        [0, 0, 0, -1, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)

    # Kernel 9: Anti-diagonal edge detector (top-right to bottom-left)
    k9 = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, -1, 0],
        [0, 0, 2, 0, 0],
        [0, -1, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.float32)

    # Stack all 9 kernels: shape [9, 5, 5]
    return np.stack([k1, k2, k3, k4, k5, k6, k7, k8, k9])


class SRMFilterLayer(nn.Module):
    """
    Apply SRM noise extraction filters to RGB images.

    This layer is NOT trained — the filter weights are FIXED mathematical
    formulas from steganalysis research. We just use PyTorch's convolution
    to apply them efficiently on GPU.

    Parameters:
        use_extended (bool): Use all 9 filters (True) or just first 3 (False)
        learnable (bool): If True, allow the filters to be fine-tuned during training.
                          Usually False because the fixed filters work well.
    """

    def __init__(self, use_extended=True, learnable=False):
        super().__init__()

        # Load the fixed filter kernels
        kernels = get_srm_kernels()  # [9, 5, 5]
        self.n_filters = 9 if use_extended else 3

        if not use_extended:
            kernels = kernels[:3]  # Use only first 3 filters

        # Reshape for PyTorch's Conv2d:
        # Conv2d expects weights of shape [out_channels, in_channels, H, W]
        # We apply EACH filter to EACH color channel independently,
        # then average across RGB to get one noise map per filter
        # So: [n_filters, 1, 5, 5] and we use groups=1, input channels=1

        # Actually, we apply each filter across all 3 RGB channels
        # and average them, which is equivalent to using a [n_filters, 3, 5, 5] kernel
        # where the weights for R, G, B are all the same (divided by 3)
        kernel_tensor = torch.tensor(kernels, dtype=torch.float32)
        # [n_filters, 5, 5] → [n_filters, 1, 5, 5] → repeat for 3 channels
        kernel_tensor = kernel_tensor.unsqueeze(1).repeat(1, 3, 1, 1)
        # Divide by 3 so the average across RGB is correct
        kernel_tensor = kernel_tensor / 3.0

        if learnable:
            # Make the filter weights trainable (unusual, but possible)
            self.filters = nn.Parameter(kernel_tensor)
        else:
            # Register as a "buffer" — saved with the model but NOT trained
            # This is the standard approach: fixed forensic filters
            self.register_buffer("filters", kernel_tensor)

    def forward(self, x):
        """
        Apply SRM filters to extract noise residuals.

        Args:
            x: [B, 3, H, W] — RGB images (float, any range)

        Returns:
            residuals: [B, n_filters, H, W] — noise residual maps
                Each of the 9 (or 3) maps highlights different noise patterns
        """
        # F.conv2d = 2D convolution (sliding window)
        # padding=2 keeps the output the same size as input (5x5 kernel needs padding=2)
        return F.conv2d(x, self.filters, padding=2)


class SRMEnhancedBackbone(nn.Module):
    """
    Combine a CNN backbone with SRM features in a two-stream architecture.

    This is a general-purpose wrapper that can add SRM to ANY backbone.
    Branch A uses this pattern internally.

    The idea: backbone sees what the image LOOKS like,
    SRM sees what the noise PATTERN is like.
    Together they're more powerful than either alone.
    """

    def __init__(self, backbone, backbone_dim, srm_dim=256, learnable_srm=False):
        super().__init__()
        self.backbone = backbone
        self.srm = SRMFilterLayer(use_extended=True, learnable=learnable_srm)

        # Small CNN to process SRM noise maps into a feature vector
        n_filters = self.srm.n_filters
        self.srm_encoder = nn.Sequential(
            nn.Conv2d(n_filters, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.combined_dim = backbone_dim + 128  # backbone features + SRM features

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] — input images

        Returns:
            combined: [B, combined_dim] — backbone + SRM features concatenated
        """
        # Stream 1: Standard visual features
        backbone_features = self.backbone(x)  # [B, backbone_dim]

        # Stream 2: Noise residual features
        noise = self.srm(x)                              # [B, 9, H, W]
        srm_features = self.srm_encoder(noise).flatten(1)  # [B, 128]

        # Concatenate both streams
        return torch.cat([backbone_features, srm_features], dim=1)
