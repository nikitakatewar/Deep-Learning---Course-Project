"""
Branch B: Full-Frame Branch with FFT/DCT Spectral Analysis
==========================================================

PURPOSE:
    Analyze the ENTIRE video frame (not just the face) to detect
    fully AI-generated videos and global manipulation artifacts.

HOW IT WORKS:
    Three streams combined:

    Stream 1 — RGB Analysis (ConvNeXt-Tiny + CBAM):
        - A modern CNN that analyzes the full frame
        - CBAM attention focuses on artifact-rich regions
        - Detects inconsistencies in lighting, shadows, reflections
        - Produces 768-dimensional spatial embedding
        - Think: "Does this frame LOOK real across all regions?"

    Stream 2 — FFT Spectrum Analysis:
        - Converts the image from "pixel space" to "frequency space"
        - AI generators leave frequency fingerprints
        - Produces 256-dimensional frequency embedding

    Stream 3 — DCT Block Analysis:
        - Analyzes 8x8 pixel blocks (same blocks JPEG uses)
        - Produces 128-dimensional compression embedding

    Combined output: 1152-dimensional embedding (768 + 256 + 128)

INPUTS:
    full_frames: [batch_size, 3, 224, 224] — full video frames (resized)

OUTPUTS:
    embeddings: [batch_size, 1152] — frame fingerprints for fusion
    logits: [batch_size, 1] — this branch's own real/fake prediction
"""

import torch
import torch.nn as nn
import timm

# Our custom spectral analysis modules (see models/spectral.py)
from .spectral import SpectralFeatureExtractor, DCTFeatureExtractor
from .cbam import CBAM  # Attention module for artifact focus


class FullFrameBranch(nn.Module):
    """
    Three-stream full-frame analysis for deepfake detection.

    Parameters:
        pretrained (bool): Load ConvNeXt weights from ImageNet.
        use_spectral (bool): Enable FFT + DCT spectral analysis streams.
    """

    def __init__(self, pretrained=True, use_spectral=True):
        super().__init__()
        self.use_spectral = use_spectral

        # ─── Stream 1: ConvNeXt-Tiny RGB backbone ───
        # Named 'backbone' so train_fast.py freezing code works unchanged
        self.backbone = timm.create_model("convnext_tiny", pretrained=pretrained)

        # Global Average Pooling: [B, 768, 7, 7] → [B, 768]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        rgb_dim = 768  # ConvNeXt-Tiny output channels

        # CBAM attention — helps focus on artifact-rich regions
        self.cbam = CBAM(in_planes=rgb_dim, ratio=16)

        # ─── Stream 2+3: FFT + DCT spectral analysis ───
        if use_spectral:
            self.fft_extractor = SpectralFeatureExtractor(embed_dim=256)
            self.dct_extractor = DCTFeatureExtractor(embed_dim=128)

            spectral_dim = 256 + 128  # 384 total from frequency analysis
            self.embed_dim = rgb_dim + spectral_dim  # 768 + 384 = 1152
        else:
            self.embed_dim = rgb_dim  # Just 768

        # ─── Per-branch classifier ───
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, full_frames):
        """
        Process full video frames through all 3 streams.

        Args:
            full_frames: [B, 3, 224, 224] — batch of full video frames

        Returns:
            embeddings: [B, embed_dim] — frame fingerprints
            logits: [B, 1] — real/fake prediction
        """
        # ─── Spatial stream: ConvNeXt with CBAM attention ───
        rgb_feats = self.backbone.forward_features(full_frames)  # [B, 768, 7, 7]
        rgb_feats = self.cbam(rgb_feats)  # Attention-weighted features
        rgb_embed = self.pool(rgb_feats).flatten(1)  # [B, 768]

        if self.use_spectral:
            # ─── Frequency streams: FFT + DCT ───
            fft_embed = self.fft_extractor(full_frames)  # [B, 256]
            dct_embed = self.dct_extractor(full_frames)  # [B, 128]

            embeddings = torch.cat([rgb_embed, fft_embed, dct_embed], dim=1)  # [B, 1152]
        else:
            embeddings = rgb_embed  # [B, 768]

        logits = self.classifier_head(embeddings)
        return embeddings, logits

    def get_embeddings(self, full_frames):
        """Return only embeddings (for temporal branch training)."""
        rgb_feats = self.backbone.forward_features(full_frames)
        rgb_feats = self.cbam(rgb_feats)
        rgb_embed = self.pool(rgb_feats).flatten(1)

        if self.use_spectral:
            fft_embed = self.fft_extractor(full_frames)
            dct_embed = self.dct_extractor(full_frames)
            return torch.cat([rgb_embed, fft_embed, dct_embed], dim=1)
        return rgb_embed