"""
Branch A: Face-Forensic Branch with SRM Residual Filters
========================================================

PURPOSE:
    Analyze FACE CROPS to detect face-swaps and reenactments.
    This branch receives cropped face images (224x224) and looks for
    manipulation artifacts in both the visual appearance AND the invisible
    noise pattern of the face.

HOW IT WORKS:
    It has TWO streams (like two detectives looking at the same face):

    Stream 1 — RGB Analysis (ConvNeXt-Tiny + CBAM):
        - A modern CNN pretrained on ImageNet (better than EfficientNet)
        - CBAM attention focuses on face boundary artifacts
        - Produces a 768-dimensional "fingerprint" of the face
        - Think: "Does this face LOOK real?"

    Stream 2 — SRM Noise Analysis:
        - Applies special math filters that REMOVE the visible content
        - Leaves behind only the noise/texture pattern
        - Real cameras leave consistent noise; deepfakes have weird noise
        - Produces a 256-dimensional noise fingerprint
        - Think: "Does the invisible noise pattern FEEL real?"

    Combined output: 1024-dimensional embedding (768 + 256)

INPUTS:
    face_crops: [batch_size, 3, 224, 224]

OUTPUTS:
    embeddings: [batch_size, 1024] — the face's "fingerprint" for fusion
    logits: [batch_size, 1] — this branch's own real/fake prediction
"""

import torch
import torch.nn as nn
import timm

# Our custom SRM noise filter module (see models/srm_filters.py)
from .srm_filters import SRMFilterLayer
from .cbam import CBAM  # Attention module for artifact focus


class FaceForensicsBranch(nn.Module):
    """
    Two-stream face analysis for deepfake detection.

    Parameters:
        pretrained (bool): If True, load ConvNeXt weights trained on ImageNet.
                           Always True for training (transfer learning).
                           Set to False only when loading our own saved weights.
        use_srm (bool): If True, enable the SRM noise analysis stream.
                        Adds ~100K extra parameters but catches subtle fakes.
    """

    def __init__(self, pretrained=True, use_srm=True):
        super().__init__()

        self.use_srm = use_srm

        # ─── Stream 1: ConvNeXt-Tiny backbone ───
        # ConvNeXt-Tiny: 28M params, 768-dim output, designed 2022
        # Better than EfficientNet for spatial artifact detection
        # Named 'backbone' so train_fast.py freezing code works unchanged
        self.backbone = timm.create_model("convnext_tiny", pretrained=pretrained)

        # Global Average Pooling: [B, 768, 7, 7] → [B, 768, 1, 1] → flatten → [B, 768]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        rgb_dim = 768  # ConvNeXt-Tiny output channels

        # CBAM attention — helps focus on face boundary artifacts
        self.cbam = CBAM(in_planes=rgb_dim, ratio=16)

        # ─── Stream 2: SRM noise residual analysis ───
        if use_srm:
            # SRM filters: 9 fixed high-pass filters that extract noise
            self.srm_filters = SRMFilterLayer(use_extended=True, learnable=False)
            n_srm = self.srm_filters.n_filters  # = 9

            # Small CNN to process the 9 noise maps into a 256-dim vector
            self.srm_cnn = nn.Sequential(
                nn.Conv2d(n_srm, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),

                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),

                nn.AdaptiveAvgPool2d((1, 1)),
            )
            srm_dim = 256
            self.embed_dim = rgb_dim + srm_dim  # 768 + 256 = 1024
        else:
            self.embed_dim = rgb_dim  # 768

        # ─── Per-branch classifier ───
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(self.embed_dim, 1),
        )

    def forward(self, face_crops):
        """
        Args:
            face_crops: [B, 3, 224, 224] — batch of face images

        Returns:
            embeddings: [B, embed_dim] — face fingerprints for fusion
            logits: [B, 1] — this branch's real/fake prediction
        """
        # ─── RGB stream: ConvNeXt with CBAM attention ───
        # forward_features returns spatial feature maps: [B, 768, 7, 7]
        rgb_feats = self.backbone.forward_features(face_crops)  # [B, 768, 7, 7]
        rgb_feats = self.cbam(rgb_feats)  # Attention-weighted features
        rgb_embed = self.pool(rgb_feats).flatten(1)  # [B, 768]

        if self.use_srm:
            # ─── SRM stream ───
            srm_residuals = self.srm_filters(face_crops)        # [B, 9, 224, 224]
            srm_embed = self.srm_cnn(srm_residuals).flatten(1)  # [B, 256]

            # Concatenate RGB features + noise features
            embeddings = torch.cat([rgb_embed, srm_embed], dim=1)  # [B, 1024]
        else:
            embeddings = rgb_embed  # [B, 768]

        logits = self.classifier_head(embeddings)  # [B, 1]

        return embeddings, logits

    def get_embeddings(self, face_crops):
        """Return only embeddings (used during E2E training)."""
        rgb_feats = self.backbone.forward_features(face_crops)
        rgb_feats = self.cbam(rgb_feats)
        rgb_embed = self.pool(rgb_feats).flatten(1)

        if self.use_srm:
            srm_residuals = self.srm_filters(face_crops)
            srm_embed = self.srm_cnn(srm_residuals).flatten(1)
            return torch.cat([rgb_embed, srm_embed], dim=1)
        return rgb_embed