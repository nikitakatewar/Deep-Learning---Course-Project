"""
Branch C: Temporal Sequence Branch (Bidirectional GRU)
=====================================================

PURPOSE:
    Detect temporal inconsistencies — things that look fine in a single
    frame but become obvious when you watch the video play.

    Examples of what this catches:
    - Flickering (face appears/disappears between frames)
    - Unnatural motion (head moves too smoothly or too jerkily)
    - Physics violations (shadows don't match between frames)
    - Real→fake transitions (video becomes AI-generated partway through)

WHY GRU INSTEAD OF TRANSFORMER:
    The previous version used a Transformer Encoder here and it failed
    catastrophically (AUC 0.49 — worse than random guessing).

    Why Transformers fail on small datasets:
    - Transformers have ZERO inductive bias for sequences
    - They need 100K+ samples to learn temporal patterns from scratch
    - On 3,000 videos, they just memorize the training set

    Why GRU works:
    - GRU has BUILT-IN recurrence — it inherently understands that
      frame 1 comes before frame 2 (no positional encoding needed)
    - Bidirectional means it looks both forward AND backward in time
    - ~10x fewer parameters than the Transformer version
    - Proven effective on small video classification datasets

    ATTENTION POOLING:
    - Instead of averaging all frame representations equally,
      we learn WHICH frames are most diagnostic
    - Some frames show strong artifacts, others are ambiguous
    - Attention pooling upweights the informative frames

INPUTS:
    frame_features: [batch_size, num_frames, 2176]
        - Per-frame embeddings from Branch A (1024) + Branch B (1152)

OUTPUTS:
    temporal_embed: [batch_size, 512] — temporal summary for fusion
    logits: [batch_size, 1] — temporal branch's real/fake prediction
"""

import torch
import torch.nn as nn


class TemporalSequenceBranch(nn.Module):
    """
    Bidirectional GRU for analyzing frame sequences.

    Parameters:
        feature_dim (int): Input dimension per frame (2176 = Branch A + Branch B).
        hidden_dim (int): Output dimension (512). GRU hidden is half this (256)
                          because bidirectional doubles it back to 512.
        num_layers (int): Number of stacked GRU layers (2).
        dropout (float): Dropout between GRU layers and in projection.
    """

    def __init__(self, feature_dim=2176, hidden_dim=512, num_layers=2, dropout=0.3):
        super().__init__()

        # ─── Step 1: Project high-dimensional features to hidden_dim ───
        # 2176 → 512 (reduces computation for GRU)
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ─── Step 2: Bidirectional GRU ───
        # hidden_size = 256, but bidirectional doubles output to 512
        # Each direction processes the sequence independently:
        #   Forward:  frame 1 → frame 2 → ... → frame 16
        #   Backward: frame 16 → frame 15 → ... → frame 1
        # Their outputs are concatenated: [forward_out; backward_out] = 512
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,  # 256, bidirectional → 512
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.layer_norm = nn.LayerNorm(hidden_dim)

        # ─── Step 3: Attention Pooling ───
        # Instead of mean pooling (treating all frames equally),
        # learn which frames are most informative for detection.
        # A small network scores each frame, then softmax → weighted sum.
        self.attn_pool = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

        self.hidden_dim = hidden_dim

        # ─── Per-branch classifier ───
        self.classifier_head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, frame_features):
        """
        Process a sequence of frame features through the bidirectional GRU.

        Args:
            frame_features: [B, T, 2176] — per-frame embeddings from A+B
                B = batch size (number of videos)
                T = number of frames (e.g., 16)
                2176 = feature dimension (1024 from face + 1152 from frame)

        Returns:
            temporal_embed: [B, 512] — temporal summary of the whole video
            logits: [B, 1] — temporal branch's real/fake prediction
        """
        b, t, _ = frame_features.shape

        # Project 2176-dim down to 512-dim for each frame
        x = self.projection(frame_features)  # [B, T, 512]

        # Run through bidirectional GRU
        # Each frame's output now contains information from ALL frames
        # (forward direction carries past context, backward carries future context)
        x, _ = self.gru(x)  # [B, T, 512]

        # Normalize
        x = self.layer_norm(x)  # [B, T, 512]

        # ─── Attention Pooling ───
        # Score each frame's importance
        attn_scores = self.attn_pool(x)  # [B, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [B, T, 1]

        # Weighted sum: frames with higher attention contribute more
        temporal_embed = (x * attn_weights).sum(dim=1)  # [B, 512]

        logits = self.classifier_head(temporal_embed)  # [B, 1]

        return temporal_embed, logits