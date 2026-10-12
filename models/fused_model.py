"""
Fused AI Video Detector v6 — ConvNeXt + SRM + Spectral + GRU Temporal
=====================================================================

PURPOSE:
    This is the MAIN MODEL that combines all 3 branches into one final prediction.

HOW IT WORKS:
    1. Branch A analyzes face crops (ConvNeXt + CBAM + SRM) → "face evidence" (1024-dim)
    2. Branch B analyzes full frames (ConvNeXt + CBAM + FFT/DCT) → "frame evidence" (1152-dim)
    3. Branch C analyzes temporal patterns (BiGRU) → "temporal evidence" (512-dim)
    4. Gated Fusion dynamically weights each branch's contribution
    5. Final MLP outputs the tampering risk score

ARCHITECTURE SUMMARY:
    Branch A (ConvNeXt + CBAM + SRM):  28M + 0.1M  →  1024-dim
    Branch B (ConvNeXt + CBAM + Spectral): 28M + 0.5M → 1152-dim
    Branch C (Bidirectional GRU):       ~3M        →   512-dim
    Gated Fusion + MLP:                 ~1M        →     1 logit
    ─────────────────────────────────────────────────────
    Total:                             ~60M params

SKIP_TEMPORAL:
    If Branch C fails to reach 60% AUC during training, skip_temporal is set
    to True and the temporal branch outputs zeros. The gated fusion learns
    to ignore it, effectively becoming a 2-branch model.
    This flag is persisted in the state_dict as a registered buffer.
"""

import torch
import torch.nn as nn

# Import all 3 branches
from .branch_a import FaceForensicsBranch  # Face + SRM noise analysis
from .branch_b import FullFrameBranch      # Full frame + FFT/DCT spectral
from .branch_c import TemporalSequenceBranch  # Temporal GRU


class GatedFusion(nn.Module):
    """
    Learned gating mechanism that dynamically weights branch contributions.

    HOW IT WORKS:
        Input: 3 branch embeddings (1024 + 1152 + 512 = 2688 total dims)

        1. Concatenate all embeddings
        2. Pass through a small neural network
        3. Output 3 gate values via Sigmoid (each between 0 and 1)
        4. Multiply each branch embedding by its gate value

    Parameters:
        branch_dims: list of ints, e.g., [1024, 1152, 512]
    """

    def __init__(self, branch_dims):
        super().__init__()
        total_dim = sum(branch_dims)  # 2688
        n_branches = len(branch_dims)  # 3

        # Gate network: takes ALL branch embeddings, outputs per-branch weights
        self.gate_net = nn.Sequential(
            nn.Linear(total_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(256, n_branches),
            nn.Sigmoid(),  # Independent gates [0, 1]
        )

        self.branch_dims = branch_dims
        self.n_branches = n_branches

    def forward(self, branch_embeddings):
        """
        Apply gated fusion to branch embeddings.

        Args:
            branch_embeddings: list of 3 tensors
                [face_embed: [B, 1024], frame_embed: [B, 1152], temporal_embed: [B, 512]]

        Returns:
            gated_concat: [B, 2688] — weighted concatenation of all branches
            gate_values: [B, 3] — the gate weights (for interpretability)
        """
        concat = torch.cat(branch_embeddings, dim=1)  # [B, 2688]
        gates = self.gate_net(concat)  # [B, 3]

        gated_parts = []
        for i, embed in enumerate(branch_embeddings):
            gate_i = gates[:, i:i+1]  # [B, 1]
            gated_parts.append(embed * gate_i)

        gated_concat = torch.cat(gated_parts, dim=1)  # [B, 2688]
        return gated_concat, gates


class FusedAIVideoDetector(nn.Module):
    """
    The complete deepfake detection model.

    This is the top-level module that:
    1. Creates all 3 branches
    2. Runs video through them
    3. Fuses results with gated fusion
    4. Outputs final prediction

    Parameters:
        pretrained (bool): Use ImageNet-pretrained backbones
        use_srm (bool): Enable SRM noise filters in Branch A
        use_spectral (bool): Enable FFT/DCT spectral analysis in Branch B
    """

    def __init__(self, pretrained=True, use_srm=True, use_spectral=True):
        super().__init__()

        self.use_srm = use_srm
        self.use_spectral = use_spectral

        # Registered buffer — persisted in state_dict, auto-saved/loaded
        # 0 = use temporal, 1 = skip temporal (fallback to 2-branch)
        self.register_buffer('_skip_temporal_flag', torch.tensor(0, dtype=torch.long))

        # ─── Create the 3 branches ───

        # Branch A: Face forensics (ConvNeXt + CBAM + optional SRM)
        self.face_branch = FaceForensicsBranch(
            pretrained=pretrained, use_srm=use_srm
        )

        # Branch B: Full-frame analysis (ConvNeXt + CBAM + optional FFT/DCT)
        self.frame_branch = FullFrameBranch(
            pretrained=pretrained, use_spectral=use_spectral
        )

        # Branch C: Temporal analysis (Bidirectional GRU)
        combined_per_frame = self.face_branch.embed_dim + self.frame_branch.embed_dim
        self.temporal_branch = TemporalSequenceBranch(
            feature_dim=combined_per_frame  # 1024 + 1152 = 2176
        )

        # ─── Set up gated fusion ───

        branch_dims = [
            self.face_branch.embed_dim,      # 1024 (with SRM) or 768
            self.frame_branch.embed_dim,     # 1152 (with spectral) or 768
            self.temporal_branch.hidden_dim,  # 512
        ]
        total_dim = sum(branch_dims)  # 2688

        self.gated_fusion = GatedFusion(branch_dims)

        # ─── Final classifier MLP ───
        self.fusion_mlp = nn.Sequential(
            nn.Linear(total_dim, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 1),
        )

    @property
    def skip_temporal(self):
        """Whether to skip the temporal branch (fallback to 2-branch fusion)."""
        return self._skip_temporal_flag.item() == 1

    @skip_temporal.setter
    def skip_temporal(self, value):
        """Set skip_temporal flag. Persisted in state_dict."""
        self._skip_temporal_flag.fill_(1 if value else 0)

    def forward(self, video_frames, face_crops):
        """
        Full forward pass through all 3 branches + gated fusion.

        Args:
            video_frames: [B, T, 3, H, W] — batch of video frame sequences
            face_crops: [B, T, 3, 224, 224] — corresponding face crops

        Returns:
            final_logit:    [B, 1] — the fused tampering prediction
            face_logit:     [B, 1] — Branch A's individual prediction
            frame_logit:    [B, 1] — Branch B's individual prediction
            temporal_logit: [B, 1] — Branch C's individual prediction
        """
        b, t, c, h, w = video_frames.shape

        # STEP 1: Process ALL frames through Branch A (face)
        flat_faces = face_crops.view(b * t, c, 224, 224)
        face_embeds, face_logit_per_frame = self.face_branch(flat_faces)

        # STEP 2: Process ALL frames through Branch B (full-frame)
        flat_frames = video_frames.view(b * t, c, h, w)
        frame_embeds, frame_logit_per_frame = self.frame_branch(flat_frames)

        # STEP 3: Temporal analysis
        if self.skip_temporal:
            # Temporal branch disabled — output zeros
            temporal_embed = torch.zeros(b, self.temporal_branch.hidden_dim,
                                         device=video_frames.device)
            temporal_logit = torch.zeros(b, 1, device=video_frames.device)
        else:
            # Combine per-frame features for Branch C
            combined_per_frame = torch.cat([face_embeds, frame_embeds], dim=1)
            sequence_feats = combined_per_frame.view(b, t, -1)  # [B, T, 2176]
            temporal_embed, temporal_logit = self.temporal_branch(sequence_feats)

        # STEP 4: Aggregate per-branch scores
        face_logit = face_logit_per_frame.view(b, t, 1).mean(dim=1)
        frame_logit = frame_logit_per_frame.view(b, t, 1).mean(dim=1)

        # STEP 5: Gated Fusion
        face_embed_avg = face_embeds.view(b, t, -1).mean(dim=1)    # [B, 1024]
        frame_embed_avg = frame_embeds.view(b, t, -1).mean(dim=1)  # [B, 1152]

        gated_features, gate_values = self.gated_fusion([
            face_embed_avg, frame_embed_avg, temporal_embed
        ])

        # STEP 6: Final classification
        final_logit = self.fusion_mlp(gated_features)  # [B, 1]

        return final_logit, face_logit, frame_logit, temporal_logit

    def forward_with_gates(self, video_frames, face_crops):
        """
        Same as forward() but ALSO returns gate values for interpretability.

        Returns:
            final_logit, face_logit, frame_logit, temporal_logit, gate_values
            gate_values: [B, 3] — weights for [face, frame, temporal] branches
        """
        b, t, c, h, w = video_frames.shape

        flat_faces = face_crops.view(b * t, c, 224, 224)
        face_embeds, face_logit_per_frame = self.face_branch(flat_faces)

        flat_frames = video_frames.view(b * t, c, h, w)
        frame_embeds, frame_logit_per_frame = self.frame_branch(flat_frames)

        if self.skip_temporal:
            temporal_embed = torch.zeros(b, self.temporal_branch.hidden_dim,
                                         device=video_frames.device)
            temporal_logit = torch.zeros(b, 1, device=video_frames.device)
        else:
            combined_per_frame = torch.cat([face_embeds, frame_embeds], dim=1)
            sequence_feats = combined_per_frame.view(b, t, -1)
            temporal_embed, temporal_logit = self.temporal_branch(sequence_feats)

        face_logit = face_logit_per_frame.view(b, t, 1).mean(dim=1)
        frame_logit = frame_logit_per_frame.view(b, t, 1).mean(dim=1)

        face_embed_avg = face_embeds.view(b, t, -1).mean(dim=1)
        frame_embed_avg = frame_embeds.view(b, t, -1).mean(dim=1)

        gated_features, gate_values = self.gated_fusion([
            face_embed_avg, frame_embed_avg, temporal_embed
        ])

        final_logit = self.fusion_mlp(gated_features)

        return final_logit, face_logit, frame_logit, temporal_logit, gate_values


class FusedDetectorInference:
    """
    Convenience wrapper for running detection on a single video file.

    Usage:
        detector = FusedDetectorInference("weights/best_model.pth")
        result = detector.predict("suspect_video.mp4")
        print(result)
    """

    def __init__(self, weights_path, calibration_path=None, device=None,
                 n_frames=16, use_srm=True, use_spectral=True):
        import os

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.n_frames = n_frames
        self.calibration_path = calibration_path

        self.model = FusedAIVideoDetector(
            pretrained=False, use_srm=use_srm, use_spectral=use_spectral
        )
        state = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        from utils.face_detect import FaceDetector
        self.face_detector = FaceDetector(device=self.device)

    @torch.no_grad()
    def predict(self, video_path):
        """Analyze a video file and return detection results."""
        from utils.video_loader import load_and_sample_frames_raw, normalize_frames
        from utils.calibrate import calibrate_scores

        raw_frames, fps, total = load_and_sample_frames_raw(
            video_path, n_frames=self.n_frames
        )

        bboxes = self.face_detector.detect_batch(raw_frames)
        face_crops, valid_mask = self.face_detector.crop_faces(raw_frames, bboxes)

        normalized_frames = normalize_frames(raw_frames)
        normalized_faces = normalize_frames(face_crops)

        video_batch = normalized_frames.unsqueeze(0).to(self.device)
        face_batch = normalized_faces.unsqueeze(0).to(self.device)

        final_logit, face_logit, frame_logit, temporal_logit, gate_values = \
            self.model.forward_with_gates(video_batch, face_batch)

        if self.calibration_path:
            final_score = calibrate_scores(final_logit, self.calibration_path).item()
        else:
            final_score = torch.sigmoid(final_logit).item()

        face_score = torch.sigmoid(face_logit).item()
        frame_score = torch.sigmoid(frame_logit).item()
        temporal_score = torch.sigmoid(temporal_logit).item()

        gates = gate_values.squeeze().cpu().tolist()

        if final_score > 0.70:
            verdict = "Likely manipulated"
        elif final_score > 0.40:
            verdict = "Suspicious -- possible manipulation"
        else:
            verdict = "Likely authentic"

        return {
            "final_score": final_score,
            "face_score": face_score,
            "frame_score": frame_score,
            "temporal_score": temporal_score,
            "gate_weights": {
                "face_gate": gates[0],
                "frame_gate": gates[1],
                "temporal_gate": gates[2],
            },
            "verdict": verdict,
            "faces_detected": int(valid_mask.sum().item()),
            "total_frames_sampled": self.n_frames,
            "source_fps": fps,
            "source_total_frames": total,
        }