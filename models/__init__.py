"""
models/ — AI Video Deepfake Detection Model Package
====================================================

Architecture: 3-Branch Gated Fusion Network (v6)
    Branch A (FaceForensicsBranch):  ConvNeXt-Tiny + SRM noise filters + CBAM attention
    Branch B (FullFrameBranch):      ConvNeXt-Tiny + FFT/DCT spectral analysis + CBAM attention
    Branch C (TemporalSequenceBranch): Bidirectional GRU + Temporal Attention Pooling

    Fusion: Gated Fusion Network learns per-sample branch weights → MLP classifier

Performance (v6, Sep 2026):
    - Val AUC:  0.9549
    - Eval Acc: 87.8% overall, 100% on HeyGen AI avatars
    - Params:   ~63.5M total (61.4M branches + 2.1M fusion)

Key design decisions:
    - ConvNeXt-Tiny backbone (pretrained ImageNet) instead of ViT-Small (overfits on 3K dataset)
    - BiGRU instead of Transformer for temporal (0.95 AUC vs 0.46 AUC on small data)
    - SRM high-pass filters catch pixel-level noise patterns invisible to RGB analysis
    - FFT/DCT spectral extractors detect frequency-domain artifacts from AI generation
    - CBAM (Channel + Spatial) attention helps focus on manipulated regions
    - Gated fusion lets the model learn which branch to trust per-video
"""

# ─── Model exports for the deepfake detection pipeline ───
from .branch_a import FaceForensicsBranch
from .branch_b import FullFrameBranch
from .branch_c import TemporalSequenceBranch
from .srm_filters import SRMFilterLayer, SRMEnhancedBackbone
from .spectral import SpectralFeatureExtractor, DCTFeatureExtractor, DualSpectralModule
from .fused_model import FusedAIVideoDetector, FusedDetectorInference, GatedFusion
