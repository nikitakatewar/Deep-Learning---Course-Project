"""
Spectral Feature Extractors (FFT + DCT)
========================================

PURPOSE:
    Extract FREQUENCY DOMAIN features from images to detect AI-generated content.

WHAT IS THE FREQUENCY DOMAIN?
    Every image can be thought of in two ways:
    
    1. SPATIAL domain (what we normally see):
       - "This pixel is red, that pixel is blue"
       - Like describing a song by listing every sound sample
    
    2. FREQUENCY domain (mathematical decomposition):
       - "This image has strong horizontal stripes and weak diagonal patterns"
       - Like describing a song by listing its notes: "lots of bass, some treble"
    
    The transform between these is called FFT (Fast Fourier Transform).
    
    WHY DOES THIS HELP?
    - Real cameras capture a natural distribution of frequencies
    - AI generators (GANs, diffusion models, Sora, etc.) produce UNNATURAL
      frequency patterns — weird peaks, missing frequencies, periodic artifacts
    - These are INVISIBLE to human eyes but show up clearly in the frequency domain

ANALOGY:
    Imagine a master art forger who can copy a painting perfectly.
    The forgery looks identical to human eyes.
    But if you examine it under UV light, the paint fluoresces differently
    because the chemical composition is different.
    FFT/DCT is our "UV light" — it reveals invisible differences.

TWO MODULES IN THIS FILE:
    1. SpectralFeatureExtractor (FFT-based) — global frequency analysis
    2. DCTFeatureExtractor (DCT-based) — local block-level analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralFeatureExtractor(nn.Module):
    """
    FFT-based spectral analysis for detecting AI-generated content.

    HOW IT WORKS (step by step):
        1. Convert RGB image to grayscale (average the 3 color channels)
        2. Apply 2D FFT (Fast Fourier Transform) to convert to frequency domain
        3. Compute the "magnitude spectrum" — how strong each frequency is
        4. Shift so low frequencies are in the center (standard convention)
        5. Take log(1 + magnitude) to compress the huge dynamic range
        6. Feed the log-magnitude through a small CNN to extract features
        7. Output: 256-dimensional frequency embedding

    WHAT DOES THE FFT OUTPUT LOOK LIKE?
        It's a 2D grid the same size as the input image.
        - CENTER = low frequencies (smooth gradients, average brightness)
        - EDGES = high frequencies (sharp edges, fine textures, noise)
        
        Real images: smooth, natural falloff from center to edges
        Fake images: weird spikes, rings, or periodic patterns

    Parameters:
        embed_dim (int): Output embedding dimension (default 256)
    """

    def __init__(self, embed_dim=256):
        super().__init__()

        # A small CNN that learns to read the FFT magnitude spectrum
        # Input: 1 channel (grayscale magnitude spectrum)
        # Output: embed_dim features
        self.spectral_cnn = nn.Sequential(
            # First conv layer: 1 → 32 channels
            # kernel_size=5: looks at 5x5 patches of the spectrum
            # stride=2: skips every other position (halves resolution)
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),  # [1, H, W] → [32, H/2, W/2]
            nn.BatchNorm2d(32),    # Normalize for stable training
            nn.ReLU(inplace=True), # Activation function

            # Second conv layer: 32 → 64 channels, halve resolution again
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # [32, H/2, W/2] → [64, H/4, W/4]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Third conv layer: 64 → 128 channels, halve again
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # [64, H/4, W/4] → [128, H/8, W/8]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Reduce all spatial dimensions to 1x1
            nn.AdaptiveAvgPool2d((1, 1)),  # [128, H/8, W/8] → [128, 1, 1]
        )

        # Final linear layer: 128 → embed_dim
        self.fc = nn.Linear(128, embed_dim)

    def forward(self, x):
        """
        Extract FFT-based spectral features from RGB images.

        Args:
            x: [B, 3, H, W] — batch of RGB images

        Returns:
            spectral_embed: [B, embed_dim] — frequency domain features
        """
        # Step 1: RGB → Grayscale (average across color channels)
        # We analyze frequency content of the overall brightness, not per-channel
        gray = x.mean(dim=1, keepdim=True)  # [B, 3, H, W] → [B, 1, H, W]

        # Step 2: Apply 2D FFT (Fast Fourier Transform)
        # torch.fft.fft2 converts spatial data → frequency data
        # The output is COMPLEX (has real and imaginary parts)
        fft = torch.fft.fft2(gray)  # [B, 1, H, W] complex

        # Step 3: Compute magnitude (how strong each frequency is)
        # |z| = sqrt(real² + imaginary²)
        magnitude = torch.abs(fft)  # [B, 1, H, W] real

        # Step 4: Shift so DC component (average brightness) is in the center
        # By default, FFT puts low frequencies at corners — this rearranges them
        magnitude = torch.fft.fftshift(magnitude, dim=(-2, -1))

        # Step 5: Log-compress to handle huge dynamic range
        # Without log, the DC component might be 10000x larger than edges
        # log(1 + x) maps [0, ∞) → [0, ∞) but compresses large values
        log_magnitude = torch.log1p(magnitude)  # [B, 1, H, W]

        # Step 6: Process through CNN to extract meaningful features
        features = self.spectral_cnn(log_magnitude)  # [B, 128, 1, 1]
        features = features.flatten(1)                # [B, 128]

        # Step 7: Project to final embedding dimension
        spectral_embed = self.fc(features)  # [B, 256]

        return spectral_embed


class DCTFeatureExtractor(nn.Module):
    """
    DCT (Discrete Cosine Transform) block-level analysis.

    HOW IT WORKS:
        1. Convert image to grayscale
        2. Split into 8×8 pixel blocks (same blocks used in JPEG compression!)
        3. Apply DCT to each block separately
        4. Analyze the DCT coefficients across ALL blocks

    WHY 8×8 BLOCKS?
        JPEG compression works on 8×8 blocks. When a deepfake is generated
        and then saved as JPEG, the compression artifacts interact with the
        synthetic content in detectable ways.
        
        Also, GAN upsampling (used in most face-swaps) creates characteristic
        patterns in 8×8 DCT blocks that don't appear in real images.

    WHAT IS DCT?
        DCT is like FFT but only uses cosine waves (no imaginary parts).
        It's the mathematical basis of JPEG compression.
        Each 8×8 block is decomposed into 64 frequency components:
        - Top-left coefficient = average brightness of the block (DC)
        - Other coefficients = various frequency patterns (AC)
        
        Real images have specific distributions of these coefficients.
        Manipulated images have abnormal distributions.

    Parameters:
        block_size (int): Size of DCT blocks (default 8, same as JPEG)
        embed_dim (int): Output embedding dimension (default 128)
    """

    def __init__(self, block_size=8, embed_dim=128):
        super().__init__()
        self.block_size = block_size

        # Number of DCT coefficients per block = block_size² = 64
        n_coeffs = block_size * block_size  # 64

        # MLP (Multi-Layer Perceptron) to process DCT coefficient statistics
        # Input: 64 * 2 = 128 (mean + std of each of the 64 DCT coefficients)
        self.mlp = nn.Sequential(
            nn.Linear(n_coeffs * 2, 256),  # 128 → 256
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),               # 10% dropout for regularization
            nn.Linear(256, embed_dim),      # 256 → 128
        )

    def forward(self, x):
        """
        Extract DCT block-level features from RGB images.

        Args:
            x: [B, 3, H, W] — batch of RGB images

        Returns:
            dct_embed: [B, embed_dim] — DCT-based features
        """
        # Step 1: RGB → Grayscale
        gray = x.mean(dim=1)  # [B, H, W]

        b, h, w = gray.shape
        bs = self.block_size  # 8

        # Step 2: Crop to exact multiple of block_size
        # If image is 224x224 and block_size=8: 224//8 = 28 blocks per side
        # 28 * 8 = 224, so no cropping needed (224 is already divisible by 8)
        h_blocks = h // bs  # 28
        w_blocks = w // bs  # 28
        gray = gray[:, :h_blocks * bs, :w_blocks * bs]  # [B, 224, 224]

        # Step 3: Reshape into 8×8 blocks
        # This is a clever reshape trick:
        # [B, 224, 224] → [B, 28, 8, 28, 8] → [B, 28, 28, 8, 8] → [B, 784, 8, 8]
        blocks = gray.reshape(b, h_blocks, bs, w_blocks, bs)  # [B, 28, 8, 28, 8]
        blocks = blocks.permute(0, 1, 3, 2, 4)                # [B, 28, 28, 8, 8]
        blocks = blocks.reshape(b, -1, bs, bs)                # [B, 784, 8, 8]
        # Now we have 784 blocks per image, each 8×8 pixels

        # Step 4: Apply 2D DCT to each block
        # We implement DCT-II manually because torch.fft.dctn is not available
        # in all PyTorch versions. The trick: DCT can be computed via FFT
        # by mirroring the input signal and extracting specific coefficients.
        #
        # Instead of the full DCT, we use a simpler but effective approach:
        # Apply 2D FFT and take the magnitude — this captures similar
        # frequency information to DCT for our detection purposes.
        fft_blocks = torch.fft.fft2(blocks.float(), dim=(-2, -1))
        dct_blocks = torch.abs(fft_blocks)  # magnitude as proxy for DCT
        # dct_blocks: [B, 784, 8, 8] — 64 frequency coefficients per block

        # Step 5: Flatten the 8×8 into 64 coefficients per block
        dct_flat = dct_blocks.reshape(b, -1, bs * bs)  # [B, 784, 64]

        # Step 6: Compute statistics across ALL blocks
        # For each of the 64 DCT positions, compute mean and std across 784 blocks
        # This summarizes the overall frequency distribution of the image
        mean_coeffs = dct_flat.mean(dim=1)  # [B, 64] — average of each coefficient
        std_coeffs = dct_flat.std(dim=1)    # [B, 64] — spread of each coefficient

        # Concatenate: [B, 128] (64 means + 64 stds)
        stats = torch.cat([mean_coeffs, std_coeffs], dim=1)

        # Step 7: Process through MLP to get final features
        dct_embed = self.mlp(stats)  # [B, 128]

        return dct_embed


class DualSpectralModule(nn.Module):
    """
    Convenience wrapper that combines both FFT and DCT analysis.

    This is used when you want a single module that does all spectral analysis.
    Branch B uses the individual extractors directly, but this wrapper is
    available if you want to use spectral features in other contexts.
    """

    def __init__(self, fft_dim=256, dct_dim=128):
        super().__init__()
        self.fft = SpectralFeatureExtractor(embed_dim=fft_dim)
        self.dct = DCTFeatureExtractor(embed_dim=dct_dim)
        self.total_dim = fft_dim + dct_dim  # 384

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W] — RGB images
        Returns:
            combined: [B, 384] — FFT + DCT features concatenated
        """
        fft_features = self.fft(x)  # [B, 256]
        dct_features = self.dct(x)  # [B, 128]
        return torch.cat([fft_features, dct_features], dim=1)  # [B, 384]
