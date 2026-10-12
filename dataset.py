"""
VideoDataset — Loads videos for training the deepfake detector
==============================================================

PURPOSE:
    This file handles ALL data loading for training and evaluation.
    It takes a directory of video files organized as:
        data_root/
        ├── real/   (authentic videos)
        └── fake/   (deepfake/manipulated videos)
    
    And converts each video into tensors that the model can process.

WHAT HAPPENS FOR EACH VIDEO:
    1. Open the video file with OpenCV (cv2)
    2. Sample N frames uniformly across the video
       - If video has 300 frames and N=16: take frames at positions
         [0, 20, 40, 60, ..., 280, 300]
    3. Resize each frame to 224×224 pixels
    4. Detect and crop faces from each frame (for Branch A)
    5. Apply data augmentations (random flips, color changes, compression)
    6. Normalize pixel values to match ImageNet statistics
    7. Return: (full_frames, face_crops, label)

WHAT IS DATA AUGMENTATION?
    Augmentation = randomly modifying training images to prevent overfitting.
    
    Without augmentation: the model memorizes specific videos
    With augmentation: the model learns GENERAL features of fakes
    
    Our augmentations:
    - Random horizontal flip (50% chance) — faces can face either direction
    - Color jitter — slight changes to brightness, contrast, saturation
    - Random grayscale (5% chance) — sometimes remove color info
    - Gaussian blur — slightly blur the image
    - JPEG compression (30% chance) — simulate social media recompression
      This is CRITICAL because real-world deepfakes are almost always
      recompressed when shared on WhatsApp, Twitter, Instagram, etc.

WHAT IS NORMALIZATION?
    Raw pixels: values 0-255 (or 0.0-1.0)
    Normalized pixels: shifted and scaled to have mean≈0 and std≈1
    
    We use ImageNet normalization because our backbones were pretrained on ImageNet:
    - mean = [0.485, 0.456, 0.406] for R, G, B
    - std  = [0.229, 0.224, 0.225] for R, G, B
    
    Formula: normalized = (pixel - mean) / std
"""

import io
import os
import glob
import random

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2

# Supported video file formats
# We check for all common video extensions since different datasets use different formats
# Celeb-DF uses .mp4, some others use .avi or .mkv
VIDEO_EXTS = ("*.mp4", "*.avi", "*.mkv", "*.mov", "*.wmv", "*.webm")


class VideoDataset(Dataset):
    """
    PyTorch Dataset for loading real/fake video pairs.

    WHAT IS A DATASET?
        A Dataset is a class that tells PyTorch:
        1. How many items there are (__len__)
        2. How to load item #N (__getitem__)
        
        PyTorch's DataLoader then handles batching, shuffling, and
        parallel loading automatically.

    Parameters:
        data_root (str): Path to directory containing 'real/' and 'fake/' subdirectories
        n_frames (int): Number of frames to sample per video (default 16)
        resize (int): Spatial resolution to resize frames to (default 224)
        augment (bool): Whether to apply training augmentations (True for training, False for validation)
        face_detector: Optional FaceDetector instance for cropping faces
                       If None, uses center crops as a fallback
    """

    def __init__(self, data_root, n_frames=16, resize=224, augment=True,
                 face_detector=None):
        # MUST call super().__init__() for Dataset to work properly
        super().__init__()

        self.n_frames = n_frames
        self.resize = resize
        self.augment = augment
        self.face_detector = face_detector

        # ─── Collect all video file paths and their labels ───
        # Each sample is a tuple: (file_path, label)
        # label 0 = real, label 1 = fake
        self.samples = []

        real_dir = os.path.join(data_root, "real")
        fake_dir = os.path.join(data_root, "fake")

        # Search for all supported video formats in the real/ directory
        for ext in VIDEO_EXTS:
            for vpath in sorted(glob.glob(os.path.join(real_dir, ext))):
                self.samples.append((vpath, 0))  # 0 = real

        # Same for fake/ directory
        for ext in VIDEO_EXTS:
            for vpath in sorted(glob.glob(os.path.join(fake_dir, ext))):
                self.samples.append((vpath, 1))  # 1 = fake

        # Print dataset statistics
        if len(self.samples) == 0:
            print(f"[WARNING] No video files found in {data_root}/real/ or {data_root}/fake/")
        else:
            n_real = sum(1 for _, l in self.samples if l == 0)
            n_fake = sum(1 for _, l in self.samples if l == 1)
            print(f"[Dataset] {data_root}: {n_real} real + {n_fake} fake = {len(self.samples)} videos")

        # ─── ImageNet normalization ───
        # These magic numbers come from computing the mean/std of all ImageNet images
        # Our pretrained backbones (EfficientNet, ConvNeXt) expect this normalization
        self.normalize = v2.Normalize(
            mean=[0.485, 0.456, 0.406],  # Mean pixel value per channel
            std=[0.229, 0.224, 0.225]    # Standard deviation per channel
        )

        # ─── Training augmentations ───
        if augment:
            self.frame_augment = v2.Compose([
                # 50% chance to flip horizontally (mirror image)
                v2.RandomHorizontalFlip(p=0.5),

                # Randomly change brightness, contrast, saturation, hue
                # These small changes help the model not rely on specific colors
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),

                # 5% chance to convert to grayscale
                # Forces the model to not rely solely on color cues
                v2.RandomGrayscale(p=0.05),

                # Slight Gaussian blur to simulate focus variations
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            ])
            # 30% chance of JPEG compression augmentation (applied separately)
            self.compress_prob = 0.3
        else:
            # No augmentation for validation/test data
            self.frame_augment = None
            self.compress_prob = 0.0

    def __len__(self):
        """How many videos in the dataset. Called by DataLoader."""
        return len(self.samples)

    def _load_frames(self, video_path):
        """
        Load and uniformly sample frames from a video file.

        UNIFORM SAMPLING:
            If a video has 300 frames and we want 16:
            We don't take the first 16 or last 16 or random 16.
            We take evenly-spaced frames: [0, 20, 40, 60, ..., 280, 300]
            This ensures we see the ENTIRE video, not just a small portion.

        Args:
            video_path: path to .mp4/.avi/.mkv file

        Returns:
            frames: [N, 3, H, W] float32 tensor with values in [0, 1]
                N = self.n_frames (e.g., 16)
                3 = RGB channels
                H = W = self.resize (e.g., 224)
        """
        # OpenCV's VideoCapture opens a video file for frame-by-frame reading
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        # Get total number of frames in the video
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total < 1:
            cap.release()
            raise ValueError(f"Video has 0 frames: {video_path}")

        # ─── Calculate which frame indices to sample ───
        if total <= self.n_frames:
            # Video is shorter than requested frames — use all frames
            # and duplicate the last frame to fill the gap
            indices = list(range(total))
            while len(indices) < self.n_frames:
                indices.append(indices[-1])  # Repeat last frame
        else:
            # Uniform sampling: pick n_frames evenly spaced indices
            # np.linspace(0, 299, 16) → [0, 20, 40, ..., 279, 299]
            indices = np.linspace(0, total - 1, self.n_frames, dtype=int).tolist()

        # ─── Read the selected frames ───
        frames = []
        for idx in indices:
            # Seek to the frame at position 'idx'
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            # Read the frame (ret=True if successful)
            ret, frame = cap.read()

            if not ret:
                # Frame read failed — use the last successful frame or a black frame
                if frames:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((self.resize, self.resize, 3), dtype=np.uint8))
                continue

            # OpenCV reads in BGR format — convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize to standard dimensions (224×224)
            frame = cv2.resize(frame, (self.resize, self.resize))

            frames.append(frame)

        # Release the video file handle
        cap.release()

        # ─── Convert to PyTorch tensor ───
        # Stack frames: [N, H, W, C] where C=3 (RGB)
        frames_np = np.stack(frames).astype(np.float32) / 255.0  # Scale 0-255 → 0.0-1.0

        # Rearrange to PyTorch's expected format: [N, C, H, W]
        # NumPy: [N, Height, Width, Channels] → PyTorch: [N, Channels, Height, Width]
        return torch.from_numpy(frames_np).permute(0, 3, 1, 2)

    def _extract_face_crops(self, frames_tensor):
        """
        Detect and crop faces from video frames.

        If a face detector is available (MTCNN), it will:
        1. Find the bounding box of the largest face in each frame
        2. Crop that region
        3. Resize to 224×224

        If no face detector is available, it uses a CENTER CROP as a fallback:
        - Takes the middle third of the image (rough face approximation)
        - Not as good but works without installing face detection libraries

        Args:
            frames_tensor: [N, 3, H, W] float tensor in [0, 1]

        Returns:
            face_crops: [N, 3, 224, 224] tensor — cropped face regions
        """
        if self.face_detector is not None:
            # Use MTCNN to detect faces and crop them
            bboxes = self.face_detector.detect_batch(frames_tensor)
            face_crops, _ = self.face_detector.crop_faces(
                frames_tensor, bboxes, output_size=224
            )
            return face_crops
        else:
            # Fallback: center crop (no face detection)
            import torch.nn.functional as F
            _, c, h, w = frames_tensor.shape

            # Take the center third of the image as a rough face region
            cy, cx = h // 2, w // 2      # Center coordinates
            half = min(h, w) // 3         # Half-size of crop region
            crops = frames_tensor[:, :, cy-half:cy+half, cx-half:cx+half]

            # Resize crops to 224×224 using bilinear interpolation
            return F.interpolate(
                crops, size=(224, 224), mode="bilinear", align_corners=False
            )

    def __getitem__(self, idx):
        """
        Load and preprocess one video.

        This is called by the DataLoader for each video in the batch.
        idx is the index into self.samples (e.g., idx=42 → 43rd video).

        Returns:
            full_frames: [T, 3, 224, 224] — normalized full video frames
            face_crops:  [T, 3, 224, 224] — normalized face crops
            label: scalar tensor — 0.0 (real) or 1.0 (fake)
        """
        video_path, label = self.samples[idx]

        try:
            # Step 1: Load raw frames [N, 3, H, W] in [0, 1]
            raw_frames = self._load_frames(video_path)
        except Exception as e:
            # If a video fails to load (corrupted file, etc.),
            # return a dummy sample instead of crashing the entire training
            print(f"[ERROR] Failed to load {video_path}: {e}")
            dummy = torch.zeros(self.n_frames, 3, self.resize, self.resize)
            return dummy, dummy, torch.tensor(label, dtype=torch.float32)

        # Step 2: Extract face crops (on raw [0,1] frames before normalization)
        face_crops = self._extract_face_crops(raw_frames)

        # Step 3: JPEG compression augmentation (30% chance during training)
        # This simulates what happens when deepfakes are shared on social media
        if self.compress_prob > 0 and random.random() < self.compress_prob:
            raw_frames = self._jpeg_compress(raw_frames)
            face_crops = self._jpeg_compress(face_crops)

        # Step 4: Apply augmentations (same random transform for frames AND faces)
        if self.frame_augment is not None:
            # Use the SAME random seed for both transforms
            # This ensures if frames are flipped, faces are flipped the same way
            seed = random.randint(0, 2**32 - 1)
            torch.manual_seed(seed)
            raw_frames = self.frame_augment(raw_frames)
            torch.manual_seed(seed)
            face_crops = self.frame_augment(face_crops)

        # Step 5: Normalize to ImageNet statistics
        # After this, pixel values are roughly mean=0, std=1
        full_frames = self.normalize(raw_frames)
        face_crops = self.normalize(face_crops)

        # Convert label to a float tensor (0.0 or 1.0)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        return full_frames, face_crops, label_tensor

    def _jpeg_compress(self, frames_tensor, quality_range=(30, 85)):
        """
        Simulate JPEG recompression artifacts.

        WHY THIS IS IMPORTANT:
            When a deepfake is created, it might be nearly undetectable.
            But when someone uploads it to WhatsApp or Twitter, the platform
            recompresses the video. This JPEG recompression creates artifacts
            that interact differently with real vs. fake content.

            By training with random JPEG compression, our model learns to
            detect fakes even after they've been through social media.

        HOW IT WORKS:
            1. Pick a random JPEG quality (30=very compressed, 85=mild compression)
            2. For each frame: encode → decode as JPEG (lossy round-trip)
            3. The resulting image has JPEG block artifacts (8x8 pixel blocks)

        Args:
            frames_tensor: [N, 3, H, W] float tensor in [0, 1]
            quality_range: (min_quality, max_quality)

        Returns:
            compressed: [N, 3, H, W] same shape with JPEG artifacts added
        """
        # Random quality for this batch (same quality for all frames in one video)
        quality = random.randint(*quality_range)
        compressed_frames = []

        for i in range(frames_tensor.shape[0]):
            # Convert tensor → numpy uint8 → BGR (OpenCV format)
            frame = (frames_tensor[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Encode as JPEG in memory (not to disk)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, encoded = cv2.imencode('.jpg', frame_bgr, encode_params)

            # Decode back — this round-trip introduces JPEG artifacts
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

            # Convert back to tensor
            compressed_frames.append(
                torch.from_numpy(decoded_rgb.astype(np.float32) / 255.0).permute(2, 0, 1)
            )

        return torch.stack(compressed_frames)


def build_dataloader(data_root, batch_size=4, n_frames=16, augment=True,
                     num_workers=2, face_detector=None):
    """
    Build a DataLoader from a data directory.

    WHAT IS A DATALOADER?
        A DataLoader wraps a Dataset and provides:
        - Automatic batching (group videos into batches of batch_size)
        - Shuffling (randomize order each epoch for training)
        - Parallel loading (use multiple CPU workers to load data faster)
        - Pin memory (speed up CPU→GPU transfer)

    Args:
        data_root: directory with real/ and fake/ subdirectories
        batch_size: videos per batch (limited by GPU memory)
        n_frames: frames per video (more frames = better but slower)
        augment: True for training, False for validation
        num_workers: parallel data loading processes
        face_detector: shared FaceDetector instance (or None for center crops)

    Returns:
        DataLoader instance — use in a for loop:
            for full_frames, face_crops, labels in loader:
                # Process one batch
    """
    from torch.utils.data import DataLoader

    dataset = VideoDataset(
        data_root=data_root,
        n_frames=n_frames,
        augment=augment,
        face_detector=face_detector,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=augment,      # Shuffle during training (not during validation)
        num_workers=num_workers,
        pin_memory=True,      # Speeds up CPU→GPU data transfer
        drop_last=augment,    # Drop incomplete batches during training
    )

    return loader
