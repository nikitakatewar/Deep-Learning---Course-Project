"""
Video frame extraction and preprocessing.
Uses OpenCV for robust decoding across platforms and codecs.
"""

import cv2
import torch
import numpy as np


# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_and_sample_frames(video_path, n_frames=16, resize=224, normalize=True):
    """Load a video file and uniformly sample n_frames across its duration.

    Args:
        video_path: path to .mp4 video file
        n_frames: number of frames to sample (uniformly spaced)
        resize: resize frames to this square size (pixels)
        normalize: if True, apply ImageNet mean/std normalization

    Returns:
        frames: [n_frames, 3, resize, resize] float32 tensor in [0,1] (or normalized)
        fps: original video FPS
        total_frames: total frame count in source video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if total_frames < 1:
        cap.release()
        raise ValueError(f"Video has 0 frames: {video_path}")

    # Compute uniformly-spaced frame indices
    if total_frames <= n_frames:
        indices = list(range(total_frames))
        # Pad by repeating last frame if video is too short
        while len(indices) < n_frames:
            indices.append(indices[-1])
    else:
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int).tolist()

    frames = []
    for target_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
        ret, frame = cap.read()
        if not ret:
            # Fallback: use last successfully read frame or a black frame
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((resize, resize, 3), dtype=np.uint8))
            continue

        # BGR → RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize
        frame = cv2.resize(frame, (resize, resize), interpolation=cv2.INTER_LINEAR)
        frames.append(frame)

    cap.release()

    # Stack into numpy array [N, H, W, C] → tensor [N, C, H, W]
    frames_np = np.stack(frames, axis=0).astype(np.float32) / 255.0
    frames_tensor = torch.from_numpy(frames_np).permute(0, 3, 1, 2)

    if normalize:
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        frames_tensor = (frames_tensor - mean) / std

    return frames_tensor, fps, total_frames


def load_and_sample_frames_raw(video_path, n_frames=16, resize=224):
    """Same as load_and_sample_frames but returns unnormalized [0,1] frames.

    Useful when you need raw pixels for face detection before normalizing.

    Returns:
        frames: [n_frames, 3, resize, resize] float32 tensor in [0, 1]
        fps: original video FPS
        total_frames: total frame count
    """
    return load_and_sample_frames(
        video_path, n_frames=n_frames, resize=resize, normalize=False
    )


def normalize_frames(frames_tensor):
    """Apply ImageNet normalization to a [N, 3, H, W] tensor in [0, 1] range.

    Args:
        frames_tensor: unnormalized float tensor

    Returns:
        normalized tensor
    """
    mean = torch.tensor(IMAGENET_MEAN, device=frames_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=frames_tensor.device).view(1, 3, 1, 1)
    return (frames_tensor - mean) / std
