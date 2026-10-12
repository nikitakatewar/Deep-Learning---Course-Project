"""
Pre-extract faces and frames from all training videos.
=========================================================

PROBLEM:
    MTCNN face detection runs on every frame, every batch, every epoch.
    This is 90% of the training time.

SOLUTION:
    Run face detection ONCE, save results to disk.
    Training then loads cached images instead of re-detecting faces.

    Speed gain: ~9x faster training (30+ hours → ~4 hours)

Usage:
    python pre_extract.py --data_dir data/train --n_frames 16
    python pre_extract.py --data_dir data/val --n_frames 16
"""

import argparse
import os
import glob
import json

import cv2
import numpy as np
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Pre-extract faces and frames")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Directory with real/ and fake/ subdirectories")
    p.add_argument("--n_frames", type=int, default=16,
                   help="Frames to sample per video")
    p.add_argument("--resize", type=int, default=224,
                   help="Frame size")
    p.add_argument("--cache_dir", type=str, default=None,
                   help="Where to save cached data (default: data_dir + '_cache')")
    return p.parse_args()


def extract_frames(video_path, n_frames=16, resize=224):
    """Load and uniformly sample frames from a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 1:
        cap.release()
        return None

    if total <= n_frames:
        indices = list(range(total))
        while len(indices) < n_frames:
            indices.append(indices[-1])
    else:
        indices = np.linspace(0, total - 1, n_frames, dtype=int).tolist()

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            if frames:
                frames.append(frames[-1].copy())
            else:
                frames.append(np.zeros((resize, resize, 3), dtype=np.uint8))
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (resize, resize))
        frames.append(frame)

    cap.release()
    return frames  # list of [H, W, 3] uint8 numpy arrays


def detect_and_crop_face(frame_rgb, detector, margin=20, output_size=224, use_retinaface=True):
    """Detect the largest face in a frame and return the crop."""
    h, w = frame_rgb.shape[:2]

    if use_retinaface:
        # InsightFace expects BGR
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        faces = detector.get(frame_bgr)

        if faces and len(faces) > 0:
            best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            box = best_face.bbox

            x1 = max(0, int(box[0]) - margin)
            y1 = max(0, int(box[1]) - margin)
            x2 = min(w, int(box[2]) + margin)
            y2 = min(h, int(box[3]) + margin)

            face_crop = frame_rgb[y1:y2, x1:x2]
            face_crop = cv2.resize(face_crop, (output_size, output_size))
            return face_crop, True
        else:
            # Fallback: center crop
            cy, cx = h // 2, w // 2
            half = min(h, w) // 3
            face_crop = frame_rgb[cy-half:cy+half, cx-half:cx+half]
            face_crop = cv2.resize(face_crop, (output_size, output_size))
            return face_crop, False
    else:
        # MTCNN fallback
        from PIL import Image
        pil_img = Image.fromarray(frame_rgb)
        boxes, probs = detector.detect(pil_img)

        if boxes is not None and len(boxes) > 0:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best_idx = int(np.argmax(areas))
            box = boxes[best_idx]

            x1 = max(0, int(box[0]) - margin)
            y1 = max(0, int(box[1]) - margin)
            x2 = min(w, int(box[2]) + margin)
            y2 = min(h, int(box[3]) + margin)

            face_crop = frame_rgb[y1:y2, x1:x2]
            face_crop = cv2.resize(face_crop, (output_size, output_size))
            return face_crop, True
        else:
            cy, cx = h // 2, w // 2
            half = min(h, w) // 3
            face_crop = frame_rgb[cy-half:cy+half, cx-half:cx+half]
            face_crop = cv2.resize(face_crop, (output_size, output_size))
            return face_crop, False


def main():
    args = parse_args()
    
    # Setup face detector: RetinaFace (preferred) or MTCNN (fallback)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[+] Device: {device}")
    
    try:
        from insightface.app import FaceAnalysis
        detector = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        detector.prepare(ctx_id=0 if device == "cuda" else -1, det_size=(640, 640))
        use_retinaface = True
        print("[+] Face detector: RetinaFace (InsightFace)")
    except ImportError:
        from facenet_pytorch import MTCNN
        detector = MTCNN(
            keep_all=True,
            min_face_size=40,
            device=device,
            post_process=False,
        )
        use_retinaface = False
        print("[+] Face detector: MTCNN (fallback)")
    
    cache_dir = args.cache_dir or (args.data_dir + "_png_cache")
    
    # Process both real and fake
    for label in ["real", "fake"]:
        video_dir = os.path.join(args.data_dir, label)
        if not os.path.isdir(video_dir):
            print(f"[SKIP] {video_dir} not found")
            continue
        
        # Find all videos
        videos = []
        for ext in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
            videos.extend(sorted(glob.glob(os.path.join(video_dir, ext))))
        
        print(f"\n[{label.upper()}] Processing {len(videos)} videos...")
        
        # Output directories
        frame_cache = os.path.join(cache_dir, label, "frames")
        face_cache = os.path.join(cache_dir, label, "faces")
        os.makedirs(frame_cache, exist_ok=True)
        os.makedirs(face_cache, exist_ok=True)
        
        stats = {"total": len(videos), "success": 0, "failed": 0, "faces_found": 0}
        
        for video_path in tqdm(videos, desc=f"  {label}"):
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # Skip if already cached
            frame_dir = os.path.join(frame_cache, video_name)
            face_dir = os.path.join(face_cache, video_name)
            
            if os.path.isdir(frame_dir) and len([f for f in os.listdir(frame_dir) if f.endswith('.png')]) == args.n_frames:
                stats["success"] += 1
                continue
            
            # Extract frames
            frames = extract_frames(video_path, args.n_frames, args.resize)
            if frames is None:
                stats["failed"] += 1
                continue
            
            os.makedirs(frame_dir, exist_ok=True)
            os.makedirs(face_dir, exist_ok=True)
            
            n_faces = 0
            for i, frame_rgb in enumerate(frames):
                # Save full frame as PNG (LOSSLESS — preserves SRM artifacts)
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(frame_dir, f"{i:03d}.png"), frame_bgr)
                
                # Detect face and save crop
                face_crop, found = detect_and_crop_face(frame_rgb, detector, use_retinaface=use_retinaface)
                face_bgr = cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(face_dir, f"{i:03d}.png"), face_bgr)
                
                if found:
                    n_faces += 1
            
            stats["success"] += 1
            stats["faces_found"] += n_faces
        
        print(f"  Success: {stats['success']}/{stats['total']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Faces found: {stats['faces_found']} / {stats['success'] * args.n_frames} frames")
    
    # Save manifest
    manifest = {
        "n_frames": args.n_frames,
        "resize": args.resize,
        "source_dir": os.path.abspath(args.data_dir),
        "cache_dir": os.path.abspath(cache_dir),
    }
    with open(os.path.join(cache_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"  PRE-EXTRACTION COMPLETE")
    print(f"  Cache saved to: {cache_dir}")
    print(f"{'='*50}")
    print(f"\nNow train with:")
    print(f"  python train_fast.py --data_dir data/train --val_dir data/val --n_frames {args.n_frames} --epochs 10")


if __name__ == "__main__":
    main()
