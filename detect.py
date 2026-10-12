"""
Deepfake Video Detection CLI (v6)
==================================

Inference script for the 3-Branch Gated Fusion deepfake detector.

Architecture:
    Branch A (Face):     ConvNeXt-Tiny + SRM noise + CBAM → face manipulation artifacts
    Branch B (Frame):    ConvNeXt-Tiny + FFT/DCT spectral + CBAM → frequency anomalies
    Branch C (Temporal): Bidirectional GRU + Attention Pooling → temporal inconsistencies
    Fusion:              Learned gating network weights each branch per-video

Pipeline:
    1. Load video → sample N frames uniformly
    2. Detect faces with RetinaFace (insightface) → crop and align
    3. Run each frame through Branch A (face crops) and Branch B (full frames)
    4. Collect per-frame embeddings → feed sequence to Branch C (temporal)
    5. Gated fusion combines all 3 branch predictions
    6. Output: final score (0-100%), per-branch scores, gate weights

Usage:
    python detect.py --video input.mp4
    python detect.py --video input.mp4 --timeline    # Frame-by-frame analysis
    python detect.py --video input.mp4 --json         # Machine-readable output

Performance: Val AUC 0.9549 | 87.8% accuracy | 100% HeyGen detection
"""

import argparse
import json
import os
import sys
import time

import torch
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Deepfake Video Detector v3")
    p.add_argument("--video", type=str, required=True,
                   help="Path to input video file")
    p.add_argument("--weights", type=str, default="weights/best_model.pth",
                   help="Path to model weights")
    p.add_argument("--calibration", type=str, default="weights/calibration.json",
                   help="Path to calibration file")
    p.add_argument("--n_frames", type=int, default=16,
                   help="Number of frames to sample")
    p.add_argument("--json", action="store_true",
                   help="Output results as JSON")
    p.add_argument("--timeline", action="store_true",
                   help="Frame-by-frame timeline analysis (detects real→fake transitions)")
    p.add_argument("--no_srm", action="store_true",
                   help="Disable SRM (must match training config)")
    p.add_argument("--no_spectral", action="store_true",
                   help="Disable spectral (must match training config)")
    return p.parse_args()


def load_model(weights_path, device, use_srm=True, use_spectral=True):
    """Load the detection model."""
    from models.fused_model import FusedAIVideoDetector

    model = FusedAIVideoDetector(
        pretrained=False, use_srm=use_srm, use_spectral=use_spectral
    )
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
    else:
        print(f"[WARNING] No weights found at {weights_path}")

    model.to(device)
    model.eval()
    return model


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_calibration(calibration_path):
    """Load calibration data if available."""
    threshold = 0.50
    temperature = 1.0
    if calibration_path and os.path.exists(calibration_path):
        with open(calibration_path, "r") as f:
            cal = json.load(f)
        temperature = cal.get("temperature", 1.0)
        threshold = cal.get("threshold", 0.50)
    return temperature, threshold


# ─── Standard Detection (whole-video) ───

def detect_video(video_path, weights_path, calibration_path=None,
                 n_frames=16, use_srm=True, use_spectral=True):
    """Run detection on a single video (averages all frames)."""
    from utils.video_loader import load_and_sample_frames_raw, normalize_frames
    from utils.face_detect import FaceDetector

    device = get_device()
    model = load_model(weights_path, device, use_srm, use_spectral)

    # Load video frames
    raw_frames, fps, total_frames = load_and_sample_frames_raw(
        video_path, n_frames=n_frames
    )

    # Face detection
    face_det = FaceDetector(device=device)
    bboxes = face_det.detect_batch(raw_frames)
    face_crops, valid_mask = face_det.crop_faces(raw_frames, bboxes)

    # Normalize
    norm_frames = normalize_frames(raw_frames)
    norm_faces = normalize_frames(face_crops)

    # Inference
    video_batch = norm_frames.unsqueeze(0).to(device)
    face_batch = norm_faces.unsqueeze(0).to(device)

    with torch.no_grad():
        final_logit, face_l, frame_l, temp_l, gates = model.forward_with_gates(
            video_batch, face_batch
        )

    temperature, cal_threshold = load_calibration(calibration_path)

    final_score = torch.sigmoid(final_logit / temperature).item()
    face_score = torch.sigmoid(face_l).item()
    frame_score = torch.sigmoid(frame_l).item()
    temporal_score = torch.sigmoid(temp_l).item()
    gate_vals = gates.squeeze().cpu().tolist()

    if final_score > cal_threshold + 0.20:
        verdict = "LIKELY MANIPULATED"
        color = "RED"
    elif final_score > cal_threshold:
        verdict = "SUSPICIOUS"
        color = "YELLOW"
    else:
        verdict = "LIKELY AUTHENTIC"
        color = "GREEN"

    # AI-Generated Avatar Detection
    # When face branch says "real" but frame branch says "fake",
    # it's likely AI-generated content (e.g., HeyGen, D-ID, Synthesia)
    # SRM misses these because AI generates uniform noise patterns,
    # but spectral analysis catches frequency-domain artifacts
    ai_generated_flag = False
    if face_score < 0.30 and frame_score > 0.60:
        ai_generated_flag = True
        color = "YELLOW"

    return {
        "video": os.path.basename(video_path),
        "tampering_risk_score": round(final_score * 100, 1),
        "verdict": verdict,
        "confidence_level": color,
        "ai_generated_flag": ai_generated_flag,
        "branch_scores": {
            "face_forensic": round(face_score * 100, 1),
            "full_frame": round(frame_score * 100, 1),
            "temporal": round(temporal_score * 100, 1),
        },
        "gate_weights": {
            "face": round(gate_vals[0], 3),
            "frame": round(gate_vals[1], 3),
            "temporal": round(gate_vals[2], 3),
        },
        "metadata": {
            "faces_detected": int(valid_mask.sum().item()),
            "frames_sampled": n_frames,
            "source_fps": round(fps, 1),
            "total_source_frames": total_frames,
            "device": device,
            "srm_enabled": use_srm,
            "spectral_enabled": use_spectral,
        }
    }


# ─── Timeline Analysis (frame-by-frame) ───

def detect_timeline(video_path, weights_path, calibration_path=None,
                    n_frames=32, use_srm=True, use_spectral=True):
    """
    Frame-by-frame timeline analysis.
    
    Instead of averaging all frames, scores EACH frame individually
    to detect real→fake transitions within a video.
    
    Returns per-frame scores and transition detection.
    """
    import cv2
    from utils.face_detect import FaceDetector
    from utils.video_loader import normalize_frames

    device = get_device()
    model = load_model(weights_path, device, use_srm, use_spectral)
    face_det = FaceDetector(device=device)
    temperature, cal_threshold = load_calibration(calibration_path)

    # Sample more frames for timeline (at least 32 for good resolution)
    n_timeline = max(n_frames, 32)

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps > 0 else 0

    if total < 1:
        cap.release()
        print("[ERROR] Cannot read video")
        return None

    indices = np.linspace(0, total - 1, n_timeline, dtype=int).tolist()

    # Score each frame individually
    frame_results = []

    for i, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (224, 224))

        # Convert to tensor [1, 3, 224, 224]
        frame_tensor = torch.from_numpy(
            frame_resized.astype(np.float32) / 255.0
        ).permute(2, 0, 1).unsqueeze(0)

        # Detect face
        bboxes = face_det.detect_batch(frame_tensor)
        face_crop, _ = face_det.crop_faces(frame_tensor, bboxes)

        # Normalize
        norm_frame = normalize_frames(frame_tensor)
        norm_face = normalize_frames(face_crop)

        # Score with face branch (most reliable for per-frame)
        with torch.no_grad():
            norm_frame_dev = norm_frame.to(device)
            norm_face_dev = norm_face.to(device)

            # Get per-frame scores from each branch
            _, face_logit = model.face_branch(norm_face_dev)
            _, frame_logit = model.frame_branch(norm_frame_dev)

            # Combined score (weighted average of branches)
            combined_logit = 0.6 * face_logit + 0.4 * frame_logit
            score = torch.sigmoid(combined_logit / temperature).item()

        timestamp = idx / fps if fps > 0 else 0
        frame_results.append({
            "frame_index": idx,
            "timestamp": round(timestamp, 2),
            "fake_score": round(score * 100, 1),
            "label": "FAKE" if score > cal_threshold else "REAL"
        })

    cap.release()

    # Detect transition point
    scores = [r["fake_score"] for r in frame_results]
    transition = detect_transition(scores, frame_results, cal_threshold * 100)

    # Overall classification
    avg_score = np.mean(scores)
    n_fake = sum(1 for s in scores if s > cal_threshold * 100)
    fake_ratio = n_fake / len(scores)

    if fake_ratio > 0.8:
        video_type = "FULLY MANIPULATED"
    elif fake_ratio > 0.3:
        video_type = "PARTIAL MANIPULATION (real->fake transition detected)"
    else:
        video_type = "LIKELY AUTHENTIC"

    return {
        "video": os.path.basename(video_path),
        "video_type": video_type,
        "average_score": round(avg_score, 1),
        "fake_frame_ratio": f"{n_fake}/{len(scores)}",
        "duration_sec": round(duration, 1),
        "transition": transition,
        "frames": frame_results,
    }


def detect_transition(scores, frame_results, threshold):
    """
    Detect where a video transitions from real to fake.
    
    Uses a sliding window to find a sharp jump in scores.
    """
    if len(scores) < 4:
        return None

    window = 3  # average over 3 frames for smoothing
    smoothed = []
    for i in range(len(scores)):
        start = max(0, i - window // 2)
        end = min(len(scores), i + window // 2 + 1)
        smoothed.append(np.mean(scores[start:end]))

    # Find biggest score jump
    max_jump = 0
    transition_idx = None

    for i in range(1, len(smoothed)):
        jump = smoothed[i] - smoothed[i - 1]
        if jump > max_jump:
            max_jump = jump
            transition_idx = i

    # Only report if jump is significant (> 20 percentage points)
    if max_jump > 20 and transition_idx is not None:
        t = frame_results[transition_idx]["timestamp"]
        return {
            "detected": True,
            "timestamp_sec": t,
            "timestamp_display": f"{int(t//60)}:{int(t%60):02d}",
            "score_jump": round(max_jump, 1),
            "description": f"Video transitions from REAL -> FAKE at ~{t:.1f}s"
        }
    
    return {"detected": False, "description": "No transition detected — video is consistent throughout"}


# ─── Pretty Print ───

def print_results(results):
    """Pretty-print standard detection results."""
    score = results["tampering_risk_score"]
    verdict = results["verdict"]
    branches = results["branch_scores"]
    gates = results["gate_weights"]
    meta = results["metadata"]
    ai_flag = results.get("ai_generated_flag", False)

    bar_len = 40
    filled = int(score / 100 * bar_len)
    bar = "#" * filled + "-" * (bar_len - filled)

    print()
    print("=" * 56)
    print("  DEEPFAKE VIDEO DETECTION REPORT")
    print("=" * 56)
    print()
    print(f"  Video:  {results['video']}")
    print()
    print(f"  Tampering Risk Score: {score}%")
    print(f"  [{bar}]")
    print()
    print(f"  Verdict: {verdict}")

    if ai_flag:
        print()
        print("  [!] AI-GENERATED CONTENT DETECTED")
        print("  This video appears to be created by an AI avatar tool")
        print("  (e.g., HeyGen, D-ID, Synthesia)")
        print("  Detection: Face noise is clean (no swap artifacts),")
        print("  but spectral analysis found AI generation patterns.")

    print()
    print("-" * 56)
    print("  BRANCH BREAKDOWN")
    print("-" * 56)
    print(f"  Face Forensic (SRM):   {branches['face_forensic']:5.1f}%  (gate: {gates['face']:.3f})")
    print(f"  Full Frame (Spectral): {branches['full_frame']:5.1f}%  (gate: {gates['frame']:.3f})")
    print(f"  Temporal Analysis:     {branches['temporal']:5.1f}%  (gate: {gates['temporal']:.3f})")
    print()
    print("-" * 56)
    print("  METADATA")
    print("-" * 56)
    print(f"  Faces detected:  {meta['faces_detected']}/{meta['frames_sampled']} frames")
    print(f"  Source FPS:       {meta['source_fps']}")
    print(f"  Total frames:    {meta['total_source_frames']}")
    print(f"  Device:          {meta['device']}")
    print(f"  SRM filters:     {'ON' if meta['srm_enabled'] else 'OFF'}")
    print(f"  Spectral (FFT):  {'ON' if meta['spectral_enabled'] else 'OFF'}")
    print("=" * 56)
    print()


def print_timeline(results):
    """Pretty-print timeline analysis results."""
    print()
    print("=" * 64)
    print("  DEEPFAKE TIMELINE ANALYSIS")
    print("=" * 64)
    print()
    print(f"  Video:     {results['video']}")
    print(f"  Duration:  {results['duration_sec']}s")
    print(f"  Type:      {results['video_type']}")
    print(f"  Avg Score: {results['average_score']}%")
    print(f"  Fake Frames: {results['fake_frame_ratio']}")
    print()

    # Transition info
    tr = results["transition"]
    if tr and tr["detected"]:
        print(f"  [!] TRANSITION DETECTED!")
        print(f"  {tr['description']}")
        print(f"  Score jump: +{tr['score_jump']}%")
        print()

    # Frame-by-frame timeline
    print("-" * 64)
    print("  TIME      SCORE    LABEL     VISUAL")
    print("-" * 64)

    for fr in results["frames"]:
        ts = fr["timestamp"]
        score = fr["fake_score"]
        label = fr["label"]

        # Visual bar
        bar_len = 30
        filled = int(score / 100 * bar_len)
        bar = "#" * filled + "-" * (bar_len - filled)

        # Color indicator (ASCII-safe for Windows)
        if label == "FAKE":
            indicator = "[X]"
        elif score > 30:
            indicator = "[?]"
        else:
            indicator = "[ ]"

        ts_str = f"{int(ts//60)}:{ts%60:05.2f}"
        print(f"  {ts_str}  {score:5.1f}%  {label:4s}  {indicator} {bar}")

    print()
    print("=" * 64)
    print()


def main():
    args = parse_args()

    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        sys.exit(1)

    start = time.time()

    if args.timeline:
        # Timeline mode: frame-by-frame analysis
        results = detect_timeline(
            video_path=args.video,
            weights_path=args.weights,
            calibration_path=args.calibration,
            n_frames=args.n_frames,
            use_srm=not args.no_srm,
            use_spectral=not args.no_spectral,
        )
        elapsed = time.time() - start
        results["inference_time_sec"] = round(elapsed, 2)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_timeline(results)
    else:
        # Standard mode: whole-video analysis
        results = detect_video(
            video_path=args.video,
            weights_path=args.weights,
            calibration_path=args.calibration,
            n_frames=args.n_frames,
            use_srm=not args.no_srm,
            use_spectral=not args.no_spectral,
        )
        elapsed = time.time() - start
        results["metadata"]["inference_time_sec"] = round(elapsed, 2)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print_results(results)


if __name__ == "__main__":
    main()