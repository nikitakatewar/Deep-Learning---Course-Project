"""
utils/ — Utility modules for the deepfake detection pipeline
=============================================================

Components:
    FaceDetector (face_detect.py):
        - RetinaFace-based face detection using insightface
        - Detects, aligns, and crops faces from video frames
        - Returns 224x224 face crops ready for Branch A

    load_and_sample_frames (video_loader.py):
        - Opens video files with OpenCV
        - Uniformly samples N frames across the video duration
        - Returns raw frames for processing

    TemperatureScaler (calibrate.py):
        - Post-training calibration using Platt scaling
        - Learns a temperature parameter T on validation data
        - Calibrated score = sigmoid(logit / T)
        - Improves probability reliability without changing AUC
"""

# ─── Utils exports ───
from .face_detect import FaceDetector
from .video_loader import load_and_sample_frames
from .calibrate import TemperatureScaler, calibrate_scores
