"""
Face detection utility using RetinaFace (via InsightFace).
Provides bounding boxes for the largest face in each frame.

IMPORTANT: This must stay aligned with pre_extract.py's face detection!
Both use: RetinaFace → pick largest face by area → margin=20
"""

import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image

# Try RetinaFace (InsightFace), fall back to MTCNN
try:
    from insightface.app import FaceAnalysis
    HAS_RETINAFACE = True
except ImportError:
    HAS_RETINAFACE = False
    from facenet_pytorch import MTCNN


class FaceDetector:
    """Wraps RetinaFace (or MTCNN fallback) to detect faces in video frames.
    
    Aligned with pre_extract.py to ensure train/inference consistency:
    - Detect ALL faces, then pick largest by AREA
    - margin=20 around detected face box
    """

    def __init__(self, device="cpu", min_face_size=40, margin=20):
        self.device = device
        self.margin = margin
        self.use_retinaface = HAS_RETINAFACE

        if HAS_RETINAFACE:
            # RetinaFace via InsightFace
            # 'buffalo_l' is the large detection model
            self.detector = FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.detector.prepare(ctx_id=0 if 'cuda' in str(device) else -1,
                                det_size=(640, 640))
            print("[FaceDetector] Using RetinaFace (InsightFace)")
        else:
            # Fallback to MTCNN
            self.detector = MTCNN(
                keep_all=True,
                min_face_size=min_face_size,
                device=device,
                post_process=False
            )
            print("[FaceDetector] Using MTCNN (fallback)")

    @torch.no_grad()
    def detect_batch(self, frames_tensor):
        """Detect the largest face (by area) in each frame.

        Args:
            frames_tensor: [N, C, H, W] float tensor in [0, 1] range

        Returns:
            bboxes: list of N elements, each either [x1, y1, x2, y2] (ints) or None
        """
        bboxes = []
        _, _, h, w = frames_tensor.shape

        for i in range(frames_tensor.shape[0]):
            frame_np = (frames_tensor[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

            if self.use_retinaface:
                # InsightFace expects BGR numpy array
                import cv2
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                faces = self.detector.get(frame_bgr)

                if faces and len(faces) > 0:
                    # Pick largest face by area
                    best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
                    box = best_face.bbox

                    x1 = max(0, int(box[0]) - self.margin)
                    y1 = max(0, int(box[1]) - self.margin)
                    x2 = min(w, int(box[2]) + self.margin)
                    y2 = min(h, int(box[3]) + self.margin)
                    bboxes.append([x1, y1, x2, y2])
                else:
                    bboxes.append(None)
            else:
                # MTCNN fallback
                pil_img = Image.fromarray(frame_np)
                boxes, probs = self.detector.detect(pil_img)

                if boxes is not None and len(boxes) > 0:
                    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                    best_idx = int(np.argmax(areas))
                    box = boxes[best_idx]

                    x1 = max(0, int(box[0]) - self.margin)
                    y1 = max(0, int(box[1]) - self.margin)
                    x2 = min(w, int(box[2]) + self.margin)
                    y2 = min(h, int(box[3]) + self.margin)
                    bboxes.append([x1, y1, x2, y2])
                else:
                    bboxes.append(None)

        return bboxes

    def crop_faces(self, frames_tensor, bboxes, output_size=224):
        """Crop and resize faces from frames using detected bboxes."""
        crops = []
        valid = []

        for i, bbox in enumerate(bboxes):
            if bbox is not None:
                x1, y1, x2, y2 = bbox
                if x1 >= x2 or y1 >= y2:
                    bbox = None
                else:
                    face = frames_tensor[i:i+1, :, y1:y2, x1:x2]
                    face_resized = F.interpolate(
                        face, size=(output_size, output_size),
                        mode="bilinear", align_corners=False
                    )
                    crops.append(face_resized.squeeze(0))
                    valid.append(True)
            
            if bbox is None:
                # Fallback: center crop
                _, c, h, w = frames_tensor.shape
                cy, cx = h // 2, w // 2
                half = min(h, w) // 3
                fallback = frames_tensor[i:i+1, :, cy-half:cy+half, cx-half:cx+half]
                fallback_resized = F.interpolate(
                    fallback, size=(output_size, output_size),
                    mode="bilinear", align_corners=False
                )
                crops.append(fallback_resized.squeeze(0))
                valid.append(False)

        face_crops = torch.stack(crops)
        valid_mask = torch.tensor(valid, dtype=torch.bool)
        return face_crops, valid_mask
