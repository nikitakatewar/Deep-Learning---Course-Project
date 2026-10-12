"""
Fast Training Pipeline for AI Video Deepfake Detector (v6)
===========================================================

Architecture:  ConvNeXt-Tiny + CBAM + SRM/Spectral + Bidirectional GRU
Performance:   Val AUC 0.9549 | Eval Acc 87.8% | HeyGen 100%
Parameters:    ~63.5M total (61.4M branches + 2.1M fusion)
Training Time: ~18 hours on RTX 4060 (4 phases)

TRAINING PHASES:
    Phase 1 — Face Branch (Branch A):
        - Trains ConvNeXt + SRM + CBAM on face crops
        - Backbone frozen for first 2 epochs, then unfrozen with 10x lower LR
        - Initial LR: args.lr * 3 (aggressive for randomly-initialized heads)
        - Target: AUC > 0.90 (achieved 0.914)

    Phase 2 — Frame Branch (Branch B):
        - Trains ConvNeXt + FFT/DCT + CBAM on full frames
        - Backbone frozen for first 3 epochs (one extra — spectral heads need more time)
        - CRITICAL FIX: Discriminative LR after unfreezing:
            * Backbone: args.lr * 0.05 (preserve ImageNet features)
            * Heads:    args.lr (aggressive for random FFT/DCT extractors)
        - Patience = 7 (full-frame detection is fundamentally harder)
        - Target: AUC > 0.60 (achieved 0.652 — typical ceiling for full-frame)

    Phase 3 — Temporal Branch (Branch C):
        - Trains BiGRU + Attention Pooling on frame sequences
        - Uses frozen embeddings from Branch A + B (no backbone training)
        - AUTO-FALLBACK: If AUC < 0.60, skip temporal and use 2-branch fusion
        - Target: AUC > 0.90 (achieved 0.948)

    Phase 4 — End-to-End Fusion:
        - Fine-tunes entire model with Gated Fusion + MLP
        - CRITICAL FIX: Discriminative LR:
            * Branch params (61.4M): args.lr * 0.03 (gentle, already trained)
            * Fusion params (2.1M): args.lr (aggressive, randomly initialized)
        - This fix improved E2E from 0.67 AUC to 0.95+ AUC
        - Target: AUC > 0.95 (achieved 0.9549)

KEY TRAINING DECISIONS:
    - BCE + Label Smoothing (not FocalLoss — FocalLoss kills gradients)
    - CosineAnnealingLR (smooth decay, no sharp drops)
    - Gradient accumulation (effective batch = batch_size * grad_accum)
    - Mixed precision (fp16 via torch.amp for 2x speed on RTX 4060)
    - Gradient clipping (max_norm=1.0 to prevent exploding gradients)

USAGE:
    # Full training from scratch:
    python train_fast.py --data_dir data/train --val_dir data/val --n_frames 16

    # Resume and retrain only Phase 2+4:
    python train_fast.py --resume weights/merged_a_c_fresh_b.pth --skip_phase1 --skip_phase3

    # Only E2E fusion:
    python train_fast.py --resume weights/best_branch_c.pth --e2e_only
"""

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torchvision.transforms.v2 as v2


def get_criterion(label_smoothing=0.1):
    """BCE with label smoothing.
    
    Instead of hard targets 0/1, uses soft targets 0.05/0.95.
    This prevents overconfidence while still producing well-separated logits.
    
    Why not FocalLoss?
    - FocalLoss gamma=2.0 aggressively down-weights easy examples
    - As model improves, ALL examples become 'easy' → gradients vanish
    - Logits converge near 0 → sigmoid(0) ≈ 50% for everything
    
    Why BCE + Label Smoothing?
    - BCE produces strong gradients that push logits far from 0
    - Label smoothing softens targets (0→0.05, 1→0.95)
    - This regularizes without killing gradient magnitude
    """
    return nn.BCEWithLogitsLoss()


def smooth_labels(labels, eps=0.1):
    """Apply label smoothing: 0 → eps/2, 1 → 1-eps/2.
    
    NEW TECHNOLOGY: Label Smoothing (Szegedy et al., 2016)
    Prevents overconfident predictions that don't generalize.
    """
    return labels * (1 - eps) + eps / 2

# ==========================================
# Main Training Loop
# ==========================================

"""
4-phase training pipeline:
  Phase 1: Branch A (Face + SRM) — ConvNeXt-Tiny + CBAM
  Phase 2: Branch B (Frame + Spectral) — ConvNeXt-Tiny + CBAM
  Phase 3: Branch C (Temporal GRU) — Bidirectional GRU + Attention Pooling
  Phase 4: End-to-End Gated Fusion (all 3 branches)

FALLBACK: If Phase 3 temporal AUC < 0.60, skip temporal and
fall back to 2-branch fusion (A+B only) for Phase 4.

Usage:
    python pre_extract.py --data_dir data/train --n_frames 16
    python pre_extract.py --data_dir data/val --n_frames 16
    python train_fast.py --data_dir data/train --val_dir data/val --n_frames 16
"""

import argparse
import csv
import os
import time
import glob
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_  # NEW: Gradient clipping for stability
# sigmoid_focal_loss removed — replaced with BCE + Label Smoothing (see get_criterion)
from tqdm import tqdm
import torchvision.transforms.v2 as v2

from models import FusedAIVideoDetector


# ─── NEW: Mixup Augmentation (Zhang et al., 2018) ───
# Mixes two samples with a random weight lambda, forcing the model
# to learn smoother decision boundaries instead of memorizing artifacts.
# This significantly improves generalization to unseen deepfake types.

def mixup_data(x, y, alpha=0.4):
    """Apply Mixup augmentation to a batch.
    
    Args:
        x: input tensor (or tuple of tensors)
        y: labels tensor
        alpha: Beta distribution parameter (0.4 = moderate mixing)
    
    Returns:
        mixed_x, mixed_y: mixed inputs and soft labels
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # Ensure lam >= 0.5 (dominant sample stays dominant)
    else:
        lam = 1.0
    
    batch_size = y.size(0)
    index = torch.randperm(batch_size, device=y.device)
    
    if isinstance(x, (list, tuple)):
        mixed_x = tuple(lam * xi + (1 - lam) * xi[index] for xi in x)
    else:
        mixed_x = lam * x + (1 - lam) * x[index]
    
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y


# ─── Cached Dataset ───

class CachedVideoDataset(Dataset):
    """
    Loads pre-extracted frames and face crops from disk.
    
    Instead of:
        video → decode → detect faces → crop → augment → train
    Now:
        cached PNGs → load → augment → train
    
    This eliminates the MTCNN bottleneck entirely.
    Uses PNG (lossless) to preserve SRM-detectable manipulation artifacts.
    """
    
    def __init__(self, data_root, n_frames=16, augment=True):
        super().__init__()
        self.n_frames = n_frames
        self.augment = augment
        
        # Try PNG cache first, fall back to JPEG cache
        cache_dir = data_root + "_png_cache"
        if not os.path.isdir(cache_dir):
            cache_dir = data_root + "_cache"
            print(f"[WARNING] PNG cache not found, falling back to JPEG cache: {cache_dir}")
        self.samples = []  # (frame_dir, face_dir, label)
        
        for label_name, label_val in [("real", 0), ("fake", 1)]:
            frame_base = os.path.join(cache_dir, label_name, "frames")
            face_base = os.path.join(cache_dir, label_name, "faces")
            
            if not os.path.isdir(frame_base):
                print(f"[WARNING] Cache not found: {frame_base}")
                continue
            
            for video_name in sorted(os.listdir(frame_base)):
                frame_dir = os.path.join(frame_base, video_name)
                face_dir = os.path.join(face_base, video_name)
                
                if os.path.isdir(frame_dir) and os.path.isdir(face_dir):
                    self.samples.append((frame_dir, face_dir, label_val))
        
        n_real = sum(1 for _, _, l in self.samples if l == 0)
        n_fake = sum(1 for _, _, l in self.samples if l == 1)
        print(f"[CachedDataset] {data_root}: {n_real} real + {n_fake} fake = {len(self.samples)} videos")
        
        self.normalize = v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        if augment:
            self.frame_augment = v2.Compose([
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
                v2.RandomGrayscale(p=0.05),
                v2.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
            ])
            # NEW: Random downscale+upscale — breaks resolution bias
            # 30% chance to downscale to 112-180px then back to 224px
            # Teaches model that low resolution != fake
            self.downscale_prob = 0.3
            # Random erasing on full frames ONLY — prevents watermark shortcut learning
            # (HeyGen videos have a logo in bottom-right that model could memorize)
            self.frame_erasing = v2.RandomErasing(p=0.3, scale=(0.02, 0.15), ratio=(0.3, 3.3))
            # REDUCED from 0.3 — heavy JPEG destroys SRM artifacts on face crops
            self.compress_prob = 0.1
        else:
            self.frame_augment = None
            self.compress_prob = 0.0
            self.downscale_prob = 0.0
    
    def __len__(self):
        return len(self.samples)
    
    def _load_images(self, img_dir):
        """Load all cached images (PNG or JPEG) from a directory."""
        files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        if not files:
            files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        
        if len(files) < self.n_frames:
            # Pad by repeating last image
            while len(files) < self.n_frames:
                files.append(files[-1] if files else None)
        
        frames = []
        for fpath in files[:self.n_frames]:
            if fpath is None:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                img = cv2.imread(fpath)
                if img is None:
                    frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames.append(img)
        
        # [N, H, W, 3] uint8 → [N, 3, H, W] float32 [0, 1]
        arr = np.stack(frames).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(0, 3, 1, 2)
    
    def _jpeg_compress(self, frames_tensor, quality_range=(30, 85)):
        """Simulate JPEG recompression artifacts."""
        quality = random.randint(*quality_range)
        compressed = []
        for i in range(frames_tensor.shape[0]):
            frame = (frames_tensor[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, encoded = cv2.imencode('.jpg', frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            decoded_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            compressed.append(torch.from_numpy(decoded_rgb.astype(np.float32) / 255.0).permute(2, 0, 1))
        return torch.stack(compressed)
    
    def __getitem__(self, idx):
        frame_dir, face_dir, label = self.samples[idx]
        
        try:
            full_frames = self._load_images(frame_dir)
            face_crops = self._load_images(face_dir)
        except Exception as e:
            print(f"[ERROR] Failed to load {frame_dir}: {e}")
            dummy = torch.zeros(self.n_frames, 3, 224, 224)
            return dummy, dummy, torch.tensor(label, dtype=torch.float32)
        
        # JPEG compression augmentation — only on full frames, NOT face crops
        # SRM filters need clean pixels to detect manipulation artifacts
        if self.compress_prob > 0 and random.random() < self.compress_prob:
            full_frames = self._jpeg_compress(full_frames, quality_range=(60, 90))
            # face_crops intentionally NOT compressed — preserves SRM signals
        
        # NEW: Random downscale+upscale — breaks resolution bias
        # Simulates what happens when a high-res real video is viewed at low res
        if self.downscale_prob > 0 and random.random() < self.downscale_prob:
            target_size = random.randint(112, 180)  # Random low resolution
            down = torch.nn.functional.interpolate(full_frames, size=target_size, mode='bilinear', align_corners=False)
            full_frames = torch.nn.functional.interpolate(down, size=224, mode='bilinear', align_corners=False)
            down = torch.nn.functional.interpolate(face_crops, size=target_size, mode='bilinear', align_corners=False)
            face_crops = torch.nn.functional.interpolate(down, size=224, mode='bilinear', align_corners=False)
        
        # Augmentations (same random transform for both)
        if self.frame_augment is not None:
            seed = random.randint(0, 2**32 - 1)
            torch.manual_seed(seed)
            full_frames = self.frame_augment(full_frames)
            torch.manual_seed(seed)
            face_crops = self.frame_augment(face_crops)
            
            # Random erasing on full frames ONLY — blocks watermark shortcuts
            full_frames = self.frame_erasing(full_frames)
        
        # Normalize
        full_frames = self.normalize(full_frames)
        face_crops = self.normalize(face_crops)
        
        return full_frames, face_crops, torch.tensor(label, dtype=torch.float32)


# ─── Training Utilities ───

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_accuracy(logits, labels):
    preds = (torch.sigmoid(logits.squeeze()) > 0.5).float()
    return (preds == labels).float().mean().item()


def compute_auc(logits_list, labels_list):
    try:
        from sklearn.metrics import roc_auc_score
        scores = torch.sigmoid(torch.cat(logits_list)).cpu().numpy()
        labels = torch.cat(labels_list).cpu().numpy()
        if len(np.unique(labels)) < 2:
            return 0.0
        return roc_auc_score(labels, scores)
    except:
        return 0.0


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None

    def should_stop(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        if score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            print(f"  [Early Stop] No improvement for {self.patience} epochs. Stopping.")
            return True
        print(f"  [Early Stop] Patience: {self.counter}/{self.patience}")
        return False


class TrainingLogger:
    def __init__(self, save_dir, phase_name):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(save_dir, f"log_phase_{phase_name}.csv")
        self.rows = []

    def log(self, **kwargs):
        self.rows.append(kwargs)

    def save(self):
        headers = ["epoch", "train_loss", "train_acc", "val_auc", "val_acc", "lr", "elapsed_min"]
        with open(self.path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)
        print(f"  [Log] Saved to {self.path}")


def save_checkpoint(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  [Checkpoint] Saved -> {path}")


# ─── Phase 1: Train Face Branch ───

def train_branch_a(model, train_loader, val_loader, device, args):
    print("\n" + "=" * 60)
    print("  PHASE 1: Training Branch A (Face + SRM)")
    print("=" * 60)
    
    branch = model.face_branch
    criterion = get_criterion(label_smoothing=0.1)
    logger = TrainingLogger(args.save_dir, "1_face")
    stopper = EarlyStopping(patience=args.patience)
    start = time.time()
    
    for param in branch.backbone.parameters():
        param.requires_grad = False
    
    # Higher LR for face branch — this is our strongest branch
    optimizer = AdamW(filter(lambda p: p.requires_grad, branch.parameters()), lr=args.lr * 3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda") if device == "cuda" else None
    best_auc = 0.0
    
    for epoch in range(args.epochs):
        if epoch == 2:
            print("[Phase 1] Unfreezing backbone...")
            for param in branch.backbone.parameters():
                param.requires_grad = True
            # Drop LR by 10x to prevent catastrophic forgetting of ImageNet weights
            optimizer = AdamW(branch.parameters(), lr=args.lr * 0.1)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 2)
        
        branch.train()
        total_loss, total_acc, count = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"[A] Epoch {epoch+1}/{args.epochs}")
        for step, (full_frames, face_crops, labels) in enumerate(pbar):
            b, t = face_crops.shape[0], face_crops.shape[1]
            flat = face_crops.view(b * t, 3, 224, 224).to(device)
            frame_labels = labels.unsqueeze(1).expand(b, t).reshape(b * t).to(device)
            
            with autocast("cuda"):
                _, logits = branch(flat)
                loss = criterion(logits.squeeze(), smooth_labels(frame_labels)) / args.grad_accum
            
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(branch.parameters(), max_norm=1.0)  # NEW: gradient clipping
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            acc = compute_accuracy(logits, frame_labels)
            total_loss += loss.item() * args.grad_accum * b
            total_acc += acc * b
            count += b
            pbar.set_postfix(loss=f"{loss.item()*args.grad_accum:.4f}", acc=f"{acc:.3f}")
        
        scheduler.step()
        avg_loss = total_loss / max(count, 1)
        avg_acc = total_acc / max(count, 1)
        print(f"[A] Epoch {epoch+1} -- Loss: {avg_loss:.4f}, Acc: {avg_acc:.3f}")
        
        val_auc = 0.0
        if val_loader:
            val_auc = validate_branch(branch, val_loader, device, "A")
            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(model, os.path.join(args.save_dir, "best_branch_a.pth"))
        
        elapsed = (time.time() - start) / 60
        logger.log(epoch=epoch+1, train_loss=avg_loss, train_acc=avg_acc,
                   val_auc=val_auc, val_acc=0.0, lr=scheduler.get_last_lr()[0],
                   elapsed_min=f"{elapsed:.1f}")
        
        if val_loader and stopper.should_stop(val_auc):
            break
    
    logger.save()
    return model


# ─── Phase 2: Train Frame Branch ───

def train_branch_b(model, train_loader, val_loader, device, args):
    print("\n" + "=" * 60)
    print("  PHASE 2: Training Branch B (Full-Frame + Spectral)")
    print("=" * 60)
    
    branch = model.frame_branch
    criterion = get_criterion(label_smoothing=0.1)
    logger = TrainingLogger(args.save_dir, "2_frame")
    stopper = EarlyStopping(patience=7)  # More patience — spectral heads are slow to learn
    start = time.time()
    
    for param in branch.backbone.parameters():
        param.requires_grad = False
    
    # FIX: Match Phase 1's initial LR (3x) — spectral heads are randomly initialized
    # and need aggressive LR just like SRM heads in Phase 1
    optimizer = AdamW(filter(lambda p: p.requires_grad, branch.parameters()), lr=args.lr * 3)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda") if device == "cuda" else None
    best_auc = 0.0
    
    for epoch in range(args.epochs):
        if epoch == 3:  # FIX: 3 epochs instead of 2 — let spectral heads stabilize
            print("[Phase 2] Unfreezing backbone with discriminative LR...")
            for param in branch.backbone.parameters():
                param.requires_grad = True
            
            # FIX: Discriminative learning rates
            # Backbone: very low LR (pretrained, don't destroy ImageNet features)
            # Heads: keep high LR (CBAM, FFT, DCT, classifier still learning)
            backbone_params = list(branch.backbone.parameters())
            backbone_ids = {id(p) for p in backbone_params}
            head_params = [p for p in branch.parameters() if id(p) not in backbone_ids]
            
            optimizer = AdamW([
                {'params': backbone_params, 'lr': args.lr * 0.05},  # 1.5e-5 for backbone
                {'params': head_params, 'lr': args.lr},             # 3e-4 for heads
            ], weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - 3)
        
        branch.train()
        total_loss, total_acc, count = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"[B] Epoch {epoch+1}/{args.epochs}")
        for step, (full_frames, face_crops, labels) in enumerate(pbar):
            b, t = full_frames.shape[0], full_frames.shape[1]
            flat = full_frames.view(b * t, 3, 224, 224).to(device)
            frame_labels = labels.unsqueeze(1).expand(b, t).reshape(b * t).to(device)
            
            with autocast("cuda"):
                _, logits = branch(flat)
                loss = criterion(logits.squeeze(), smooth_labels(frame_labels)) / args.grad_accum
            
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(branch.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            acc = compute_accuracy(logits, frame_labels)
            total_loss += loss.item() * args.grad_accum * b
            total_acc += acc * b
            count += b
            pbar.set_postfix(loss=f"{loss.item()*args.grad_accum:.4f}", acc=f"{acc:.3f}")
        
        scheduler.step()
        avg_loss = total_loss / max(count, 1)
        avg_acc = total_acc / max(count, 1)
        print(f"[B] Epoch {epoch+1} -- Loss: {avg_loss:.4f}, Acc: {avg_acc:.3f}")
        
        val_auc = 0.0
        if val_loader:
            val_auc = validate_branch(branch, val_loader, device, "B")
            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(model, os.path.join(args.save_dir, "best_branch_b.pth"))
        
        elapsed = (time.time() - start) / 60
        logger.log(epoch=epoch+1, train_loss=avg_loss, train_acc=avg_acc,
                   val_auc=val_auc, val_acc=0.0, lr=scheduler.get_last_lr()[0],
                   elapsed_min=f"{elapsed:.1f}")
        
        if val_loader and stopper.should_stop(val_auc):
            break
    
    logger.save()
    return model


# ─── Phase 3: Train Temporal Branch (BiGRU) ───

def train_branch_c(model, train_loader, val_loader, device, args):
    print("\n" + "=" * 60)
    print("  PHASE 3: Training Branch C (Temporal GRU)")
    print("=" * 60)
    
    # Freeze branches A and B completely — C trains on their fixed outputs
    for param in model.face_branch.parameters():
        param.requires_grad = False
    for param in model.frame_branch.parameters():
        param.requires_grad = False
    
    branch_c = model.temporal_branch
    criterion = get_criterion(label_smoothing=0.1)
    logger = TrainingLogger(args.save_dir, "3_temporal")
    stopper = EarlyStopping(patience=7)  # More patience for temporal — GRU learns slower at first
    start = time.time()
    
    # Higher LR for randomly initialized GRU (no pretraining unlike ConvNeXt)
    optimizer = AdamW(branch_c.parameters(), lr=args.lr * 2)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.temporal_epochs)
    scaler = GradScaler("cuda") if device == "cuda" else None
    best_auc = 0.0
    
    for epoch in range(args.temporal_epochs):
        branch_c.train()
        model.face_branch.eval()
        model.frame_branch.eval()
        total_loss, total_acc, count = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"[C] Epoch {epoch+1}/{args.temporal_epochs}")
        for step, (full_frames, face_crops, labels) in enumerate(pbar):
            full_dev = full_frames.to(device)
            face_dev = face_crops.to(device)
            labels_dev = labels.to(device)
            b, t = full_dev.shape[0], full_dev.shape[1]
            
            # Generate features from frozen A+B (no gradients needed)
            with torch.no_grad():
                flat_faces = face_dev.view(b * t, 3, 224, 224)
                face_embeds = model.face_branch.get_embeddings(flat_faces)
                
                c_dim = full_dev.shape[2]
                flat_frames = full_dev.view(b * t, c_dim, full_dev.shape[3], full_dev.shape[4])
                frame_embeds = model.frame_branch.get_embeddings(flat_frames)
                
                combined = torch.cat([face_embeds, frame_embeds], dim=1)
                sequence_feats = combined.view(b, t, -1)  # [B, T, 2176]
            
            with autocast("cuda"):
                _, logits = branch_c(sequence_feats)
                loss = criterion(logits.squeeze(), smooth_labels(labels_dev)) / args.grad_accum
            
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(branch_c.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            acc = compute_accuracy(logits, labels_dev)
            total_loss += loss.item() * args.grad_accum * b
            total_acc += acc * b
            count += b
            pbar.set_postfix(loss=f"{loss.item()*args.grad_accum:.4f}", acc=f"{acc:.3f}")
        
        scheduler.step()
        avg_loss = total_loss / max(count, 1)
        avg_acc = total_acc / max(count, 1)
        print(f"[C] Epoch {epoch+1} -- Loss: {avg_loss:.4f}, Acc: {avg_acc:.3f}")
        
        val_auc = 0.0
        if val_loader:
            val_auc = validate_temporal(model, val_loader, device)
            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(model, os.path.join(args.save_dir, "best_branch_c.pth"))
        
        elapsed = (time.time() - start) / 60
        logger.log(epoch=epoch+1, train_loss=avg_loss, train_acc=avg_acc,
                   val_auc=val_auc, val_acc=0.0, lr=scheduler.get_last_lr()[0],
                   elapsed_min=f"{elapsed:.1f}")
        
        if val_loader and stopper.should_stop(val_auc):
            break
    
    logger.save()
    
    # Unfreeze branches A and B for E2E
    for param in model.face_branch.parameters():
        param.requires_grad = True
    for param in model.frame_branch.parameters():
        param.requires_grad = True
    
    return model, best_auc


def validate_temporal(model, val_loader, device):
    """Validate the temporal branch using frozen A+B features."""
    model.face_branch.eval()
    model.frame_branch.eval()
    model.temporal_branch.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for full_frames, face_crops, labels in val_loader:
            full_dev = full_frames.to(device)
            face_dev = face_crops.to(device)
            b, t = full_dev.shape[0], full_dev.shape[1]
            
            flat_faces = face_dev.view(b * t, 3, 224, 224)
            face_embeds = model.face_branch.get_embeddings(flat_faces)
            
            c_dim = full_dev.shape[2]
            flat_frames = full_dev.view(b * t, c_dim, full_dev.shape[3], full_dev.shape[4])
            frame_embeds = model.frame_branch.get_embeddings(flat_frames)
            
            combined = torch.cat([face_embeds, frame_embeds], dim=1)
            sequence_feats = combined.view(b, t, -1)
            
            _, logits = model.temporal_branch(sequence_feats)
            all_logits.append(logits.squeeze())
            all_labels.append(labels.to(device))
    
    auc = compute_auc(all_logits, all_labels)
    acc = compute_accuracy(torch.cat(all_logits), torch.cat(all_labels))
    print(f"  [Val C] AUC: {auc:.4f}, Acc: {acc:.3f}")
    return auc


# ─── Phase 4: End-to-End (3-branch or 2-branch fallback) ───

def train_end_to_end(model, train_loader, val_loader, device, args):
    n_branches = "2-Branch" if model.skip_temporal else "3-Branch"
    print("\n" + "=" * 60)
    print(f"  PHASE 4: End-to-End Fine-tuning ({n_branches} Gated Fusion)")
    print("=" * 60)
    
    criterion = get_criterion(label_smoothing=0.1)
    logger = TrainingLogger(args.save_dir, "4_e2e")
    stopper = EarlyStopping(patience=args.patience)
    start = time.time()
    
    # FIX: Discriminative learning rates for E2E
    # The fusion layers (gate_net + fusion_mlp) are RANDOMLY INITIALIZED —
    # they were never trained in Phases 1-3. They need a HIGH LR.
    # The branch backbones are well-trained and need a LOW LR.
    branch_params = (
        list(model.face_branch.parameters()) +
        list(model.frame_branch.parameters()) +
        list(model.temporal_branch.parameters())
    )
    branch_ids = {id(p) for p in branch_params}
    fusion_params = [p for p in model.parameters() if id(p) not in branch_ids]
    
    n_branch = sum(p.numel() for p in branch_params)
    n_fusion = sum(p.numel() for p in fusion_params)
    print(f"  [LR] Branch params: {n_branch:,} @ lr={args.lr * 0.03:.1e}")
    print(f"  [LR] Fusion params: {n_fusion:,} @ lr={args.lr:.1e}")
    
    optimizer = AdamW([
        {'params': branch_params, 'lr': args.lr * 0.03},  # 9e-6: very gentle fine-tuning
        {'params': fusion_params, 'lr': args.lr},          # 3e-4: aggressive for random fusion layers
    ], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.e2e_epochs)
    scaler = GradScaler("cuda") if device == "cuda" else None
    best_auc = 0.0
    
    for epoch in range(args.e2e_epochs):
        model.train()
        total_loss, total_acc, count = 0, 0, 0
        
        pbar = tqdm(train_loader, desc=f"[E2E] Epoch {epoch+1}/{args.e2e_epochs}")
        for step, (full_frames, face_crops, labels) in enumerate(pbar):
            full_dev = full_frames.to(device)
            face_dev = face_crops.to(device)
            labels_dev = labels.to(device)
            
            with autocast("cuda"):
                final_logit, face_l, frame_l, temp_l = model(full_dev, face_dev)
                smoothed = smooth_labels(labels_dev)
                loss_main = criterion(final_logit.squeeze(), smoothed)
                loss_face = criterion(face_l.squeeze(), smoothed)
                loss_frame = criterion(frame_l.squeeze(), smoothed)
                # Weight face branch MORE (it's our strongest branch)
                if model.skip_temporal:
                    loss = (loss_main + 0.5 * loss_face + 0.1 * loss_frame) / args.grad_accum
                else:
                    loss_temporal = criterion(temp_l.squeeze(), smoothed)
                    loss = (loss_main + 0.5 * loss_face + 0.1 * loss_frame + 0.1 * loss_temporal) / args.grad_accum
            
            scaler.scale(loss).backward()
            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            acc = compute_accuracy(final_logit, labels_dev)
            total_loss += loss.item() * args.grad_accum * full_frames.shape[0]
            total_acc += acc * full_frames.shape[0]
            count += full_frames.shape[0]
            pbar.set_postfix(loss=f"{loss.item()*args.grad_accum:.4f}", acc=f"{acc:.3f}")
        
        scheduler.step()
        avg_loss = total_loss / max(count, 1)
        avg_acc = total_acc / max(count, 1)
        print(f"[E2E] Epoch {epoch+1} -- Loss: {avg_loss:.4f}, Acc: {avg_acc:.3f}")
        
        val_auc = 0.0
        if val_loader:
            val_auc = validate_full(model, val_loader, device)
            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(model, os.path.join(args.save_dir, "best_model.pth"))
                print(f"  * New best AUC: {val_auc:.4f}")
        
        elapsed = (time.time() - start) / 60
        logger.log(epoch=epoch+1, train_loss=avg_loss, train_acc=avg_acc,
                   val_auc=val_auc, val_acc=0.0, lr=scheduler.get_last_lr()[0],
                   elapsed_min=f"{elapsed:.1f}")
        
        if val_loader and stopper.should_stop(val_auc):
            break
    
    logger.save()
    return model


# ─── Validation ───

def validate_branch(branch, val_loader, device, name):
    branch.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for full_frames, face_crops, labels in val_loader:
            b, t = full_frames.shape[0], full_frames.shape[1]
            if name == "A":
                flat = face_crops.view(b * t, 3, 224, 224).to(device)
            else:
                flat = full_frames.view(b * t, 3, 224, 224).to(device)
            frame_labels = labels.unsqueeze(1).expand(b, t).reshape(b * t).to(device)
            _, logits = branch(flat)
            all_logits.append(logits.squeeze())
            all_labels.append(frame_labels)
    
    auc = compute_auc(all_logits, all_labels)
    acc = compute_accuracy(torch.cat(all_logits), torch.cat(all_labels))
    print(f"  [Val {name}] AUC: {auc:.4f}, Acc: {acc:.3f}")
    return auc


def validate_full(model, val_loader, device):
    model.eval()
    all_logits, all_labels = [], []
    
    with torch.no_grad():
        for full_frames, face_crops, labels in val_loader:
            full_dev = full_frames.to(device)
            face_dev = face_crops.to(device)
            labels_dev = labels.to(device)
            final_logit, _, _, _ = model(full_dev, face_dev)
            all_logits.append(final_logit.squeeze())
            all_labels.append(labels_dev)
    
    auc = compute_auc(all_logits, all_labels)
    acc = compute_accuracy(torch.cat(all_logits), torch.cat(all_labels))
    print(f"  [Val FUSED] AUC: {auc:.4f}, Acc: {acc:.3f}")
    return auc


# ─── Main ───

def main():
    p = argparse.ArgumentParser(description="Fast Deepfake Training v6 (cached data)")
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--val_dir", type=str, default=None)
    p.add_argument("--epochs", type=int, default=15,
                   help="Epochs for face branch (most important)")
    p.add_argument("--frame_epochs", type=int, default=15,
                   help="Epochs for frame branch (needs more than face)")
    p.add_argument("--temporal_epochs", type=int, default=10,
                   help="Epochs for temporal GRU branch")
    p.add_argument("--e2e_epochs", type=int, default=10,
                   help="Epochs for end-to-end fusion")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--n_frames", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--save_dir", type=str, default="weights")
    p.add_argument("--resume", type=str, default=None,
                   help="Resume from checkpoint (e.g. weights/best_model.pth)")
    p.add_argument("--e2e_only", action="store_true",
                   help="Skip Phase 1-3, only run E2E fusion (use with --resume)")
    p.add_argument("--skip_phase1", action="store_true",
                   help="Skip Phase 1 (use with --resume weights/best_branch_a.pth)")
    p.add_argument("--skip_phase3", action="store_true",
                   help="Skip Phase 3 training (keep resumed temporal weights)")
    p.add_argument("--skip_temporal", action="store_true",
                   help="Skip Phase 3 temporal and force 2-branch fusion")
    args = p.parse_args()
    
    device = get_device()
    print(f"[+] Device: {device}")
    print(f"[+] Using CACHED data (no face detection at runtime)")
    print(f"[+] LR: {args.lr}")
    print(f"[+] n_frames: {args.n_frames}, batch_size: {args.batch_size}, grad_accum: {args.grad_accum}")
    if args.e2e_only:
        print(f"[+] E2E ONLY mode — skipping Phase 1-3")
        print(f"[+] Epochs: e2e={args.e2e_epochs}")
    else:
        print(f"[+] Epochs: face={args.epochs}, frame={args.frame_epochs}, temporal={args.temporal_epochs}, e2e={args.e2e_epochs}")
    
    # Build cached dataloaders
    train_ds = CachedVideoDataset(args.data_dir, n_frames=args.n_frames, augment=True)
    val_ds = CachedVideoDataset(args.val_dir, n_frames=args.n_frames, augment=False) if args.val_dir else None
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=2, pin_memory=True) if val_ds else None
    
    # Initialize model
    model = FusedAIVideoDetector(pretrained=True, use_srm=True, use_spectral=True)
    
    if args.resume and os.path.exists(args.resume):
        print(f"[+] Resuming from {args.resume}")
        state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
    
    model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Model parameters: {total_params:,}")
    
    if not args.e2e_only:
        # Phase 1: Face branch (most important — gets most epochs)
        if args.skip_phase1:
            print("\n[+] --skip_phase1: Skipping Phase 1 (using resumed weights)")
        else:
            model = train_branch_a(model, train_loader, val_loader, device, args)
        
        # Phase 2: Frame branch
        saved_epochs = args.epochs
        args.epochs = args.frame_epochs
        model = train_branch_b(model, train_loader, val_loader, device, args)
        args.epochs = saved_epochs
        
        # Phase 3: Temporal branch (with <60% AUC fallback)
        if args.skip_temporal:
            print("\n[+] --skip_temporal flag set, skipping Phase 3 (2-branch fusion)")
            model.skip_temporal = True
        elif args.skip_phase3:
            print("\n[+] --skip_phase3: Keeping resumed temporal weights (no retraining)")
        else:
            model, temporal_best_auc = train_branch_c(model, train_loader, val_loader, device, args)
            
            if temporal_best_auc < 0.60:
                print(f"\n  {'='*60}")
                print(f"  [TEMPORAL ABORT] Best AUC {temporal_best_auc:.4f} < 0.60")
                print(f"  [TEMPORAL ABORT] Falling back to 2-branch fusion (A+B only)")
                print(f"  {'='*60}")
                model.skip_temporal = True
            else:
                print(f"\n  [TEMPORAL OK] Best AUC {temporal_best_auc:.4f} >= 0.60 — keeping 3-branch fusion")
    
    # Phase 4: E2E fusion
    model = train_end_to_end(model, train_loader, val_loader, device, args)
    
    # Save final
    save_checkpoint(model, os.path.join(args.save_dir, "final_model.pth"))
    
    # Calibrate
    if val_loader:
        print("\n" + "=" * 60)
        print("  POST-TRAINING: Calibrating scores")
        print("=" * 60)
        from utils.calibrate import TemperatureScaler
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for full_frames, face_crops, labels in val_loader:
                full_dev = full_frames.to(device)
                face_dev = face_crops.to(device)
                final_logit, _, _, _ = model(full_dev, face_dev)
                all_logits.append(final_logit.squeeze())
                all_labels.append(labels)
        logits_cat = torch.cat(all_logits).cpu()
        labels_cat = torch.cat(all_labels).cpu()
        temp_scaler = TemperatureScaler()
        temp_scaler.fit(logits_cat, labels_cat)
        temp_scaler.save(os.path.join(args.save_dir, "calibration.json"))
    
    print("\n[+] Training complete!")
    print(f"    Weights: {args.save_dir}/best_model.pth")
    if model.skip_temporal:
        print(f"    Mode: 2-branch fusion (temporal skipped)")
    else:
        print(f"    Mode: 3-branch fusion (temporal active)")
    print(f"    Run: python detect.py --video your_video.mp4")


if __name__ == "__main__":
    main()
