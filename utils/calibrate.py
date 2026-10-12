"""
Post-hoc calibration for deepfake detection scores.

Raw sigmoid outputs from neural nets are overconfident — a 0.92 doesn't mean
92% probability. Temperature scaling and Platt scaling fix this so the output
can be interpreted as an actual risk probability.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os


class TemperatureScaler(nn.Module):
    """Learned temperature scaling for calibrating neural net confidence.

    After training, run `fit()` on validation logits + labels to learn
    a single temperature parameter that softens/sharpens the sigmoid output.
    """

    def __init__(self):
        super().__init__()
        # Initialize temperature to 1.0 (no change)
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        """Apply temperature scaling to raw logits.

        Args:
            logits: raw model output BEFORE sigmoid, shape [N, 1] or [N]

        Returns:
            calibrated probabilities in [0, 1]
        """
        scaled = logits / self.temperature
        return torch.sigmoid(scaled)

    def fit(self, logits, labels, lr=0.01, max_iter=100):
        """Learn the optimal temperature on a validation set.

        Args:
            logits: [N] or [N, 1] raw logits (before sigmoid)
            labels: [N] ground truth binary labels (0 or 1)
            lr: learning rate for temperature optimization
            max_iter: max optimization steps
        """
        logits = logits.detach().float()
        labels = labels.detach().float()
        if logits.dim() == 2:
            logits = logits.squeeze(1)
        if labels.dim() == 2:
            labels = labels.squeeze(1)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            scaled_logits = logits / self.temperature
            loss = criterion(scaled_logits, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"[Calibration] Learned temperature: {self.temperature.item():.4f}")

    def save(self, path):
        """Save calibration parameters to JSON."""
        data = {"temperature": self.temperature.item()}
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[Calibration] Saved to {path}")

    def load(self, path):
        """Load calibration parameters from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        self.temperature.data = torch.tensor([data["temperature"]])
        print(f"[Calibration] Loaded temperature={data['temperature']:.4f} from {path}")


def calibrate_scores(raw_logits, calibration_path=None):
    """Convenience function: apply saved calibration to raw logits.

    Args:
        raw_logits: [N, 1] or scalar tensor — raw model output before sigmoid
        calibration_path: path to saved calibration JSON. If None or missing,
                          falls back to plain sigmoid.

    Returns:
        calibrated probabilities in [0, 1]
    """
    scaler = TemperatureScaler()

    if calibration_path and os.path.exists(calibration_path):
        scaler.load(calibration_path)
    else:
        # No calibration available — just use raw sigmoid
        pass

    with torch.no_grad():
        return scaler(raw_logits)
