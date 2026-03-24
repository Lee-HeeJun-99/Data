import pickle
from collections import deque
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from models import GRUClassifier


CLASS_NAMES = {
    0: "normal",
    1: "loose",
    2: "arc",
    3: "overcurrent",
}


class GRURuntimeInferencer:
    def __init__(
        self,
        model_path: str,
        scaler_path: str,
        window: int = 128,
        stride: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        num_classes: int = 4,
    ):
        self.window = window
        self.stride = stride
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        self.model = GRUClassifier(
            input_dim=4,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout,
        ).to(self.device)

        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.buffer = deque(maxlen=window)
        self.step_count = 0
        self.last_pred = None

    def reset(self):
        self.buffer.clear()
        self.step_count = 0
        self.last_pred = None

    def step(self, sensor: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        feature = np.array([
            sensor["I_meas"],
            sensor["HF_energy"],
            sensor["HF_rms"],
            sensor["T_meas"],
        ], dtype=np.float32)

        self.buffer.append(feature)
        self.step_count += 1

        if len(self.buffer) < self.window:
            return None

        if (self.step_count - self.window) % self.stride != 0:
            return self.last_pred

        x = np.array(self.buffer, dtype=np.float32)
        x = self.scaler.transform(x)
        x = np.expand_dims(x, axis=0)  # [1, T, F]

        x_tensor = torch.from_numpy(x).float().to(self.device)

        with torch.no_grad():
            logits = self.model(x_tensor)
            probs = F.softmax(logits, dim=1)[0]
            conf, pred = torch.max(probs, dim=0)

        pred = int(pred.item())
        conf = float(conf.item())
        probs_np = probs.detach().cpu().numpy()

        result = {
            "pred_mode": pred,
            "pred_mode_name": CLASS_NAMES[pred],
            "confidence": conf,
            "probabilities": {
                "normal": float(probs_np[0]),
                "loose": float(probs_np[1]),
                "arc": float(probs_np[2]),
                "overcurrent": float(probs_np[3]),
            }
        }

        self.last_pred = result
        return result