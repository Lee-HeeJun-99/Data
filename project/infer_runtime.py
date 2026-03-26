import pickle
from collections import deque
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from models import HybridGRUClassifier


CLASS_NAMES = {
    0: "normal",
    1: "loose",
    2: "arc",
    3: "overcurrent",
}


class HybridGRURuntimeInferencer:
    def __init__(
        self,
        model_path: str,
        seq_scaler_path: str,
        feat_scaler_path: str,
        window: int = 64,
        stride: int = 16,
        seq_input_dim: int = 4,
        feat_input_dim: int = 11,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        num_classes: int = 4,
    ):
        self.window = window
        self.stride = stride
        self.seq_input_dim = seq_input_dim
        self.feat_input_dim = feat_input_dim
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        with open(seq_scaler_path, "rb") as f:
            self.seq_scaler = pickle.load(f)

        with open(feat_scaler_path, "rb") as f:
            self.feat_scaler = pickle.load(f)

        self.model = HybridGRUClassifier(
            seq_input_dim=seq_input_dim,
            feat_input_dim=feat_input_dim,
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

    def _make_seq_feature(self, sensor: Dict[str, Any]) -> np.ndarray:
        """
        sequence 입력 4개
        [I_meas, HF_energy, HF_rms, T_meas]
        """
        return np.array([
            sensor["I_meas"],
            sensor["HF_energy"],
            sensor["HF_rms"],
            sensor["T_meas"],
        ], dtype=np.float32)

    def _make_window_feat(self, x_seq: np.ndarray) -> np.ndarray:
        """
        x_seq shape: [T, 4]

        일단 체크포인트 차원(11)에 맞추기 위한 feature 구성:
        - mean(4)
        - std(4)
        - max(2) : I_meas, HF_energy
        - 마지막 T_meas(1)

        총 11개
        """
        mean_feat = np.mean(x_seq, axis=0)      # 4
        std_feat = np.std(x_seq, axis=0)        # 4
        max_i = np.max(x_seq[:, 0])             # 1
        max_hf_energy = np.max(x_seq[:, 1])     # 1
        last_t = x_seq[-1, 3]                   # 1

        feat = np.concatenate([
            mean_feat,
            std_feat,
            np.array([max_i, max_hf_energy, last_t], dtype=np.float32)
        ], axis=0)  # 총 11개

        return feat.astype(np.float32)

    def step(self, sensor: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        seq_feature = self._make_seq_feature(sensor)

        self.buffer.append(seq_feature)
        self.step_count += 1

        if len(self.buffer) < self.window:
            return None

        if (self.step_count - self.window) % self.stride != 0:
            return self.last_pred

        x_seq = np.array(self.buffer, dtype=np.float32)          # [T, 4]
        x_feat = self._make_window_feat(x_seq)                   # [11]

        x_seq_scaled = self.seq_scaler.transform(x_seq)          # [T, 4]
        x_feat_scaled = self.feat_scaler.transform([x_feat])[0]  # [11]

        x_seq_scaled = np.expand_dims(x_seq_scaled, axis=0)      # [1, T, 4]
        x_feat_scaled = np.expand_dims(x_feat_scaled, axis=0)    # [1, 11]

        x_seq_tensor = torch.from_numpy(x_seq_scaled).float().to(self.device)
        x_feat_tensor = torch.from_numpy(x_feat_scaled).float().to(self.device)

        with torch.no_grad():
            logits = self.model(x_seq_tensor, x_feat_tensor)
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
