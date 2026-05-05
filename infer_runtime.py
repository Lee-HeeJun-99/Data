import pickle
from collections import deque
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from models import HybridGRUClassifier


CLASS_NAMES = {
    0: "M0",
    1: "M1",
    2: "M2",
    3: "M3",
    4: "M4",
    5: "M5",
    6: "M6",
    7: "M7",
}

# 모드별 알람 여부 (M0 = 정상, 나머지는 이상)
ALARM_MODES = {0: False, 1: True, 2: True, 3: True, 4: True, 5: True, 6: True, 7: True}


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
        num_classes: int = 8,
        debug: bool = False,
    ):
        self.window = window
        self.stride = stride
        self.seq_input_dim = seq_input_dim
        self.feat_input_dim = feat_input_dim
        self.num_classes = num_classes
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.debug = debug

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
        학습 코드와 동일한 engineered feature 생성

        x_seq shape: [T, 4]
        columns: [I_meas, HF_energy, HF_rms, T_meas]

        return shape: [11]
        """
        I = x_seq[:, 0]
        HF_energy = x_seq[:, 1]
        HF_rms = x_seq[:, 2]
        T = x_seq[:, 3]

        dT = np.diff(T, prepend=T[0])

        feats = [
            I.mean(),
            I.std(),
            np.sqrt(np.mean(I ** 2)),   # I RMS
            HF_energy.mean(),
            HF_energy.std(),
            HF_energy.max(),
            HF_rms.mean(),
            HF_rms.std(),
            T.mean(),
            dT.mean(),
            dT.max(),
        ]
        feat = np.array(feats, dtype=np.float32)

        if feat.shape[0] != self.feat_input_dim:
            raise ValueError(
                f"Feature dimension mismatch: got {feat.shape[0]}, "
                f"expected {self.feat_input_dim}"
            )

        return feat

    def step(self, sensor: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        seq_feature = self._make_seq_feature(sensor)

        self.buffer.append(seq_feature)
        self.step_count += 1

        if len(self.buffer) < self.window:
            return None

        if (self.step_count - self.window) % self.stride != 0:
            return self.last_pred

        x_seq = np.array(self.buffer, dtype=np.float32)   # [T, 4]
        x_feat = self._make_window_feat(x_seq)            # [11]

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

        if self.debug:
            print("=" * 80)
            print("DEBUG RAW seq last row:", x_seq[-1])
            print("DEBUG RAW feat:", x_feat)
            print("DEBUG SCALED seq min/max:", float(x_seq_scaled.min()), float(x_seq_scaled.max()))
            print("DEBUG SCALED feat:", x_feat_scaled[0])
            print("DEBUG LOGITS:", logits.detach().cpu().numpy()[0])
            print("DEBUG PROBS:", probs_np, "pred:", pred)

        # num_classes에 맞게 동적으로 확률 딕셔너리 생성
        probabilities = {
            CLASS_NAMES[i]: float(probs_np[i])
            for i in range(self.num_classes)
        }

        result = {
            "pred_mode": pred,
            "pred_mode_name": CLASS_NAMES.get(pred, f"M{pred}"),
            "confidence": conf,
            "probabilities": probabilities,
            "is_alarm": ALARM_MODES.get(pred, True),
        }

        self.last_pred = result
        return result
