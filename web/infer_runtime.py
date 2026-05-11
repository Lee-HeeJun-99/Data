import pickle
from collections import deque
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

from models import HybridGRUClassifier


CLASS_NAMES = {
    0: "normal",
    1: "loose_early",
    2: "loose_severe",
    3: "arc_intermittent",
    4: "arc_sustained",
    5: "overcurrent_mild",
    6: "overcurrent_severe",
    7: "arc_overcurrent",
}


class HybridGRURuntimeInferencer:
    def __init__(
        self,
        model_path: str,
        seq_scaler_path: str,
        feat_scaler_path: str,
        window: int = 64,
        stride: int = 16,
        seq_input_dim: int = 7,
        feat_input_dim: int = 28,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_classes: int = 8,
        debug: bool = False,
    ):
        self.window = window
        self.stride = stride
        self.seq_input_dim = seq_input_dim
        self.feat_input_dim = feat_input_dim
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
        return np.array([
            sensor["I_meas"],
            sensor["HF_energy"],
            sensor["HF_rms"],
            sensor["hf_norm"],
            sensor["arc_flag"],
            sensor["arc_detect"],
            sensor["T_meas"],
        ], dtype=np.float32)

    def _make_window_feat(self, x_seq: np.ndarray) -> np.ndarray:
        I = x_seq[:, 0]
        HF_energy = x_seq[:, 1]
        HF_rms = x_seq[:, 2]
        hf_norm = x_seq[:, 3]
        arc_flag = x_seq[:, 4]
        arc_detect = x_seq[:, 5]
        T = x_seq[:, 6]

        dI = np.diff(I, prepend=I[0])
        dHF = np.diff(HF_energy, prepend=HF_energy[0])
        dN = np.diff(hf_norm, prepend=hf_norm[0])
        dT = np.diff(T, prepend=T[0])

        feats = [
            I.mean(),
            I.std(),
            I.min(),
            I.max(),
            np.sqrt(np.mean(I ** 2)),
            dI.mean(),
            dI.max(),

            HF_energy.mean(),
            HF_energy.std(),
            HF_energy.min(),
            HF_energy.max(),
            dHF.mean(),
            dHF.max(),

            HF_rms.mean(),
            HF_rms.std(),
            HF_rms.max(),

            hf_norm.mean(),
            hf_norm.std(),
            hf_norm.max(),
            dN.mean(),
            dN.max(),

            arc_flag.mean(),
            arc_detect.mean(),

            T.mean(),
            T.std(),
            T.max(),
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

        x_seq = np.array(self.buffer, dtype=np.float32)
        x_feat = self._make_window_feat(x_seq)

        x_seq_scaled = self.seq_scaler.transform(x_seq)
        x_feat_scaled = self.feat_scaler.transform([x_feat])[0]

        x_seq_scaled = np.expand_dims(x_seq_scaled, axis=0)
        x_feat_scaled = np.expand_dims(x_feat_scaled, axis=0)

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
                CLASS_NAMES[i]: float(probs_np[i])
                for i in range(len(CLASS_NAMES))
            }
        }

        self.last_pred = result
        return result