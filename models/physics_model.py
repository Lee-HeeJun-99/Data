import torch
import torch.nn as nn

class PhysicsGRUClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_classes=4):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.thermal_head = nn.Linear(hidden_dim, 1)  # dT/dt 근사 출력

    def forward(self, x):
        h, _ = self.gru(x)
        z = h[:, -1, :]
        logits = self.cls_head(z)
        dT_pred = self.thermal_head(z)
        return logits, dT_pred