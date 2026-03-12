import torch
import torch.nn as nn


class GRUClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, num_classes=4, dropout=0.0):
        super().__init__()

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, T, F]
        h, _ = self.gru(x)          # [B, T, H]
        z = h[:, -1, :]             # [B, H]
        logits = self.fc(z)         # [B, C]
        return logits


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, num_classes=4, dropout=0.0):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)       # [B, T, H]
        z = out[:, -1, :]           # [B, H]
        logits = self.fc(z)         # [B, C]
        return logits


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):
        # h: [B, T, H]
        score = self.v(torch.tanh(self.proj(h)))   # [B, T, 1]
        alpha = torch.softmax(score, dim=1)        # [B, T, 1]
        ctx = torch.sum(h * alpha, dim=1)          # [B, H]
        return ctx, alpha


class GRUAttentionClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, num_classes=4, dropout=0.0):
        super().__init__()

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout
        )
        self.attn = TemporalAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: [B, T, F]
        h, _ = self.gru(x)          # [B, T, H]
        ctx, alpha = self.attn(h)   # [B, H], [B, T, 1]
        logits = self.fc(ctx)       # [B, C]
        return logits, alpha
    
class HybridGRUClassifier(nn.Module):
    def __init__(
        self,
        seq_input_dim=3,
        feat_input_dim=9,
        hidden_dim=64,
        num_layers=2,
        num_classes=4,
        dropout=0.0
    ):
        super().__init__()

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout
        )

        self.mlp_feat = nn.Sequential(
            nn.Linear(feat_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + 32, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_seq, x_feat):
        seq_out, _ = self.gru(x_seq)      # [B,T,H]
        seq_out = seq_out[:, -1, :]       # [B,H]
        feat_out = self.mlp_feat(x_feat)  # [B,32]
        fused = torch.cat([seq_out, feat_out], dim=1)
        logits = self.classifier(fused)
        return logits
    
class PhysicsGRUClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, num_classes=4, dropout=0.0):
        super().__init__()

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout
        )

        self.cls_head = nn.Linear(hidden_dim, num_classes)
        self.thermal_head = nn.Linear(hidden_dim, 1)  # dT/dt 예측

    def forward(self, x):
        h, _ = self.gru(x)          # [B,T,H]
        z = h[:, -1, :]             # [B,H]
        logits = self.cls_head(z)   # [B,C]
        dT_pred = self.thermal_head(z)  # [B,1]
        return logits, dT_pred