import torch
import torch.nn as nn

class HybridGRUClassifier(nn.Module):
    def __init__(self, seq_input_dim=3, feat_input_dim=9, hidden_dim=64, num_classes=4):
        super().__init__()

        self.gru = nn.GRU(
            input_size=seq_input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
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
        seq_out, _ = self.gru(x_seq)
        seq_out = seq_out[:, -1, :]
        feat_out = self.mlp_feat(x_feat)
        fused = torch.cat([seq_out, feat_out], dim=1)
        return self.classifier(fused)