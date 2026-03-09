import torch
import torch.nn as nn

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):
        score = self.v(torch.tanh(self.proj(h)))   # [B,T,1]
        alpha = torch.softmax(score, dim=1)        # [B,T,1]
        ctx = torch.sum(h * alpha, dim=1)          # [B,H]
        return ctx, alpha

class GRUAttentionClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_layers=2, num_classes=3):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.attn = TemporalAttention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h, _ = self.gru(x)
        ctx, alpha = self.attn(h)
        out = self.fc(ctx)
        return out, alpha