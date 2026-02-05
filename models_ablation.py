import torch
import torch.nn as nn

# ---------
# Attention
# ---------
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):  # (B,T,H)
        score = self.v(torch.tanh(self.proj(h)))   # (B,T,1)
        alpha = torch.softmax(score, dim=1).squeeze(-1)
        ctx = torch.sum(h * alpha.unsqueeze(-1), dim=1)
        return ctx, alpha


# ---------
# A) GRU + GAP
# ---------
class GRU_GAP(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_classes=4):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        h, _ = self.rnn(x)
        h = h.mean(dim=1)          # Global Avg Pooling
        out = self.fc(h)
        return out, None


# ---------
# B) GRU + Attention
# ---------
class GRU_Attn(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_classes=4):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn = TemporalAttention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        h, _ = self.rnn(x)
        ctx, alpha = self.attn(h)
        out = self.fc(ctx)
        return out, alpha


# ---------
# C) LSTM + Attention
# ---------
class LSTM_Attn(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_classes=4):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.attn = TemporalAttention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        h, _ = self.rnn(x)
        ctx, alpha = self.attn(h)
        out = self.fc(ctx)
        return out, alpha
