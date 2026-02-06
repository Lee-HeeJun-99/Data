import argparse, time, os
import numpy as np
import torch
import torch.nn as nn

# ----------------------
# Model (same as C-stage)
# ----------------------
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):
        score = self.v(torch.tanh(self.proj(h)))  # (B,T,1)
        alpha = torch.softmax(score, dim=1).squeeze(-1)  # (B,T)
        ctx = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # (B,H)
        return ctx, alpha

class GRUAttnClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_classes=4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn = TemporalAttention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, num_classes)
        )

    def forward(self, x):
        h, _ = self.gru(x)
        ctx, alpha = self.attn(h)
        logits = self.fc(ctx)
        return logits, alpha

# ----------------------
# Streaming buffer
# ----------------------
class RingBuffer:
    def __init__(self, win_len, channels):
        self.win_len = win_len
        self.channels = channels
        self.buf = np.zeros((win_len, channels), dtype=np.float32)
        self.filled = 0

    def push(self, sample):  # sample: (C,)
        self.buf[:-1] = self.buf[1:]
        self.buf[-1] = sample
        self.filled = min(self.win_len, self.filled + 1)

    def ready(self):
        return self.filled >= self.win_len

    def window(self):
        return self.buf.copy()  # (T,C)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="train_seed*.npz")
    ap.add_argument("--ckpt", required=True, help="best_model.pt from C seed folder")
    ap.add_argument("--win_len", type=int, default=256)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--measure", type=int, default=500)
    args = ap.parse_args()

    data = np.load(args.npz)
    X = data["X_test"]  # (N,T,C) - already normalized
    # 스트리밍처럼 만들기 위해 테스트 윈도우들에서 "시간 샘플"을 이어붙인다고 가정(근사)
    # 실제 센서에서는 1샘플씩 들어오며 버퍼가 채워짐.
    # 여기서는 간단히 X_test에서 첫 샘플의 시계열을 반복 사용(지연측정 목적이므로 충분)
    stream_seq = X[0]  # (T,C)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRUAttnClassifier(3, 128, 4).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    # ckpt 형태가 {"model": state_dict} 또는 state_dict 단독일 수 있음
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()

    rb = RingBuffer(args.win_len, 3)

    # stride마다 추론한다는 가정(실제 운영과 동일)
    stride = args.stride
    step = 0

    def infer_once():
        w = rb.window()
        x = torch.tensor(w, dtype=torch.float32).unsqueeze(0).to(device)  # (1,T,C)
        with torch.no_grad():
            logits, _ = model(x)
            prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return prob

    # Warmup
    for _ in range(args.warmup):
        for i in range(args.win_len):
            rb.push(stream_seq[i])
        infer_once()

    # Measure
    times = []
    inferences = 0
    for _ in range(args.measure):
        # 스트리밍 입력 stride 샘플만큼 들어온다고 가정
        for i in range(stride):
            rb.push(stream_seq[(step + i) % args.win_len])
        step += stride

        if rb.ready():
            t0 = time.perf_counter()
            _ = infer_once()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000.0)  # ms
            inferences += 1

    times = np.array(times, dtype=np.float64)
    print(f"[INFO] device: {device}")
    print(f"[INFO] inferences: {inferences}")
    print(f"[LATENCY] mean={times.mean():.3f} ms | p95={np.percentile(times,95):.3f} ms | max={times.max():.3f} ms")
    print(f"[THROUGHPUT] {inferences / (times.sum()/1000.0):.2f} windows/sec")

if __name__ == "__main__":
    main()
