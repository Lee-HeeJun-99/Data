import argparse, os, json
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

# ---- same model as C ----
class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, h):
        score = self.v(torch.tanh(self.proj(h)))
        alpha = torch.softmax(score, dim=1).squeeze(-1)
        ctx = torch.sum(h * alpha.unsqueeze(-1), dim=1)
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
        ctx, _ = self.attn(h)
        logits = self.fc(ctx)
        return logits

@torch.no_grad()
def eval_model(model, X, y, device):
    model.eval()
    bs = 512
    preds = []
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i+bs], dtype=torch.float32).to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().numpy()
        preds.append(pred)
    y_pred = np.concatenate(preds)
    acc = (y_pred == y).mean()
    rep = classification_report(
        y, y_pred, labels=[0,1,2,3],
        target_names=["normal","loose","arc","overcurrent"],
        zero_division=0, output_dict=True, digits=4
    )
    return float(acc), rep

# ---- perturbations ----
def add_noise(X, sigma_rel=0.05, seed=0):
    rng = np.random.default_rng(seed)
    Xn = X.copy()
    # 채널별 표준편차 기준 상대 노이즈
    for c in range(X.shape[2]):
        sd = X[:,:,c].std() + 1e-8
        Xn[:,:,c] = X[:,:,c] + rng.normal(0, sigma_rel*sd, size=X[:,:,c].shape)
    return Xn

def quantize(X, levels=1024):
    # 입력이 zscore라서 [-inf,inf] 가능 → 실사용에서는 클리핑이 발생.
    # 단순 현실 근사: [-3,3]로 클리핑 후 양자화
    Xq = np.clip(X, -3.0, 3.0)
    step = 6.0 / (levels - 1)
    Xq = np.round((Xq + 3.0) / step) * step - 3.0
    return Xq

def downsample_time(X, factor=2):
    # 시간축 다운샘플: T -> T/factor, 다시 원래 길이로 nearest upsample(현실 근사)
    if factor <= 1:
        return X
    Xd = X[:, ::factor, :]
    # upsample back to original length
    T = X.shape[1]
    idx = np.minimum((np.arange(T) / factor).astype(int), Xd.shape[1]-1)
    Xu = Xd[:, idx, :]
    return Xu

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default="robustness.json")
    args = ap.parse_args()

    data = np.load(args.npz)
    X = data["X_test"].astype(np.float32)
    y = data["y_test"].astype(np.int64)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GRUAttnClassifier(3,128,4).to(device)
    ckpt = torch.load(args.ckpt, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)

    results = {}

    # baseline
    acc0, rep0 = eval_model(model, X, y, device)
    results["baseline"] = {"acc": acc0, "f1_arc": rep0["arc"]["f1-score"]}

    # noise
    for s in [0.02, 0.05, 0.10]:
        Xn = add_noise(X, sigma_rel=s, seed=0)
        acc, rep = eval_model(model, Xn, y, device)
        results[f"noise_rel_{s}"] = {"acc": acc, "f1_arc": rep["arc"]["f1-score"]}

    # quantization (10-bit ~ 1024, 12-bit ~ 4096)
    for lv in [1024, 4096]:
        Xq = quantize(X, levels=lv)
        acc, rep = eval_model(model, Xq, y, device)
        results[f"quant_{lv}"] = {"acc": acc, "f1_arc": rep["arc"]["f1-score"]}

    # downsample
    for f in [2, 4]:
        Xd = downsample_time(X, factor=f)
        acc, rep = eval_model(model, Xd, y, device)
        results[f"downsample_x{f}"] = {"acc": acc, "f1_arc": rep["arc"]["f1-score"]}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("[DONE] saved:", args.out)
    for k,v in results.items():
        print(f"{k:>14}: acc={v['acc']:.4f}, f1_arc={v['f1_arc']:.4f}")

if __name__ == "__main__":
    main()
