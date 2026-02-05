import argparse, os, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from models_ablation import GRU_GAP, GRU_Attn, LSTM_Attn


class NpzDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def train_epoch(model, loader, opt, crit, device):
    model.train()
    tot, cor, loss_sum = 0, 0, 0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        opt.zero_grad()
        logits,_ = model(x)
        loss = crit(logits,y)
        loss.backward()
        opt.step()
        loss_sum += loss.item()*x.size(0)
        cor += (logits.argmax(1)==y).sum().item()
        tot += x.size(0)
    return loss_sum/tot, cor/tot


@torch.no_grad()
def eval_epoch(model, loader, crit, device):
    model.eval()
    tot, cor, loss_sum = 0, 0, 0
    yt, yp = [], []
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits,_ = model(x)
        loss = crit(logits,y)
        loss_sum += loss.item()*x.size(0)
        pred = logits.argmax(1)
        cor += (pred==y).sum().item()
        tot += x.size(0)
        yt.append(y.cpu().numpy())
        yp.append(pred.cpu().numpy())
    return loss_sum/tot, cor/tot, np.concatenate(yt), np.concatenate(yp)


def get_model(name):
    if name == "gru_gap": return GRU_GAP()
    if name == "gru_attn": return GRU_Attn()
    if name == "lstm_attn": return LSTM_Attn()
    raise ValueError("unknown model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--model", required=True,
                    choices=["gru_gap","gru_attn","lstm_attn"])
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data = np.load(args.npz)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = NpzDataset(data["X_train"], data["y_train"])
    val_ds   = NpzDataset(data["X_val"],   data["y_val"])
    test_ds  = NpzDataset(data["X_test"],  data["y_test"])

    train_ld = DataLoader(train_ds, 256, shuffle=True)
    val_ld   = DataLoader(val_ds,   256)
    test_ld  = DataLoader(test_ds,  256)

    model = get_model(args.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    best, bad = -1, 0
    for ep in range(1, 21):
        tl, ta = train_epoch(model, train_ld, opt, crit, device)
        vl, va, _, _ = eval_epoch(model, val_ld, crit, device)
        print(f"[{args.model} E{ep:02d}] tr_acc={ta:.3f} va_acc={va:.3f}")
        if va > best:
            best = va
            bad = 0
            torch.save(model.state_dict(), os.path.join(args.outdir,"best.pt"))
        else:
            bad += 1
            if bad >= 5: break

    model.load_state_dict(torch.load(os.path.join(args.outdir,"best.pt")))
    tl, ta, yt, yp = eval_epoch(model, test_ld, crit, device)

    report = classification_report(
        yt, yp, labels=[0,1,2,3],
        target_names=["normal","loose","arc","overcurrent"],
        output_dict=True, zero_division=0
    )

    cm = confusion_matrix(yt, yp, labels=[0,1,2,3])

    with open(os.path.join(args.outdir,"metrics.json"),"w") as f:
        json.dump({
            "model": args.model,
            "test_acc": ta,
            "report": report,
            "cm": cm.tolist()
        }, f, indent=2)

    print(f"[DONE] {args.model} test_acc={ta:.4f}")


if __name__ == "__main__":
    main()
