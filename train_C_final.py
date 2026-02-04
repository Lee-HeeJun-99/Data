import os, argparse, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

class NpzDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
    def forward(self, h):
        score = self.v(torch.tanh(self.proj(h)))   # (B,T,1)
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

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    tot_loss=0; corr=0; tot=0
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        optimizer.zero_grad()
        logits,_ = model(x)
        loss = criterion(logits,y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        tot_loss += loss.item()*x.size(0)
        corr += (logits.argmax(1)==y).sum().item()
        tot += x.size(0)
    return tot_loss/tot, corr/tot

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    tot_loss=0; corr=0; tot=0
    y_true=[]; y_pred=[]
    for x,y in loader:
        x=x.to(device); y=y.to(device)
        logits,_ = model(x)
        loss = criterion(logits,y)
        tot_loss += loss.item()*x.size(0)
        pred = logits.argmax(1)
        corr += (pred==y).sum().item()
        tot += x.size(0)
        y_true.append(y.cpu().numpy()); y_pred.append(pred.cpu().numpy())
    y_true=np.concatenate(y_true); y_pred=np.concatenate(y_pred)
    return tot_loss/tot, corr/tot, y_true, y_pred

def plot_curves(hist, outdir):
    plt.figure(); plt.plot(hist["tr_loss"], label="train"); plt.plot(hist["va_loss"], label="val")
    plt.title("Loss Curve"); plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"loss_curve.png"), dpi=200); plt.close()

    plt.figure(); plt.plot(hist["tr_acc"], label="train"); plt.plot(hist["va_acc"], label="val")
    plt.title("Accuracy Curve"); plt.xlabel("epoch"); plt.ylabel("acc"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir,"acc_curve.png"), dpi=200); plt.close()

def plot_cm(cm, names, outpath, normalize=False, title="Confusion Matrix"):
    cm_disp = cm.astype(np.float32)
    if normalize:
        cm_disp = cm_disp / (cm_disp.sum(axis=1, keepdims=True)+1e-8)
    plt.figure()
    plt.imshow(cm_disp, interpolation="nearest")
    plt.title(title); plt.colorbar()
    ticks=np.arange(len(names))
    plt.xticks(ticks, names, rotation=30, ha="right"); plt.yticks(ticks, names)
    fmt=".2f" if normalize else "d"
    thr=cm_disp.max()*0.6
    for i in range(4):
        for j in range(4):
            plt.text(j,i,format(cm_disp[i,j],fmt),ha="center",va="center",
                     color="white" if cm_disp[i,j]>thr else "black")
    plt.ylabel("True"); plt.xlabel("Pred"); plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--outdir", type=str, default="outputs_C")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=5)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    data = np.load(args.npz)

    train_ds = NpzDataset(data["X_train"], data["y_train"])
    val_ds   = NpzDataset(data["X_val"],   data["y_val"])
    test_ds  = NpzDataset(data["X_test"],  data["y_test"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    class_names = ["normal","loose","arc","overcurrent"]

    model = GRUAttnClassifier(3, 128, 4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_val = -1.0
    bad = 0
    hist={"tr_loss":[], "tr_acc":[], "va_loss":[], "va_acc":[]}

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    best_path = os.path.join(args.outdir, "best_model.pt")

    for ep in range(1, args.epochs+1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = eval_epoch(model, val_loader, criterion, device)

        hist["tr_loss"].append(tr_loss); hist["tr_acc"].append(tr_acc)
        hist["va_loss"].append(va_loss); hist["va_acc"].append(va_acc)

        print(f"[E{ep:02d}] train loss={tr_loss:.4f} acc={tr_acc:.3f} | val loss={va_loss:.4f} acc={va_acc:.3f}")

        if va_acc > best_val:
            best_val = va_acc
            bad = 0
            torch.save({"model": model.state_dict(), "best_val": best_val}, best_path)
        else:
            bad += 1
            if bad >= args.patience:
                print(f"[INFO] Early stopping. best_val_acc={best_val:.3f}")
                break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    te_loss, te_acc, y_true, y_pred = eval_epoch(model, test_loader, criterion, device)
    print(f"\n[Test] loss={te_loss:.4f} acc={te_acc:.3f}\n")

    # reports
    report = classification_report(
        y_true, y_pred, labels=[0,1,2,3],
        target_names=class_names, zero_division=0, digits=4, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])

    with open(os.path.join(args.outdir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({
            "npz": args.npz,
            "test_loss": float(te_loss),
            "test_acc": float(te_acc),
            "report": report,
            "confusion_matrix": cm.tolist()
        }, f, ensure_ascii=False, indent=2)

    # figures
    plot_curves(hist, args.outdir)
    plot_cm(cm, class_names, os.path.join(args.outdir,"cm_counts.png"), normalize=False,
            title="Confusion Matrix (Counts)")
    plot_cm(cm, class_names, os.path.join(args.outdir,"cm_norm.png"), normalize=True,
            title="Confusion Matrix (Row-normalized)")

    print(f"[SAVED] {args.outdir}")

if __name__ == "__main__":
    main()
