import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


# ----------------------
# Dataset
# ----------------------
class NpzDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ----------------------
# Temporal Attention
# ----------------------
class TemporalAttention(nn.Module):
    """
    h: (B, T, H)
    returns:
      context: (B, H)
      alpha:   (B, T)  attention weights over time
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):
        score = self.v(torch.tanh(self.proj(h)))  # (B, T, 1)
        alpha = torch.softmax(score, dim=1).squeeze(-1)  # (B, T)
        context = torch.sum(h * alpha.unsqueeze(-1), dim=1)  # (B, H)
        return context, alpha


# ----------------------
# Model (B-stage)
# ----------------------
class GRUAttnClassifier(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, num_classes=4):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.attn = TemporalAttention(hidden_dim)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        h, _ = self.gru(x)               # (B, T, H)
        ctx, alpha = self.attn(h)        # (B, H), (B, T)
        logits = self.fc(ctx)            # (B, K)
        return logits, alpha


# ----------------------
# Utils
# ----------------------
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def move_to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)

@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true_all, y_pred_all = [], []

    for batch in loader:
        x, y = move_to_device(batch, device)
        logits, _ = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

        y_true_all.append(y.detach().cpu().numpy())
        y_pred_all.append(pred.detach().cpu().numpy())

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    y_true_all = np.concatenate(y_true_all) if len(y_true_all) else np.array([])
    y_pred_all = np.concatenate(y_pred_all) if len(y_pred_all) else np.array([])
    return avg_loss, acc, y_true_all, y_pred_all

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch in loader:
        x, y = move_to_device(batch, device)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)

def plot_curves(history, save_dir=None):
    # Loss
    plt.figure()
    plt.plot(history["tr_loss"], label="train")
    plt.plot(history["va_loss"], label="val")
    plt.title("Loss Curve")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200)
    plt.show()

    # Accuracy
    plt.figure()
    plt.plot(history["tr_acc"], label="train")
    plt.plot(history["va_acc"], label="val")
    plt.title("Accuracy Curve")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "acc_curve.png"), dpi=200)
    plt.show()

def plot_confusion_matrix(cm, class_names, normalize=False, title="Confusion Matrix", save_path=None):
    cm_disp = cm.astype(np.float32)
    if normalize:
        cm_disp = cm_disp / (cm_disp.sum(axis=1, keepdims=True) + 1e-8)

    plt.figure()
    plt.imshow(cm_disp, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=30, ha="right")
    plt.yticks(ticks, class_names)
    fmt = ".2f" if normalize else "d"
    thresh = cm_disp.max() * 0.6

    for i in range(cm_disp.shape[0]):
        for j in range(cm_disp.shape[1]):
            plt.text(j, i, format(cm_disp[i, j], fmt),
                     ha="center", va="center",
                     color="white" if cm_disp[i, j] > thresh else "black")

    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200)
    plt.show()

@torch.no_grad()
def attention_explain(model, dataset, device, class_names, save_dir=None, num_per_class=1):
    """
    Pick a few samples per class from test set, plot signals + attention alpha(t)
    Input channels: [I, HF, T] (C=3)
    """
    model.eval()

    # collect indices by class
    y_np = dataset.y.cpu().numpy()
    idx_by_c = {c: np.where(y_np == c)[0].tolist() for c in range(4)}

    chosen = []
    for c in range(4):
        take = idx_by_c[c][:num_per_class]
        chosen += [(c, i) for i in take]

    if len(chosen) == 0:
        print("[WARN] No samples found for attention visualization.")
        return

    for c, idx in chosen:
        x, y = dataset[idx]
        x_in = x.unsqueeze(0).to(device)  # (1,T,C)
        logits, alpha = model(x_in)
        pred = logits.argmax(dim=1).item()
        alpha = alpha.squeeze(0).detach().cpu().numpy()  # (T,)
        x_np = x.detach().cpu().numpy()  # (T,C)

        T = x_np.shape[0]
        t = np.arange(T)

        # Plot: signals + attention
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        ax1.plot(t, x_np[:, 0], label="I(t)")
        ax1.plot(t, x_np[:, 1], label="HF(t)")
        ax1.plot(t, x_np[:, 2], label="T(t)")
        ax1.set_xlabel("time index (window)")
        ax1.set_ylabel("signal (normalized)")
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(t, alpha, label="attention α(t)")
        ax2.set_ylabel("attention weight")

        title = f"True={class_names[y.item()]} | Pred={class_names[pred]}"
        plt.title(title)
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            fname = f"attn_true_{y.item()}_pred_{pred}_idx_{idx}.png"
            plt.savefig(os.path.join(save_dir, fname), dpi=200)

        plt.show()


# ----------------------
# Main
# ----------------------
def main():
    set_seed(42)

    save_dir = "outputs_B"
    os.makedirs(save_dir, exist_ok=True)

    data = np.load("train_dataset.npz")
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]

    train_ds = NpzDataset(X_train, y_train)
    val_ds   = NpzDataset(X_val,   y_val)
    test_ds  = NpzDataset(X_test,  y_test)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] device:", device)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=256, shuffle=False)

    model = GRUAttnClassifier(input_dim=3, hidden_dim=128, num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Early stopping
    best_val_acc = -1.0
    best_path = os.path.join(save_dir, "best_model.pt")
    patience = 5
    bad = 0

    history = {"tr_loss": [], "tr_acc": [], "va_loss": [], "va_acc": []}

    EPOCHS = 20
    for ep in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc, _, _ = eval_epoch(model, val_loader, criterion, device)

        history["tr_loss"].append(tr_loss)
        history["tr_acc"].append(tr_acc)
        history["va_loss"].append(va_loss)
        history["va_acc"].append(va_acc)

        print(f"[E{ep:02d}] train loss={tr_loss:.4f} acc={tr_acc:.3f} | val loss={va_loss:.4f} acc={va_acc:.3f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            bad = 0
            torch.save({"model": model.state_dict()}, best_path)
        else:
            bad += 1
            if bad >= patience:
                print(f"[INFO] Early stopping triggered. best_val_acc={best_val_acc:.3f}")
                break

    # Load best
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    # Test
    te_loss, te_acc, y_true, y_pred = eval_epoch(model, test_loader, criterion, device)
    print(f"\n[Test] loss={te_loss:.4f} acc={te_acc:.3f}\n")

    class_names = ["normal", "loose", "arc", "overcurrent"]

    # Always force labels=[0,1,2,3] to avoid the error you hit
    print("[Classification Report]")
    print(classification_report(
        y_true, y_pred,
        labels=[0, 1, 2, 3],
        target_names=class_names,
        zero_division=0,
        digits=4
    ))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print("[Confusion Matrix]\n", cm)

    # Curves
    plot_curves(history, save_dir=save_dir)

    # Confusion matrix heatmap (counts + normalized)
    plot_confusion_matrix(cm, class_names, normalize=False,
                          title="Confusion Matrix (Counts)",
                          save_path=os.path.join(save_dir, "cm_counts.png"))
    plot_confusion_matrix(cm, class_names, normalize=True,
                          title="Confusion Matrix (Row-normalized)",
                          save_path=os.path.join(save_dir, "cm_norm.png"))

    # Attention explain: pick 1 sample per class from test set
    attention_explain(model, test_ds, device, class_names, save_dir=save_dir, num_per_class=1)

    print(f"[INFO] Saved figures/models to: {save_dir}")


if __name__ == "__main__":
    main()
