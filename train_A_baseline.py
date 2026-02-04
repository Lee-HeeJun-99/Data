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
# Model (A-stage baseline)
# ----------------------
class GRUBaseline(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, num_classes=4):
        super().__init__()
        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (B, T, C)
        h, _ = self.gru(x)            # (B, T, H)
        h_mean = h.mean(dim=1)        # Global Avg Pooling
        out = self.fc(h_mean)
        return out

# ----------------------
# Train / Eval
# ----------------------
def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, total = 0, 0, 0
    all_pred, all_gt = [], []

    for x, y in loader:
        if is_train:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

        all_pred.append(pred.cpu().numpy())
        all_gt.append(y.cpu().numpy())

    return (
        total_loss / total,
        correct / total,
        np.concatenate(all_gt),
        np.concatenate(all_pred)
    )

# ----------------------
# Main
# ----------------------
data = np.load("train_dataset.npz")

train_ds = NpzDataset(data["X_train"], data["y_train"])
val_ds   = NpzDataset(data["X_val"],   data["y_val"])
test_ds  = NpzDataset(data["X_test"],  data["y_test"])

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=256)
test_loader  = DataLoader(test_ds,  batch_size=256)

model = GRUBaseline()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

history = {"tr_loss":[], "va_loss":[], "tr_acc":[], "va_acc":[]}

EPOCHS = 10

for ep in range(1, EPOCHS+1):
    tr_loss, tr_acc, _, _ = run_epoch(model, train_loader, optimizer)
    va_loss, va_acc, _, _ = run_epoch(model, val_loader)

    history["tr_loss"].append(tr_loss)
    history["va_loss"].append(va_loss)
    history["tr_acc"].append(tr_acc)
    history["va_acc"].append(va_acc)

    print(f"[E{ep:02d}] "
          f"train loss={tr_loss:.4f} acc={tr_acc:.3f} | "
          f"val loss={va_loss:.4f} acc={va_acc:.3f}")

# ----------------------
# Test + Reports
# ----------------------
_, test_acc, y_true, y_pred = run_epoch(model, test_loader)
print(f"\n[Test Accuracy] {test_acc:.3f}\n")

print("[Classification Report]")
print(classification_report(
    y_true, y_pred,
    target_names=["normal","loose","arc","overcurrent"]
))

cm = confusion_matrix(y_true, y_pred)
print("[Confusion Matrix]\n", cm)

# ----------------------
# Curves
# ----------------------
plt.figure()
plt.plot(history["tr_loss"], label="train")
plt.plot(history["va_loss"], label="val")
plt.title("Loss Curve")
plt.legend()
plt.show()

plt.figure()
plt.plot(history["tr_acc"], label="train")
plt.plot(history["va_acc"], label="val")
plt.title("Accuracy Curve")
plt.legend()
plt.show()
