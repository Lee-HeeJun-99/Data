import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.gru_model import GRUClassifier
from utils.datasets import SeqDataset
from utils.metrics import save_classification_report, save_confusion_matrix, save_curves

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "./results/gru_baseline"
os.makedirs(SAVE_DIR, exist_ok=True)

# 데이터 로드
X_train = np.load("./prepared_dataset/X_seq_train.npy")
X_val   = np.load("./prepared_dataset/X_seq_val.npy")
X_test  = np.load("./prepared_dataset/X_seq_test.npy")

y_train = np.load("./prepared_dataset/y_train.npy")
y_val   = np.load("./prepared_dataset/y_val.npy")
y_test  = np.load("./prepared_dataset/y_test.npy")

train_loader = DataLoader(SeqDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader   = DataLoader(SeqDataset(X_val, y_val), batch_size=64, shuffle=False)
test_loader  = DataLoader(SeqDataset(X_test, y_test), batch_size=64, shuffle=False)

model = GRUClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_acc_hist, val_acc_hist = [], []
train_loss_hist, val_loss_hist = [], []

for epoch in range(20):
    model.train()
    total_loss, total_correct, total_num = 0, 0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = logits.argmax(dim=1)
        total_correct += (pred == y).sum().item()
        total_num += y.size(0)

    train_loss = total_loss / len(train_loader)
    train_acc = total_correct / total_num

    model.eval()
    total_loss, total_correct, total_num = 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            total_correct += (pred == y).sum().item()
            total_num += y.size(0)

    val_loss = total_loss / len(val_loader)
    val_acc = total_correct / total_num

    train_loss_hist.append(train_loss)
    train_acc_hist.append(train_acc)
    val_loss_hist.append(val_loss)
    val_acc_hist.append(val_acc)

    print(f"Epoch {epoch+1:02d} | train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

# test
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(dim=1).cpu().numpy()
        y_pred.extend(pred.tolist())
        y_true.extend(y.numpy().tolist())

save_classification_report(y_true, y_pred, os.path.join(SAVE_DIR, "classification_report.txt"))
save_confusion_matrix(y_true, y_pred, os.path.join(SAVE_DIR, "confusion_matrix.png"))
save_curves(train_acc_hist, val_acc_hist, train_loss_hist, val_loss_hist, SAVE_DIR)