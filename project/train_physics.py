import os
import json
import random
import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models import PhysicsGRUClassifier


# =========================================================
# Dataset
# =========================================================
class PhysicsSeqDataset(Dataset):
    """
    X_seq에서:
      - 입력 x_seq: [T, 3] = [I, HF, T]
      - 분류 라벨 y
      - 물리 타깃 dT/dt (마지막 시점 근사값)
    를 함께 반환
    """
    def __init__(self, X_seq, y):
        self.X_seq = torch.from_numpy(X_seq).float()
        self.y = torch.from_numpy(y).long()

        # 온도 채널: x_seq[:, :, 2]
        # dT/dt 근사: T[t] - T[t-1]
        T = X_seq[:, :, 2]  # [N, T]
        dT = np.diff(T, axis=1, prepend=T[:, :1])  # [N, T]
        dT_last = dT[:, -1].astype(np.float32)     # [N]
        self.dT = torch.from_numpy(dT_last).float().unsqueeze(1)  # [N,1]

    def __len__(self):
        return len(self.X_seq)

    def __getitem__(self, idx):
        return self.X_seq[idx], self.y[idx], self.dT[idx]


# =========================================================
# Utility
# =========================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_seq_data(data_root: str):
    X_seq_train = np.load(os.path.join(data_root, "X_seq_train.npy"))
    X_seq_val = np.load(os.path.join(data_root, "X_seq_val.npy"))
    X_seq_test = np.load(os.path.join(data_root, "X_seq_test.npy"))

    y_train = np.load(os.path.join(data_root, "y_train.npy"))
    y_val = np.load(os.path.join(data_root, "y_val.npy"))
    y_test = np.load(os.path.join(data_root, "y_test.npy"))

    return X_seq_train, X_seq_val, X_seq_test, y_train, y_val, y_test


# =========================================================
# Train / Eval
# =========================================================
def train_one_epoch(model, loader, cls_criterion, phys_criterion, optimizer, device, lambda_phys):
    model.train()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_phys_loss = 0.0

    all_preds = []
    all_targets = []

    for x_seq, y, dT_true in loader:
        x_seq = x_seq.to(device)
        y = y.to(device)
        dT_true = dT_true.to(device)

        optimizer.zero_grad()

        logits, dT_pred = model(x_seq)

        loss_cls = cls_criterion(logits, y)
        loss_phys = phys_criterion(dT_pred, dT_true)
        loss = loss_cls + lambda_phys * loss_phys

        loss.backward()
        optimizer.step()

        batch_size = x_seq.size(0)
        total_loss += loss.item() * batch_size
        total_cls_loss += loss_cls.item() * batch_size
        total_phys_loss += loss_phys.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    avg_cls_loss = total_cls_loss / len(loader.dataset)
    avg_phys_loss = total_phys_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    acc = accuracy_score(all_targets, all_preds)

    return avg_loss, avg_cls_loss, avg_phys_loss, acc


@torch.no_grad()
def eval_one_epoch(model, loader, cls_criterion, phys_criterion, device, lambda_phys):
    model.eval()

    total_loss = 0.0
    total_cls_loss = 0.0
    total_phys_loss = 0.0

    all_preds = []
    all_targets = []

    for x_seq, y, dT_true in loader:
        x_seq = x_seq.to(device)
        y = y.to(device)
        dT_true = dT_true.to(device)

        logits, dT_pred = model(x_seq)

        loss_cls = cls_criterion(logits, y)
        loss_phys = phys_criterion(dT_pred, dT_true)
        loss = loss_cls + lambda_phys * loss_phys

        batch_size = x_seq.size(0)
        total_loss += loss.item() * batch_size
        total_cls_loss += loss_cls.item() * batch_size
        total_phys_loss += loss_phys.item() * batch_size

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    avg_cls_loss = total_cls_loss / len(loader.dataset)
    avg_phys_loss = total_phys_loss / len(loader.dataset)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    acc = accuracy_score(all_targets, all_preds)

    return avg_loss, avg_cls_loss, avg_phys_loss, acc, all_targets, all_preds


# =========================================================
# Save / Visualization
# =========================================================
def save_learning_curve(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history["train_loss"], label="train_total_loss")
    plt.plot(epochs, history["val_loss"], label="val_total_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Total Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history["train_cls_loss"], label="train_cls_loss")
    plt.plot(epochs, history["val_cls_loss"], label="val_cls_loss")
    plt.plot(epochs, history["train_phys_loss"], label="train_phys_loss")
    plt.plot(epochs, history["val_phys_loss"], label="val_phys_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Cls / Phys Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_confusion_matrix_heatmap(cm, class_names, save_path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_classification_report_text(y_true, y_pred, class_names, save_path):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)


def save_metrics_json(metrics, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True,
                        help="예: ./prepared_dataset_multi/win_128")
    parser.add_argument("--save_root", type=str, default="./results/physics",
                        help="결과 저장 루트")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--num_classes", type=int, default=4)

    parser.add_argument("--lambda_phys", type=float, default=0.1,
                        help="physics loss 가중치")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    # -----------------------------------------------------
    # data load
    # -----------------------------------------------------
    X_seq_train, X_seq_val, X_seq_test, y_train, y_val, y_test = load_seq_data(args.data_root)

    print(f"[INFO] X_seq_train: {X_seq_train.shape}, y_train: {y_train.shape}")
    print(f"[INFO] X_seq_val  : {X_seq_val.shape}, y_val  : {y_val.shape}")
    print(f"[INFO] X_seq_test : {X_seq_test.shape}, y_test : {y_test.shape}")

    input_dim = X_seq_train.shape[-1]

    train_dataset = PhysicsSeqDataset(X_seq_train, y_train)
    val_dataset = PhysicsSeqDataset(X_seq_val, y_val)
    test_dataset = PhysicsSeqDataset(X_seq_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    # -----------------------------------------------------
    # save dir
    # -----------------------------------------------------
    data_name = os.path.basename(os.path.normpath(args.data_root))
    exp_name = f"physics_gru_h{args.hidden_dim}_l{args.num_layers}_lam{args.lambda_phys}"
    save_dir = os.path.join(args.save_root, data_name, exp_name)
    ensure_dir(save_dir)

    # -----------------------------------------------------
    # model
    # -----------------------------------------------------
    model = PhysicsGRUClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=args.num_classes,
        dropout=args.dropout
    ).to(device)

    cls_criterion = nn.CrossEntropyLoss()
    phys_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(model)

    # -----------------------------------------------------
    # train
    # -----------------------------------------------------
    history = {
        "train_loss": [],
        "train_cls_loss": [],
        "train_phys_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_cls_loss": [],
        "val_phys_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_model_path = os.path.join(save_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_cls_loss, train_phys_loss, train_acc = train_one_epoch(
            model, train_loader, cls_criterion, phys_criterion, optimizer, device, args.lambda_phys
        )

        val_loss, val_cls_loss, val_phys_loss, val_acc, _, _ = eval_one_epoch(
            model, val_loader, cls_criterion, phys_criterion, device, args.lambda_phys
        )

        history["train_loss"].append(train_loss)
        history["train_cls_loss"].append(train_cls_loss)
        history["train_phys_loss"].append(train_phys_loss)
        history["train_acc"].append(train_acc)

        history["val_loss"].append(val_loss)
        history["val_cls_loss"].append(val_cls_loss)
        history["val_phys_loss"].append(val_phys_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        print(
            f"[Epoch {epoch:03d}/{args.epochs:03d}] "
            f"train_total={train_loss:.4f} | train_cls={train_cls_loss:.4f} | train_phys={train_phys_loss:.4f} | train_acc={train_acc:.4f} || "
            f"val_total={val_loss:.4f} | val_cls={val_cls_loss:.4f} | val_phys={val_phys_loss:.4f} | val_acc={val_acc:.4f}"
        )

    # -----------------------------------------------------
    # best model load
    # -----------------------------------------------------
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    test_loss, test_cls_loss, test_phys_loss, test_acc, y_true, y_pred = eval_one_epoch(
        model, test_loader, cls_criterion, phys_criterion, device, args.lambda_phys
    )

    print(
        f"[TEST] total={test_loss:.4f}, cls={test_cls_loss:.4f}, "
        f"phys={test_phys_loss:.4f}, acc={test_acc:.4f}"
    )

    # -----------------------------------------------------
    # metrics / reports
    # -----------------------------------------------------
    class_names_default = ["normal", "loose", "arc", "overcurrent"]
    if args.num_classes == 4:
        class_names = class_names_default
    else:
        class_names = [f"class_{i}" for i in range(args.num_classes)]

    cm = confusion_matrix(y_true, y_pred)

    save_learning_curve(history, os.path.join(save_dir, "learning_curve.png"))
    save_confusion_matrix_heatmap(cm, class_names, os.path.join(save_dir, "confusion_matrix.png"))
    save_classification_report_text(
        y_true, y_pred, class_names,
        os.path.join(save_dir, "classification_report.txt")
    )

    metrics = {
        "model": "physics_gru",
        "data_root": args.data_root,
        "input_dim": int(input_dim),
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "num_classes": int(args.num_classes),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "lambda_phys": float(args.lambda_phys),
        "best_val_acc": float(best_val_acc),
        "test_total_loss": float(test_loss),
        "test_cls_loss": float(test_cls_loss),
        "test_phys_loss": float(test_phys_loss),
        "test_acc": float(test_acc),
    }
    save_metrics_json(metrics, os.path.join(save_dir, "metrics.json"))

    print(f"[SAVE] results saved to: {save_dir}")


if __name__ == "__main__":
    main()