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
from torch.utils.data import DataLoader

from datasets import SeqDataset
from models import GRUClassifier, LSTMClassifier, GRUAttentionClassifier


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


def build_model(model_name: str, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int, dropout: float):
    model_name = model_name.lower()

    if model_name == "gru":
        model = GRUClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
    elif model_name == "lstm":
        model = LSTMClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
    elif model_name == "gru_attn":
        model = GRUAttentionClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
    else:
        raise ValueError(f"지원하지 않는 모델입니다: {model_name}")

    return model


def get_logits_from_model_output(output):
    if isinstance(output, tuple):
        return output[0]
    return output


# =========================================================
# Label transform
# =========================================================
def make_stage1_labels(y):
    """
    original:
      0: normal
      1: loose
      2: arc
      3: overcurrent

    stage1:
      0: normal
      1: abnormal
    """
    return (y != 0).astype(np.int64)


def filter_abnormal_only(X_seq, y):
    """
    abnormal(1,2,3)만 남기고
    stage2 label로 0,1,2로 변환
      1 -> 0 (loose)
      2 -> 1 (arc)
      3 -> 2 (overcurrent)
    """
    mask = (y != 0)
    X_seq_ab = X_seq[mask]
    y_ab = y[mask] - 1
    return X_seq_ab, y_ab


def restore_stage2_to_original(stage2_pred):
    """
    stage2 pred: 0,1,2
    restore: 1,2,3
    """
    return stage2_pred + 1


# =========================================================
# Train / Eval
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for x_seq, y in loader:
        x_seq = x_seq.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(x_seq)
        logits = get_logits_from_model_output(output)

        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x_seq.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    acc = accuracy_score(all_targets, all_preds)

    return avg_loss, acc


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for x_seq, y in loader:
        x_seq = x_seq.to(device)
        y = y.to(device)

        output = model(x_seq)
        logits = get_logits_from_model_output(output)

        loss = criterion(logits, y)
        total_loss += loss.item() * x_seq.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(y.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    acc = accuracy_score(all_targets, all_preds)

    return avg_loss, acc, all_targets, all_preds


@torch.no_grad()
def predict_labels(model, loader, device):
    model.eval()

    all_preds = []
    all_targets = []

    for x_seq, y in loader:
        x_seq = x_seq.to(device)

        output = model(x_seq)
        logits = get_logits_from_model_output(output)

        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(y.numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    return all_targets, all_preds


# =========================================================
# Save / Visualization
# =========================================================
def save_learning_curve(history, save_path):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
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
# Generic stage trainer
# =========================================================
def run_stage_training(
    stage_name,
    model,
    train_loader,
    val_loader,
    test_loader,
    device,
    epochs,
    lr,
    save_dir,
    class_names
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_acc = -1.0
    best_model_path = os.path.join(save_dir, f"{stage_name}_best_model.pt")

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = eval_one_epoch(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)

        print(
            f"[{stage_name.upper()}][Epoch {epoch:03d}/{epochs:03d}] "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    test_loss, test_acc, y_true, y_pred = eval_one_epoch(model, test_loader, criterion, device)

    cm = confusion_matrix(y_true, y_pred)

    save_learning_curve(history, os.path.join(save_dir, f"{stage_name}_learning_curve.png"))
    save_confusion_matrix_heatmap(cm, class_names, os.path.join(save_dir, f"{stage_name}_confusion_matrix.png"))
    save_classification_report_text(
        y_true, y_pred, class_names,
        os.path.join(save_dir, f"{stage_name}_classification_report.txt")
    )

    metrics = {
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
    }

    return model, metrics, y_true, y_pred


# =========================================================
# Final 2-stage inference
# =========================================================
@torch.no_grad()
def run_final_two_stage_inference(stage1_model, stage2_model, test_loader_stage1, device):
    """
    최종 예측 규칙:
    - stage1 pred == 0 -> final pred = 0
    - stage1 pred == 1 -> stage2 pred를 수행해서 final pred = stage2 + 1
    """
    stage1_model.eval()
    stage2_model.eval()

    final_preds = []
    final_targets = []

    for x_seq, y_true in test_loader_stage1:
        x_seq = x_seq.to(device)

        out1 = stage1_model(x_seq)
        logits1 = get_logits_from_model_output(out1)
        pred1 = torch.argmax(logits1, dim=1).cpu().numpy()

        batch_size = len(pred1)
        batch_final = np.zeros(batch_size, dtype=np.int64)

        abnormal_idx = np.where(pred1 == 1)[0]

        if len(abnormal_idx) > 0:
            x_ab = x_seq[abnormal_idx]

            out2 = stage2_model(x_ab)
            logits2 = get_logits_from_model_output(out2)
            pred2 = torch.argmax(logits2, dim=1).cpu().numpy()   # 0,1,2
            pred2_restored = restore_stage2_to_original(pred2)   # 1,2,3

            batch_final[abnormal_idx] = pred2_restored

        final_preds.append(batch_final)
        final_targets.append(y_true.numpy())

    final_preds = np.concatenate(final_preds)
    final_targets = np.concatenate(final_targets)
    return final_targets, final_preds


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", type=str, required=True,
                        help="예: ./prepared_dataset_multi/win_128")
    parser.add_argument("--save_root", type=str, default="./results/stage2",
                        help="결과 저장 루트")

    parser.add_argument("--stage1_model", type=str, default="gru",
                        choices=["gru", "lstm", "gru_attn"])
    parser.add_argument("--stage2_model", type=str, default="gru",
                        choices=["gru", "lstm", "gru_attn"])

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device: {device}")

    # -----------------------------------------------------
    # Load original data
    # -----------------------------------------------------
    X_seq_train, X_seq_val, X_seq_test, y_train, y_val, y_test = load_seq_data(args.data_root)

    print(f"[INFO] Original train: {X_seq_train.shape}, {y_train.shape}")
    print(f"[INFO] Original val  : {X_seq_val.shape}, {y_val.shape}")
    print(f"[INFO] Original test : {X_seq_test.shape}, {y_test.shape}")

    input_dim = X_seq_train.shape[-1]

    # -----------------------------------------------------
    # Stage 1 data (normal vs abnormal)
    # -----------------------------------------------------
    y_train_s1 = make_stage1_labels(y_train)
    y_val_s1 = make_stage1_labels(y_val)
    y_test_s1 = make_stage1_labels(y_test)

    train_dataset_s1 = SeqDataset(X_seq_train, y_train_s1)
    val_dataset_s1 = SeqDataset(X_seq_val, y_val_s1)
    test_dataset_s1 = SeqDataset(X_seq_test, y_test_s1)

    train_loader_s1 = DataLoader(
        train_dataset_s1, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader_s1 = DataLoader(
        val_dataset_s1, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader_s1 = DataLoader(
        test_dataset_s1, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # -----------------------------------------------------
    # Stage 2 data (abnormal subclass only)
    # -----------------------------------------------------
    X_train_s2, y_train_s2 = filter_abnormal_only(X_seq_train, y_train)
    X_val_s2, y_val_s2 = filter_abnormal_only(X_seq_val, y_val)
    X_test_s2, y_test_s2 = filter_abnormal_only(X_seq_test, y_test)

    print(f"[INFO] Stage2 train: {X_train_s2.shape}, {y_train_s2.shape}")
    print(f"[INFO] Stage2 val  : {X_val_s2.shape}, {y_val_s2.shape}")
    print(f"[INFO] Stage2 test : {X_test_s2.shape}, {y_test_s2.shape}")

    if len(y_train_s2) == 0 or len(y_val_s2) == 0 or len(y_test_s2) == 0:
        raise ValueError("Stage 2용 abnormal 샘플이 부족합니다.")

    train_dataset_s2 = SeqDataset(X_train_s2, y_train_s2)
    val_dataset_s2 = SeqDataset(X_val_s2, y_val_s2)
    test_dataset_s2 = SeqDataset(X_test_s2, y_test_s2)

    train_loader_s2 = DataLoader(
        train_dataset_s2, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader_s2 = DataLoader(
        val_dataset_s2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader_s2 = DataLoader(
        test_dataset_s2, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    # -----------------------------------------------------
    # Save dir
    # -----------------------------------------------------
    data_name = os.path.basename(os.path.normpath(args.data_root))
    exp_name = (
        f"s1_{args.stage1_model}_s2_{args.stage2_model}"
        f"_h{args.hidden_dim}_l{args.num_layers}"
    )
    save_dir = os.path.join(args.save_root, data_name, exp_name)
    ensure_dir(save_dir)

    # -----------------------------------------------------
    # Build models
    # -----------------------------------------------------
    stage1_model = build_model(
        model_name=args.stage1_model,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=2,
        dropout=args.dropout
    ).to(device)

    stage2_model = build_model(
        model_name=args.stage2_model,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=3,
        dropout=args.dropout
    ).to(device)

    print("[INFO] Stage1 model")
    print(stage1_model)
    print("[INFO] Stage2 model")
    print(stage2_model)

    # -----------------------------------------------------
    # Train stage 1
    # -----------------------------------------------------
    stage1_class_names = ["normal", "abnormal"]
    stage1_model, stage1_metrics, _, _ = run_stage_training(
        stage_name="stage1",
        model=stage1_model,
        train_loader=train_loader_s1,
        val_loader=val_loader_s1,
        test_loader=test_loader_s1,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=save_dir,
        class_names=stage1_class_names
    )

    # -----------------------------------------------------
    # Train stage 2
    # -----------------------------------------------------
    stage2_class_names = ["loose", "arc", "overcurrent"]
    stage2_model, stage2_metrics, _, _ = run_stage_training(
        stage_name="stage2",
        model=stage2_model,
        train_loader=train_loader_s2,
        val_loader=val_loader_s2,
        test_loader=test_loader_s2,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        save_dir=save_dir,
        class_names=stage2_class_names
    )

    # -----------------------------------------------------
    # Final combined inference on original test set
    # -----------------------------------------------------
    final_y_true, final_y_pred = run_final_two_stage_inference(
        stage1_model=stage1_model,
        stage2_model=stage2_model,
        test_loader_stage1=DataLoader(
            SeqDataset(X_seq_test, y_test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers
        ),
        device=device
    )

    final_acc = accuracy_score(final_y_true, final_y_pred)
    final_class_names = ["normal", "loose", "arc", "overcurrent"]
    final_cm = confusion_matrix(final_y_true, final_y_pred)

    save_confusion_matrix_heatmap(
        final_cm,
        final_class_names,
        os.path.join(save_dir, "final_confusion_matrix.png")
    )
    save_classification_report_text(
        final_y_true,
        final_y_pred,
        final_class_names,
        os.path.join(save_dir, "final_classification_report.txt")
    )

    metrics = {
        "data_root": args.data_root,
        "stage1_model": args.stage1_model,
        "stage2_model": args.stage2_model,
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "dropout": float(args.dropout),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "stage1": stage1_metrics,
        "stage2": stage2_metrics,
        "final_test_acc": float(final_acc),
    }
    save_metrics_json(metrics, os.path.join(save_dir, "metrics.json"))

    print(f"[FINAL] 2-stage test acc = {final_acc:.4f}")
    print(f"[SAVE] results saved to: {save_dir}")


if __name__ == "__main__":
    main()