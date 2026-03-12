import os
import argparse
import json

import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from datasets import SeqDataset, HybridDataset
from models import (
    GRUClassifier,
    GRUAttentionClassifier,
    HybridGRUClassifier,
    PhysicsGRUClassifier
)


# =========================================================
# Utility
# =========================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_dataset(data_root):

    X_seq_test = np.load(os.path.join(data_root, "X_seq_test.npy"))
    X_feat_test = np.load(os.path.join(data_root, "X_feat_test.npy"))
    y_test = np.load(os.path.join(data_root, "y_test.npy"))

    return X_seq_test, X_feat_test, y_test


# =========================================================
# Model loader
# =========================================================
def load_model(model_type, model_path, device, input_dim, feat_dim=None):

    if model_type == "gru":
        model = GRUClassifier(input_dim=input_dim)

    elif model_type == "gru_attn":
        model = GRUAttentionClassifier(input_dim=input_dim)

    elif model_type == "hybrid":
        model = HybridGRUClassifier(
            seq_input_dim=input_dim,
            feat_input_dim=feat_dim
        )

    elif model_type == "physics":
        model = PhysicsGRUClassifier(input_dim=input_dim)

    else:
        raise ValueError("Unknown model type")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model


# =========================================================
# Prediction
# =========================================================
@torch.no_grad()
def predict_seq_model(model, loader, device):

    probs_all = []
    targets = []

    for x_seq, y in loader:

        x_seq = x_seq.to(device)

        output = model(x_seq)

        if isinstance(output, tuple):
            logits = output[0]
        else:
            logits = output

        probs = torch.softmax(logits, dim=1)

        probs_all.append(probs.cpu().numpy())
        targets.append(y.numpy())

    probs_all = np.concatenate(probs_all)
    targets = np.concatenate(targets)

    return probs_all, targets


@torch.no_grad()
def predict_hybrid_model(model, loader, device):

    probs_all = []
    targets = []

    for x_seq, x_feat, y in loader:

        x_seq = x_seq.to(device)
        x_feat = x_feat.to(device)

        logits = model(x_seq, x_feat)

        probs = torch.softmax(logits, dim=1)

        probs_all.append(probs.cpu().numpy())
        targets.append(y.numpy())

    probs_all = np.concatenate(probs_all)
    targets = np.concatenate(targets)

    return probs_all, targets


# =========================================================
# Visualization
# =========================================================
def save_confusion_matrix(cm, class_names, save_path):

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()

    ticks = np.arange(len(class_names))

    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)

    thresh = cm.max() / 2

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True")
    plt.xlabel("Predicted")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# =========================================================
# Main
# =========================================================
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_root", required=True)
    parser.add_argument("--save_dir", default="./results/ensemble")

    parser.add_argument("--gru_model")
    parser.add_argument("--attn_model")
    parser.add_argument("--hybrid_model")
    parser.add_argument("--physics_model")

    parser.add_argument("--batch_size", default=64, type=int)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ensure_dir(args.save_dir)

    # =====================================================
    # load data
    # =====================================================
    X_seq_test, X_feat_test, y_test = load_dataset(args.data_root)

    input_dim = X_seq_test.shape[-1]
    feat_dim = X_feat_test.shape[-1]

    seq_dataset = SeqDataset(X_seq_test, y_test)
    seq_loader = DataLoader(seq_dataset, batch_size=args.batch_size)

    hybrid_dataset = HybridDataset(X_seq_test, X_feat_test, y_test)
    hybrid_loader = DataLoader(hybrid_dataset, batch_size=args.batch_size)

    prob_list = []

    # =====================================================
    # GRU
    # =====================================================
    if args.gru_model:

        model = load_model(
            "gru",
            args.gru_model,
            device,
            input_dim
        )

        probs, _ = predict_seq_model(model, seq_loader, device)

        prob_list.append(probs)

        print("GRU loaded")

    # =====================================================
    # GRU ATTENTION
    # =====================================================
    if args.attn_model:

        model = load_model(
            "gru_attn",
            args.attn_model,
            device,
            input_dim
        )

        probs, _ = predict_seq_model(model, seq_loader, device)

        prob_list.append(probs)

        print("Attention GRU loaded")

    # =====================================================
    # HYBRID
    # =====================================================
    if args.hybrid_model:

        model = load_model(
            "hybrid",
            args.hybrid_model,
            device,
            input_dim,
            feat_dim
        )

        probs, _ = predict_hybrid_model(model, hybrid_loader, device)

        prob_list.append(probs)

        print("Hybrid loaded")

    # =====================================================
    # PHYSICS
    # =====================================================
    if args.physics_model:

        model = load_model(
            "physics",
            args.physics_model,
            device,
            input_dim
        )

        probs, _ = predict_seq_model(model, seq_loader, device)

        prob_list.append(probs)

        print("Physics model loaded")

    # =====================================================
    # Ensemble
    # =====================================================
    probs_ensemble = np.mean(prob_list, axis=0)

    preds = np.argmax(probs_ensemble, axis=1)

    acc = accuracy_score(y_test, preds)

    print("Ensemble accuracy:", acc)

    class_names = ["normal", "loose", "arc", "overcurrent"]

    cm = confusion_matrix(y_test, preds)

    save_confusion_matrix(
        cm,
        class_names,
        os.path.join(args.save_dir, "ensemble_confusion_matrix.png")
    )

    report = classification_report(
        y_test,
        preds,
        target_names=class_names
    )

    with open(os.path.join(args.save_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    metrics = {
        "ensemble_acc": float(acc)
    }

    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("Saved results →", args.save_dir)


if __name__ == "__main__":
    main()