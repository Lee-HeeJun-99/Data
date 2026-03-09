import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

def save_classification_report(y_true, y_pred, save_path):
    report = classification_report(y_true, y_pred, digits=4)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)

def save_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()

def save_curves(train_acc, val_acc, train_loss, val_loss, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(train_acc, label="train acc")
    plt.plot(val_acc, label="val acc")
    plt.legend()
    plt.title("Accuracy Curve")
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(train_loss, label="train loss")
    plt.plot(val_loss, label="val loss")
    plt.legend()
    plt.title("Loss Curve")
    plt.savefig(os.path.join(save_dir, "loss_curve.png"), dpi=200, bbox_inches="tight")
    plt.close()