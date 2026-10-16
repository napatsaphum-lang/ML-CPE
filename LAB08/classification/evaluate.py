"""บันทึกกราฟและผลการทำนาย"""
from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report


def plot_history(history, output_dir, model_name):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in zip(axes, ("accuracy", "loss"), ("Accuracy", "Loss")):
        ax.plot(history.history[metric], label="Train")
        ax.plot(history.history["val_" + metric], label="Validation")
        ax.set(title=title, xlabel="Epoch", ylabel=metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"history_{model_name}.png", dpi=150)
    plt.close(fig)


def save_evaluation(model, test_ds, classes, output_dir):
    truths, predictions = [], []
    sample_imgs, sample_true, sample_pred = [], [], []
    for images, labels in test_ds:
        scores = model.predict(images, verbose=0)
        pred = np.argmax(scores, axis=1)
        truths.extend(labels.numpy().tolist())
        predictions.extend(pred.tolist())
        if len(sample_imgs) < 9:
            for i in range(min(len(images), 9 - len(sample_imgs))):
                sample_imgs.append(images[i].numpy().astype("uint8"))
                sample_true.append(int(labels[i]))
                sample_pred.append(int(pred[i]))
    report = classification_report(truths, predictions, target_names=classes, zero_division=0)
    (output_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    matrix = confusion_matrix(truths, predictions)
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(ax=ax, xticks_rotation=45, colorbar=False)
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(sample_imgs[i])
        ax.set_title(f"True: {classes[sample_true[i]]}\nPred: {classes[sample_pred[i]]}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / "predictions.png", dpi=150)
    plt.close(fig)


def save_comparison(rows, output_dir):
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "conv_layers", "neurons", "epochs", "train_accuracy", "val_accuracy", "test_accuracy", "test_loss"])
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar([r["model"] for r in rows], [r["test_accuracy"] for r in rows])
    ax.set(xlabel="Experiment", ylabel="Test accuracy", title="DCNN Model Comparison", ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(output_dir / "comparison.png", dpi=150)
    plt.close(fig)
