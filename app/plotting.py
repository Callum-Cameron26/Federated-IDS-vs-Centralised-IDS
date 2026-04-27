from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    # create heatmap of confusion matrix with counts
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    # set labels and ticks
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"])
    ax.set_yticklabels(["Benign", "Attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    # add count numbers to each cell
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_normalized_confusion_matrix(cm: np.ndarray, path: Path, title: str) -> None:
    # create heatmap of row-normalised confusion matrix
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sums
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Benign", "Attack"])
    ax.set_yticklabels(["Benign", "Attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path, title: str) -> None:
    #plot ROC curve with chance baseline
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, label="ROC")
    except ValueError:
        pass  #handle single class case

    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path, title: str) -> None:
    # plot precision-recall curve
    fig, ax = plt.subplots(figsize=(6, 4))
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(recall, precision, label="PR")
    except ValueError:
        pass  # handle single class case

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_convergence(round_metrics: pd.DataFrame, path: Path) -> None:
    # plot federated learning convergence over rounds
    fig, ax = plt.subplots(figsize=(7, 4))

    if not round_metrics.empty:
        # plot accuracy and F1 over rounds
        ax.plot(round_metrics["round"], round_metrics["accuracy"], label="Accuracy")
        ax.plot(round_metrics["round"], round_metrics["f1"], label="F1")

    ax.set_xlabel("Round")
    ax.set_ylabel("Score")
    ax.set_title("Federated Convergence")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
