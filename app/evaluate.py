from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, TensorDataset

from .plotting import plot_confusion_matrix, plot_normalized_confusion_matrix, plot_pr_curve, plot_roc_curve


@torch.no_grad()
def predict_probabilities(
    model: torch.nn.Module,
    X: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    # get model predictions without gradients
    model.eval()
    tensor_x = torch.tensor(X, dtype=torch.float32)
    dataset = TensorDataset(tensor_x)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probs: list[np.ndarray] = []
    # loop through batches and get predictions
    for (features,) in loader:
        features = features.to(device)
        logits = model(features)
        batch_probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(batch_probs)

    return np.concatenate(probs, axis=0)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> tuple[dict, np.ndarray]:
    # convert probs to binary predictions
    y_pred = (y_prob >= threshold).astype(int)

    # calculate basic metrics
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).astype(int).tolist(),
    }

    # try to calculate ROC AUC 
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["roc_auc"] = None

    # try to calculate PR AUC
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["pr_auc"] = None

    return metrics, y_pred


def evaluate_model(
    model: torch.nn.Module,
    X: np.ndarray,
    y: np.ndarray,
    device: str,
    batch_size: int,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    # get predictions and calculate metrics
    y_prob = predict_probabilities(model=model, X=X, device=device, batch_size=batch_size)
    metrics, y_pred = compute_binary_metrics(y_true=y, y_prob=y_prob)
    return metrics, y, y_prob, y_pred


def save_evaluation_plots(
    prefix: str,
    run_dir: Path,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    # generate and save all evaluation plots
    cm = confusion_matrix(y_true, y_pred).astype(int)
    plot_confusion_matrix(
        cm,
        run_dir / f"{prefix}_confusion_matrix.png",
        title=f"{prefix.upper()} Confusion Matrix",
    )
    plot_normalized_confusion_matrix(
        cm,
        run_dir / f"{prefix}_normalized_confusion_matrix.png",
        title=f"{prefix.upper()} Normalised Confusion Matrix",
    )
    plot_roc_curve(
        y_true,
        y_prob,
        run_dir / f"{prefix}_roc.png",
        title=f"{prefix.upper()} ROC Curve",
    )
    plot_pr_curve(
        y_true,
        y_prob,
        run_dir / f"{prefix}_pr.png",
        title=f"{prefix.upper()} Precision-Recall Curve",
    )
