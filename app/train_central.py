from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .evaluate import evaluate_model, save_evaluation_plots
from .model import build_model, load_model, save_model
from .utils import (
    copy_reports_to_run,
    save_json,
    set_seed,
    timestamp_run_dir,
)


def _load_pool_and_test(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # load training pool and test set from partition step
    processed_dir = data_root / "processed"
    train_path = processed_dir / "train_pool.npz"
    test_path = processed_dir / "global_test.npz"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Run partition first. train_pool.npz and global_test.npz are required")

    train_data = np.load(train_path)
    test_data = np.load(test_path)
    return train_data["X"], train_data["y"], test_data["X"], test_data["y"]


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    # create PyTorch DataLoader from numpy arrays
    tensor_x = torch.tensor(X, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_and_evaluate_central(
    data_root: Path,
    runs_root: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    hidden_size: int,
    layers: int,
    cli_args: dict,
    experiment_tag: str = "cicids",
) -> tuple[Path, dict]:
    # train centralized model and evaluate on test set
    set_seed(seed)

    prefix = f"central_{experiment_tag}"
    run_dir = timestamp_run_dir(runs_root, prefix)
    copy_reports_to_run(data_root, run_dir)
    save_json(cli_args, run_dir / "configs.json")

    # load data
    X_train, y_train, X_test, y_test = _load_pool_and_test(data_root)
    input_dim = int(X_train.shape[1])

    # build model and data loaders
    model = build_model(input_dim=input_dim, hidden_size=hidden_size, layers=layers).to(device)
    train_loader = _make_loader(X_train, y_train, batch_size=batch_size, shuffle=True)

    # training setup
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"Training for {epochs} epochs on {device}")

    # train the model
    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for features, targets in train_loader:
            features = features.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * len(targets)

        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{epochs} loss: {avg_loss:.6f}")

    # evaluate on test set
    metrics, y_true, y_prob, y_pred = evaluate_model(
        model=model,
        X=X_test,
        y=y_test,
        device=device,
        batch_size=batch_size,
    )

    # save results
    save_json(metrics, run_dir / f"{prefix}_metrics.json")
    save_evaluation_plots(
        prefix=prefix,
        run_dir=run_dir,
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
    )

    save_model(
        model=model,
        path=run_dir / f"{prefix}_model.pt",
        input_dim=input_dim,
        hidden_size=hidden_size,
        layers=layers,
    )

    print(f"Saved central outputs in: {run_dir}")
    return run_dir, metrics


def _latest_central_model(runs_root: Path) -> Path:
    candidates = sorted(runs_root.glob("central_*/central_model.pt"))
    if not candidates:
        raise FileNotFoundError("No central_model.pt found in runs. Run central-train first")
    return candidates[-1]


def evaluate_saved_central_model(
    data_root: Path,
    runs_root: Path,
    model_path: Path | None,
    seed: int,
    batch_size: int,
    device: str,
    cli_args: dict,
    experiment_tag: str = "cicids",
) -> tuple[Path, dict]:
    set_seed(seed)
    prefix = f"central_{experiment_tag}"
    run_dir = timestamp_run_dir(runs_root, f"{prefix}_eval")
    copy_reports_to_run(data_root, run_dir)
    save_json(cli_args, run_dir / "configs.json")

    selected_model_path = model_path if model_path else _latest_central_model(runs_root)
    model, _ = load_model(selected_model_path, device=device)

    test_path = data_root / "processed" / "global_test.npz"
    if not test_path.exists():
        raise FileNotFoundError("global_test.npz not found. Run partition first")

    test_data = np.load(test_path)
    X_test = test_data["X"]
    y_test = test_data["y"]

    metrics, y_true, y_prob, y_pred = evaluate_model(
        model=model,
        X=X_test,
        y=y_test,
        device=device,
        batch_size=batch_size,
    )

    save_json(metrics, run_dir / f"{prefix}_metrics.json")
    save_evaluation_plots(
        prefix=prefix,
        run_dir=run_dir,
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
    )

    source_model_path = Path(selected_model_path)
    copied_model_path = run_dir / f"{prefix}_model.pt"
    copied_model_path.write_bytes(source_model_path.read_bytes())

    print(f"Saved central evaluation outputs in: {run_dir}")
    return run_dir, metrics
