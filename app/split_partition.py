from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .utils import ensure_dir, save_json


def _class_distribution(y: np.ndarray) -> dict[str, int]:
    # count samples per class in array
    labels, counts = np.unique(y, return_counts=True)
    return {str(int(label)): int(count) for label, count in zip(labels, counts)}


def _iid_indices(size: int, clients: int, seed: int) -> list[np.ndarray]:
    # random shuffle then split for IID partitions
    rng = np.random.default_rng(seed)
    indices = np.arange(size)
    rng.shuffle(indices)
    return [part.astype(int) for part in np.array_split(indices, clients)]


def _non_iid_indices(y: np.ndarray, clients: int, seed: int) -> list[np.ndarray]:
    # sort by label then split for non-IID partitions
    rng = np.random.default_rng(seed)
    ordered: list[int] = []
    for label in sorted(np.unique(y)):
        label_idx = np.where(y == label)[0]
        rng.shuffle(label_idx)
        ordered.extend(label_idx.tolist())
    ordered_idx = np.array(ordered, dtype=int)
    return [part.astype(int) for part in np.array_split(ordered_idx, clients)]


def create_partitions(
    data_root: Path,
    label_col: str,
    clients: int,
    test_size: float,
    seed: int,
    iid: bool,
) -> dict:
    # split data into train/test and create client partitions
    processed_dir = ensure_dir(data_root / "processed")
    cleaned_path = processed_dir / "cleaned.csv"

    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found: {cleaned_path}")

    # load cleaned data
    df = pd.read_csv(cleaned_path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in cleaned data")

    y = df[label_col].to_numpy().astype(int)
    X = df.drop(columns=[label_col])
    feature_columns = X.columns.tolist()

    # stratified train/test split
    stratify = y if len(np.unique(y)) > 1 else None
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )

    # scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.to_numpy(dtype=np.float32)).astype(np.float32)
    X_test = scaler.transform(X_test_df.to_numpy(dtype=np.float32)).astype(np.float32)

    # save scaler and feature list
    joblib.dump(scaler, processed_dir / "scaler.joblib")
    save_json({"features": feature_columns}, processed_dir / "feature_list.json")

    # save train pool and test set
    np.savez_compressed(processed_dir / "train_pool.npz", X=X_train, y=y_train.astype(np.int64))
    np.savez_compressed(processed_dir / "global_test.npz", X=X_test, y=y_test.astype(np.int64))

    # clean old partitions
    partition_dir = ensure_dir(processed_dir / "partitions")
    for old_file in partition_dir.glob("client_*.npz"):
        old_file.unlink()

    # create client partitions
    if iid:
        index_groups = _iid_indices(len(y_train), clients, seed)
    else:
        index_groups = _non_iid_indices(y_train, clients, seed)

    # save each client's partition
    client_reports: list[dict] = []
    for client_id, idx in enumerate(index_groups):
        client_x = X_train[idx]
        client_y = y_train[idx]
        np.savez_compressed(partition_dir / f"client_{client_id}.npz", X=client_x, y=client_y)
        client_reports.append(
            {
                "client_id": client_id,
                "samples": int(len(client_y)),
                "class_distribution": _class_distribution(client_y),
            }
        )

    # create partition report
    report = {
        "label_column": label_col,
        "clients": int(clients),
        "iid": bool(iid),
        "test_size": float(test_size),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "train_class_distribution": _class_distribution(y_train),
        "test_class_distribution": _class_distribution(y_test),
        "clients_report": client_reports,
    }

    save_json(report, processed_dir / "partition_report.json")

    print(f"Train samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")
    print(f"Saved {clients} client partitions in: {partition_dir}")

    return report
