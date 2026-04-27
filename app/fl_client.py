import os
import re
from pathlib import Path

import flwr as fl
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .model import build_model, get_parameters, set_parameters
from .utils import set_seed


def _hostname_to_client_id(hostname: str) -> int | None:
    # extract client number from hostname
    match = re.search(r"-(\d+)$", hostname)
    if not match:
        return None
    index = int(match.group(1)) - 1
    return max(index, 0)


def resolve_client_id(cli_client_id: int | None) -> int:
    # figure out which client ID this instance should use
    #ive passed from cli then skip the rest and return arg
    if cli_client_id is not None:
        return cli_client_id

    #or get client ID from enviroment variable or the hostname, allows for different testing enviroments such as multi process or docker, just adds flexibility
    env_client_id = os.getenv("CLIENT_ID")
    host_client_id = _hostname_to_client_id(os.getenv("HOSTNAME", ""))

    if env_client_id:
        if env_client_id.lower() == "auto":
            if host_client_id is not None:
                return host_client_id
            raise ValueError("CLIENT_ID=auto needs a hostname")

        parsed_env_id = int(env_client_id)
        if parsed_env_id == 0 and host_client_id is not None:
            return host_client_id
        return parsed_env_id

    if host_client_id is not None:
        return host_client_id

    raise ValueError("Provide --client-id or set CLIENT_ID")


def _available_client_ids(data_root: Path) -> list[int]:
    # find all client partition files that exist
    partition_dir = data_root / "processed" / "partitions"
    files = sorted(partition_dir.glob("client_*.npz"))
    ids: list[int] = []
    #loop files and use regex to extract ID from file names and append to list of client IDS
    for file_path in files:
        match = re.search(r"client_(\d+)\.npz$", file_path.name)
        if match:
            ids.append(int(match.group(1)))
    return ids


def _assign_client_id_from_locks(data_root: Path) -> int:
    # assign client ID using lock files to avoid conflicts
    client_ids = _available_client_ids(data_root)
    if not client_ids:
        raise FileNotFoundError("No client partitions found. Run partition first")

    host = os.getenv("HOSTNAME", "unknown-host")
    runs_dir = Path(os.getenv("RUNS_DIR", "runs"))
    assign_dir = runs_dir / "client_assignments"
    assign_dir.mkdir(parents=True, exist_ok=True)

    # check if already assigned this host
    existing_assignment = assign_dir / f"{host}.txt"
    if existing_assignment.exists():
        value = existing_assignment.read_text(encoding="utf-8").strip()
        if value:
            return int(value)

    # try to claim an unused client ID
    for client_id in client_ids:
        lock_path = assign_dir / f"client_{client_id}.lock"
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            existing_assignment.write_text(str(client_id), encoding="utf-8")
            return client_id
        except FileExistsError:
            continue

    # fallback to hash based assignment if all IDS are taken
    fallback = client_ids[hash(host) % len(client_ids)]
    existing_assignment.write_text(str(fallback), encoding="utf-8")
    return fallback


def _load_client_partition(data_root: Path, client_id: int) -> tuple[np.ndarray, np.ndarray]:
    # load this client's training data from its partition file
    path = data_root / "processed" / "partitions" / f"client_{client_id}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Partition file not found for client ID {client_id}: {path}")

    data = np.load(path)
    X = data["X"].astype(np.float32)
    y = data["y"].astype(np.float32)

    if len(y) == 0:
        raise ValueError(f"Client {client_id} partition is empty")

    return X, y


class IDSClient(fl.client.NumPyClient):
    def __init__(
        self,
        model: torch.nn.Module,
        X_train: np.ndarray,
        y_train: np.ndarray,
        device: str,
        batch_size: int,
        lr: float,
        local_epochs: int,
    ) -> None:
        # store training setup
        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.local_epochs = local_epochs

        # convert numpy arrays to torch dataset
        tensor_x = torch.tensor(X_train, dtype=torch.float32)
        tensor_y = torch.tensor(y_train, dtype=torch.float32)
        self.train_dataset = TensorDataset(tensor_x, tensor_y)

    def get_parameters(self, config):
        # return current model weights to server
        return get_parameters(self.model)

    def fit(self, parameters, config):
        # train locally on this client's data
        set_parameters(self.model, parameters)

        local_epochs = int(config.get("local_epochs", self.local_epochs))
        batch_size = int(config.get("batch_size", self.batch_size))
        lr = float(config.get("lr", self.lr))

        loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        running_loss = 0.0

        # train for specified epochs
        for _ in range(local_epochs):
            for features, targets in loader:
                features = features.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                logits = self.model(features)
                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                running_loss += float(loss.item()) * len(targets)

        avg_loss = running_loss / (len(loader.dataset) * local_epochs)
        return get_parameters(self.model), len(loader.dataset), {"train_loss": avg_loss}

    def evaluate(self, parameters, config):
        # server handles evaluation, just return placeholder
        set_parameters(self.model, parameters)
        return 0.0, len(self.train_dataset), {}


def start_fl_client(
    data_root: Path,
    server_addr: str,
    client_id: int | None,
    seed: int,
    batch_size: int,
    lr: float,
    local_epochs: int,
    hidden_size: int,
    layers: int,
    device: str,
) -> None:
    # figure out which client this is and start training
    resolved_client_id = resolve_client_id(client_id)
    if resolved_client_id == 0:
        resolved_client_id = _assign_client_id_from_locks(data_root)
    set_seed(seed + resolved_client_id)

    # load this clients data and build model
    X_train, y_train = _load_client_partition(data_root, resolved_client_id)
    input_dim = int(X_train.shape[1])

    model = build_model(input_dim=input_dim, hidden_size=hidden_size, layers=layers).to(device)
    client = IDSClient(
        model=model,
        X_train=X_train,
        y_train=y_train,
        device=device,
        batch_size=batch_size,
        lr=lr,
        local_epochs=local_epochs,
    )

    print(f"Client {resolved_client_id} connecting to {server_addr}")
    fl.client.start_numpy_client(server_address=server_addr, client=client)
