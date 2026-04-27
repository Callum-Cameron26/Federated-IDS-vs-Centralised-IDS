from pathlib import Path

import numpy as np
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        # build hidden layers with ReLU to set negative values to 0
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        # final output layer no sigmoiid used as i train with BCEwithLogitsLoss
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


def build_model(input_dim: int, hidden_size: int = 64, layers: int = 2) -> MLP:
    # create MLP with specified architecture
    if layers < 1:
        raise ValueError("layers must be >= 1")
    hidden_sizes = [hidden_size for _ in range(layers)]
    return MLP(input_dim=input_dim, hidden_sizes=hidden_sizes)


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    # extract model weights as numpy arrays
    return [value.detach().cpu().numpy() for _, value in model.state_dict().items()]


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    # update model weights from numpy arrays
    current_state = model.state_dict()
    updated_state = {}
    for (name, tensor), array in zip(current_state.items(), parameters):
        updated_state[name] = torch.tensor(array, dtype=tensor.dtype)
    model.load_state_dict(updated_state, strict=True)


def save_model(
    model: nn.Module,
    path: Path | str,
    input_dim: int,
    hidden_size: int,
    layers: int,
) -> None:
    # save model
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_size": hidden_size,
            "layers": layers,
        },
        path_obj,
    )


def load_model(path: Path | str, device: str = "cpu") -> tuple[nn.Module, dict]:
    # load model and recreate architecture from saved config
    checkpoint = torch.load(Path(path), map_location=device)
    model = build_model(
        input_dim=int(checkpoint["input_dim"]),
        hidden_size=int(checkpoint["hidden_size"]),
        layers=int(checkpoint["layers"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint
