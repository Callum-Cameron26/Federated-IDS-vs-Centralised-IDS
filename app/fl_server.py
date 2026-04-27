from pathlib import Path

import flwr as fl
import numpy as np
import pandas as pd
from flwr.common import parameters_to_ndarrays

from .evaluate import evaluate_model, save_evaluation_plots
from .model import build_model, get_parameters, save_model, set_parameters
from .plotting import plot_convergence
from .utils import copy_reports_to_run, save_json, set_seed, timestamp_run_dir

#customise flowers fedavg by adding in storage for weights
class TrackingFedAvg(fl.server.strategy.FedAvg):
    def __init__(self, initial_ndarrays: list[np.ndarray], **kwargs) -> None:
        # store latest aggregated weights
        self.latest_ndarrays = initial_ndarrays
        super().__init__(initial_parameters=fl.common.ndarrays_to_parameters(initial_ndarrays), **kwargs)

    def aggregate_fit(self, server_round, results, failures):
        # aggregate client updates and keep latest weights
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is not None:
            self.latest_ndarrays = parameters_to_ndarrays(aggregated_parameters)
        return aggregated_parameters, aggregated_metrics


def _load_global_test(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    # load the global test set for evaluation
    test_path = data_root / "processed" / "global_test.npz"
    if not test_path.exists():
        raise FileNotFoundError("global_test.npz not found Run partition first")
    test_data = np.load(test_path)
    return test_data["X"], test_data["y"]


def _load_input_dim(data_root: Path) -> int:
    # get feature count from training data
    train_pool_path = data_root / "processed" / "train_pool.npz"
    if not train_pool_path.exists():
        raise FileNotFoundError("train_pool.npz not found Run partition first")
    train_pool = np.load(train_pool_path)
    return int(train_pool["X"].shape[1])


def _binary_log_loss(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    # calculate binary cross entropy loss, cant use BCEWithLogitsLoss again because these are probabilities, also cant use nn.BCELoss function because it needs pytorch tensors, these are numpy arays
    #clipped to avoid nan o inf values
    eps = 1e-8
    clipped = np.clip(y_prob, eps, 1.0 - eps)
    loss = -(y_true * np.log(clipped) + (1 - y_true) * np.log(1 - clipped)).mean()
    return float(loss)


def start_fl_server(
    data_root: Path,
    runs_root: Path,
    server_addr: str,
    clients: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    hidden_size: int,
    layers: int,
    device: str,
    cli_args: dict,
    iid: bool = True,
    experiment_tag: str = "cicids",
) -> tuple[Path, dict]:
    # setup and run federated learning server
    #makes results reproducable
    set_seed(seed)

    # build descriptive prefix capture
    iid_str = "iid" if iid else "noniid"
    prefix = f"fl_{experiment_tag}_{iid_str}_{clients}clients_rounds{rounds}"

    # create output directory and save config
    run_dir = timestamp_run_dir(runs_root, prefix)
    copy_reports_to_run(data_root, run_dir)
    save_json(cli_args, run_dir / "configs.json")

    # load test data and build model
    X_test, y_test = _load_global_test(data_root)
    input_dim = _load_input_dim(data_root)

    model = build_model(input_dim=input_dim, hidden_size=hidden_size, layers=layers).to(device)
    initial_params = get_parameters(model)

    round_records: list[dict] = []

    def evaluate_fn(server_round, parameters, config):
        # evaluate aggregated model on global test set
        if isinstance(parameters, list):
            ndarrays = parameters
        else:
            ndarrays = parameters_to_ndarrays(parameters)
        set_parameters(model, ndarrays)

        # run evaluation
        metrics, y_true, y_prob, y_pred = evaluate_model(
            model=model,
            X=X_test,
            y=y_test,
            device=device,
            batch_size=batch_size,
        )

        loss = _binary_log_loss(y_true.astype(np.float32), y_prob.astype(np.float32))

        # record metrics for this round
        row = {
            "round": int(server_round),
            "loss": float(loss),
            "accuracy": float(metrics["accuracy"]),
            "precision": float(metrics["precision"]),
            "recall": float(metrics["recall"]),
            "f1": float(metrics["f1"]),
            "roc_auc": metrics["roc_auc"],
            "pr_auc": metrics["pr_auc"],
        }
        round_records.append(row)

        # print round results
        print(
            f"Round {server_round} eval "
            f"acc={row['accuracy']:.4f} f1={row['f1']:.4f} "
            f"roc_auc={row['roc_auc']} pr_auc={row['pr_auc']}"
        )

        # return loss and scalar metrics to Flower, the auc values are locked behind an if statement incase a 2nd class doesnt exist in the data then it the try block to calculat them wouldnt be possible
        scalar_metrics = {
            "accuracy": row["accuracy"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
        }
        if row["roc_auc"] is not None:
            scalar_metrics["roc_auc"] = float(row["roc_auc"])
        if row["pr_auc"] is not None:
            scalar_metrics["pr_auc"] = float(row["pr_auc"])

        return float(loss), scalar_metrics

    def on_fit_config_fn(server_round: int) -> dict[str, float | int]:
        # send training config to clients each round
        return {
            "local_epochs": int(local_epochs),
            "batch_size": int(batch_size),
            "lr": float(lr),
        }

    # create federated averaging strategy
    strategy = TrackingFedAvg(
        initial_ndarrays=initial_params,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=1,
        min_available_clients=1,
        min_evaluate_clients=0,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=on_fit_config_fn,
    )

    # start the Flower server
    print(f"Starting Flower server at {server_addr} for {rounds} rounds")
    fl.server.start_server(
        server_address=server_addr,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )

    # final evaluation with latest weights
    set_parameters(model, strategy.latest_ndarrays)
    final_metrics, y_true, y_prob, y_pred = evaluate_model(
        model=model,
        X=X_test,
        y=y_test,
        device=device,
        batch_size=batch_size,
    )

    # save final results
    save_json(final_metrics, run_dir / f"{prefix}_final_metrics.json")
    save_evaluation_plots(
        prefix=prefix,
        run_dir=run_dir,
        y_true=y_true,
        y_prob=y_prob,
        y_pred=y_pred,
    )

    save_model(
        model=model,
        path=run_dir / f"{prefix}_global_model.pt",
        input_dim=input_dim,
        hidden_size=hidden_size,
        layers=layers,
    )

    # save metrics over rounds and plot convergence
    columns = ["round", "loss", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    round_df = pd.DataFrame(round_records, columns=columns)
    round_df.to_csv(run_dir / f"{prefix}_metrics_over_rounds.csv", index=False)
    plot_convergence(round_df, run_dir / f"{prefix}_convergence.png")

    print(f"Saved federated outputs in: {run_dir}")
    return run_dir, final_metrics
