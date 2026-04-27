import argparse
import os
from pathlib import Path

from .fl_client import start_fl_client
from .fl_server import start_fl_server
from .preprocess import preprocess_dataset
from .split_partition import create_partitions
from .train_central import evaluate_saved_central_model, train_and_evaluate_central
from .utils import resolve_data_root, resolve_input_data_paths, resolve_runs_root

_CICIDS_DEFAULT_PATHS = ",".join([
    "Data/MachineLearningCVE/Monday-WorkingHours.pcap_ISCX.csv",
    "Data/MachineLearningCVE/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Data/MachineLearningCVE/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
])

_CICIDS_DROP_COLS = ",".join([
    "Flow ID", "Source IP", "Source Port",
    "Destination IP", "Destination Port", "Timestamp",
])


def str_to_bool(value: str) -> bool:
    value_lower = value.lower().strip()
    if value_lower in {"1", "true", "yes", "y"}:
        return True
    if value_lower in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Use true or false")


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-path", type=str, default=None, help="Comma-separated CSV paths")
    parser.add_argument("--label-col", type=str, default="Label")
    parser.add_argument("--binary", type=str_to_bool, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--runs-dir", type=str, default=None)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "y"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Federated IDS pipeline (CICIDS2017)")
    subparsers = parser.add_subparsers(dest="command")

    preprocess_parser = subparsers.add_parser("preprocess")
    add_common_args(preprocess_parser)
    preprocess_parser.add_argument("--sep", type=str, default=",")
    preprocess_parser.add_argument(
        "--drop-cols", type=str, default=_CICIDS_DROP_COLS,
        help="csv column names to drop (defaults to CICIDS2017 identifier columns)",
    )
    preprocess_parser.add_argument(
        "--encode-cols", type=str, default="",
        help="csv string column names to label-encode as integers",
    )

    partition_parser = subparsers.add_parser("partition")
    add_common_args(partition_parser)
    partition_parser.add_argument("--clients", type=int, default=2)
    partition_parser.add_argument("--iid", action="store_true")
    partition_parser.add_argument("--non-iid", action="store_true")

    central_train_parser = subparsers.add_parser("central-train")
    add_common_args(central_train_parser)
    central_train_parser.add_argument("--epochs", type=int, default=5)
    central_train_parser.add_argument("--batch-size", type=int, default=256)
    central_train_parser.add_argument("--lr", type=float, default=1e-3)
    central_train_parser.add_argument("--device", type=str, default="cpu")
    central_train_parser.add_argument("--hidden-size", type=int, default=64)
    central_train_parser.add_argument("--layers", type=int, default=2)
    central_train_parser.add_argument("--experiment-tag", type=str, default="cicids")

    central_eval_parser = subparsers.add_parser("central-eval")
    add_common_args(central_eval_parser)
    central_eval_parser.add_argument("--batch-size", type=int, default=256)
    central_eval_parser.add_argument("--device", type=str, default="cpu")
    central_eval_parser.add_argument("--model-path", type=str, default=None)
    central_eval_parser.add_argument("--experiment-tag", type=str, default="cicids")

    fl_server_parser = subparsers.add_parser("fl-server")
    add_common_args(fl_server_parser)
    fl_server_parser.add_argument("--clients", type=int, default=2)
    fl_server_parser.add_argument("--rounds", type=int, default=5)
    fl_server_parser.add_argument("--local-epochs", type=int, default=1)
    fl_server_parser.add_argument("--batch-size", type=int, default=256)
    fl_server_parser.add_argument("--lr", type=float, default=1e-3)
    fl_server_parser.add_argument("--device", type=str, default="cpu")
    fl_server_parser.add_argument("--server-addr", type=str, default="0.0.0.0:8080")
    fl_server_parser.add_argument("--hidden-size", type=int, default=64)
    fl_server_parser.add_argument("--layers", type=int, default=2)
    fl_server_parser.add_argument("--iid", action="store_true")
    fl_server_parser.add_argument("--non-iid", action="store_true")
    fl_server_parser.add_argument("--experiment-tag", type=str, default="cicids")

    fl_client_parser = subparsers.add_parser("fl-client")
    add_common_args(fl_client_parser)
    fl_client_parser.add_argument("--client-id", type=int, default=None)
    fl_client_parser.add_argument("--local-epochs", type=int, default=1)
    fl_client_parser.add_argument("--batch-size", type=int, default=256)
    fl_client_parser.add_argument("--lr", type=float, default=1e-3)
    fl_client_parser.add_argument("--device", type=str, default="cpu")
    fl_client_parser.add_argument("--server-addr", type=str, default="server:8080")
    fl_client_parser.add_argument("--hidden-size", type=int, default=64)
    fl_client_parser.add_argument("--layers", type=int, default=2)


    return parser


def check_prerequisites(data_root: Path, command: str) -> None:
    # check if required files exist for this command
    processed_dir = data_root / "processed"

    if command in {"partition", "central-train", "central-eval", "fl-server", "fl-client"}:
        if not (processed_dir / "cleaned.csv").exists():
            raise FileNotFoundError("Run preprocess first - no cleaned dataset found")

    if command in {"central-train", "central-eval", "fl-server", "fl-client"}:
        if not (processed_dir / "train_pool.npz").exists():
            raise FileNotFoundError("Run partition first - no training data found")
        if not (processed_dir / "global_test.npz").exists():
            raise FileNotFoundError("Run partition first - no test data found")

    if command in {"fl-server", "fl-client"}:
        partition_dir = processed_dir / "partitions"
        if not partition_dir.exists() or not list(partition_dir.glob("client_*.npz")):
            raise FileNotFoundError("Run partition first - no client partitions found")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    role = os.getenv("ROLE", "").strip().lower()

    if not args.command and role in {"server", "client"}:
        data_root = resolve_data_root(None)
        runs_root = resolve_runs_root(None)

        if role == "server":
            iid = _env_bool("IID", True)
            start_fl_server(
                data_root=data_root,
                runs_root=runs_root,
                server_addr=_env_str("SERVER_ADDR", "0.0.0.0:8080"),
                clients=_env_int("CLIENTS", 2),
                rounds=_env_int("ROUNDS", 5),
                local_epochs=_env_int("LOCAL_EPOCHS", 1),
                batch_size=_env_int("BATCH_SIZE", 256),
                lr=_env_float("LR", 1e-3),
                seed=_env_int("SEED", 42),
                hidden_size=_env_int("HIDDEN_SIZE", 64),
                layers=_env_int("LAYERS", 2),
                device=_env_str("DEVICE", "cpu"),
                cli_args={"role": "server", "source": "environment"},
                iid=iid,
                experiment_tag=_env_str("EXPERIMENT_TAG", "cicids"),
            )
            return

        start_fl_client(
            data_root=data_root,
            server_addr=_env_str("SERVER_ADDR", "server:8080"),
            client_id=None,
            seed=_env_int("SEED", 42),
            batch_size=_env_int("BATCH_SIZE", 256),
            lr=_env_float("LR", 1e-3),
            local_epochs=_env_int("LOCAL_EPOCHS", 1),
            hidden_size=_env_int("HIDDEN_SIZE", 64),
            layers=_env_int("LAYERS", 2),
            device=_env_str("DEVICE", "cpu"),
        )
        return

    if not args.command:
        parser.print_help()
        raise SystemExit(1)


    data_root = resolve_data_root(args.data_dir)
    runs_root = resolve_runs_root(args.runs_dir)

    # check prerequisites before running command
    check_prerequisites(data_root, args.command)

    if args.command == "preprocess":
        raw_path = args.data_path or _CICIDS_DEFAULT_PATHS
        data_paths = resolve_input_data_paths(raw_path)
        drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()] if args.drop_cols else []
        encode_cols = [c.strip() for c in args.encode_cols.split(",") if c.strip()] if args.encode_cols else []
        preprocess_dataset(
            data_paths=data_paths,
            label_col=args.label_col,
            binary=args.binary,
            data_root=data_root,
            sep=args.sep,
            drop_cols=drop_cols,
            encode_cols=encode_cols,
        )
        return

    if args.command == "partition":
        if args.iid and args.non_iid:
            raise ValueError("Use only one of --iid or --non-iid")
        iid = not args.non_iid
        create_partitions(
            data_root=data_root,
            label_col=args.label_col,
            clients=args.clients,
            test_size=args.test_size,
            seed=args.seed,
            iid=iid,
        )
        return

    if args.command == "central-train":
        train_and_evaluate_central(
            data_root=data_root,
            runs_root=runs_root,
            seed=args.seed,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            hidden_size=args.hidden_size,
            layers=args.layers,
            cli_args=vars(args),
            experiment_tag=args.experiment_tag,
        )
        return

    if args.command == "central-eval":
        evaluate_saved_central_model(
            data_root=data_root,
            runs_root=runs_root,
            model_path=Path(args.model_path) if args.model_path else None,
            seed=args.seed,
            batch_size=args.batch_size,
            device=args.device,
            cli_args=vars(args),
            experiment_tag=args.experiment_tag,
        )
        return

    if args.command == "fl-server":
        if args.iid and args.non_iid:
            raise ValueError("Use only one of --iid or --non-iid")
        iid = not args.non_iid
        start_fl_server(
            data_root=data_root,
            runs_root=runs_root,
            server_addr=args.server_addr,
            clients=args.clients,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            hidden_size=args.hidden_size,
            layers=args.layers,
            device=args.device,
            cli_args=vars(args),
            iid=iid,
            experiment_tag=args.experiment_tag,
        )
        return

    if args.command == "fl-client":
        start_fl_client(
            data_root=data_root,
            server_addr=args.server_addr,
            client_id=args.client_id,
            seed=args.seed,
            batch_size=args.batch_size,
            lr=args.lr,
            local_epochs=args.local_epochs,
            hidden_size=args.hidden_size,
            layers=args.layers,
            device=args.device,
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
