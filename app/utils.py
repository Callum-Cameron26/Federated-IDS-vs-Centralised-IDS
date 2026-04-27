import json
import os
import random
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def ensure_dir(path: Path | str) -> Path:
    # create folder structure for saving files
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_json(data: dict, path: Path | str) -> None:
    # save metrics/configs to JSON file for later use
    path_obj = Path(path)
    ensure_dir(path_obj.parent)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path | str) -> dict:
    # load metrics/configs from JSON file
    path_obj = Path(path)
    with path_obj.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_data_root(data_dir: str | None = None) -> Path:
    # get data directory path for storing processed datasets
    if data_dir:
        return Path(data_dir)
    return Path(os.getenv("DATA_DIR", "data"))


def resolve_runs_root(runs_dir: str | None = None) -> Path:
    # get runs directory path for storing training results
    if runs_dir:
        return Path(runs_dir)
    return Path(os.getenv("RUNS_DIR", "runs"))


def resolve_input_data_paths(cli_data_path: str | None) -> list[Path]:
    # get list of raw CSV paths from comma-separated CLI arg or DATA_PATH env var
    raw = cli_data_path or os.getenv("DATA_PATH")
    if not raw:
        raise ValueError("Provide --data-path (comma-separated) or set DATA_PATH")
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


def timestamp_run_dir(runs_root: Path, prefix: str) -> Path:
    # create unique run folder with timestamp for this training run
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{prefix}_{stamp}"
    ensure_dir(run_dir)
    return run_dir


def copy_if_exists(src: Path, dst: Path) -> None:
    # copy file if it exists (used for reports)
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def copy_reports_to_run(data_root: Path, run_dir: Path) -> None:
    # copy preprocessing/partition reports to run folder for reference
    processed_dir = data_root / "processed"
    copy_if_exists(processed_dir / "preprocessing_report.json", run_dir / "preprocessing_report.json")
    copy_if_exists(processed_dir / "partition_report.json", run_dir / "partition_report.json")
