from pathlib import Path

import numpy as np
import pandas as pd

from .utils import ensure_dir, save_json


CICIDS_ID_COLS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Timestamp",
]


def _class_distribution(values: pd.Series) -> dict[str, int]:
    # count samples per class
    counts = values.value_counts().sort_index()
    return {str(int(label)): int(count) for label, count in counts.items()}


def _to_binary_labels(label_series: pd.Series) -> pd.Series:
    # convert labels to 0 benign or 1 attack
    # handles CICIDS2017 string labels BENIGN=0, anything else=1
    # handles numeric labels 0=0, non-zero=1
    if pd.api.types.is_numeric_dtype(label_series):
        return (label_series.fillna(1).astype(float) != 0).astype(int)
    labels = label_series.astype(str).str.strip().str.lower()
    return (labels != "benign").astype(int)


def load_and_concat_csvs(paths: list[Path], sep: str = ",") -> pd.DataFrame:
    # load multiple CSV files strip whitespace from column names and concatenate
    dfs: list[pd.DataFrame] = []
    for p in paths:
        print(f"Loading: {p}")
        df = pd.read_csv(Path(p), sep=sep, low_memory=False)
        df.columns = df.columns.str.strip()
        print(f"  Shape: {df.shape}")
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Combined shape after concat: {combined.shape}")
    return combined


def preprocess_dataset(
    data_paths: list[Path],
    label_col: str,
    binary: bool,
    data_root: Path,
    sep: str = ",",
    drop_cols: list[str] | None = None,
    encode_cols: list[str] | None = None,
) -> dict:
    # clean and prepare dataset for training
    processed_dir = ensure_dir(data_root / "processed")

    # load all source CSVS and concatenate column names are stripped in loader
    df = load_and_concat_csvs(data_paths, sep=sep)
    df.columns = df.columns.str.strip()
    raw_shape = df.shape

    # check label column exists
    if label_col not in df.columns:
        raise ValueError(
            f"Label column '{label_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # drop unnamed index columns and explicit drop list
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    explicit_cols = [c for c in (drop_cols or []) if c in df.columns and c != label_col]
    cols_to_drop = list(dict.fromkeys(unnamed_cols + explicit_cols))
    if cols_to_drop:
        print(f"Dropping columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)

    # encode string columns as integers
    encoding_maps: dict[str, dict[str, int]] = {}
    for col in (encode_cols or []):
        if col not in df.columns or col == label_col:
            continue
        unique_vals = sorted(df[col].astype(str).unique())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        df[col] = df[col].astype(str).map(mapping).fillna(-1).astype(int)
        encoding_maps[col] = mapping
        print(f"Encoded '{col}': {mapping}")

    if encoding_maps:
        save_json(encoding_maps, processed_dir / "encoding_maps.json")

    # convert labels: BENIGN->0, all attacks->1
    label_values = df[label_col]
    if binary:
        y = _to_binary_labels(label_values)
    else:
        y = pd.to_numeric(label_values, errors="coerce")

    # retain only numeric feature columns
    feature_df = df.drop(columns=[label_col])
    numeric_df = feature_df.select_dtypes(include=[np.number]).copy()

    dropped_str_cols = [c for c in feature_df.columns if c not in numeric_df.columns]
    if dropped_str_cols:
        print(f"Dropping remaining non-numeric columns: {dropped_str_cols}")

    # replace inf/-inf with NaN then drop all rows with invalid values
    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    cleaned = pd.concat([numeric_df, y.rename(label_col)], axis=1)
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna(axis=0)

    rows_removed = raw_shape[0] - cleaned.shape[0]
    print(f"Rows removed due to inf/NaN: {rows_removed:,}")

    if binary:
        cleaned[label_col] = cleaned[label_col].astype(int)

    # save cleaned dataset
    cleaned_path = processed_dir / "cleaned.csv"
    cleaned.to_csv(cleaned_path, index=False)

    class_dist = _class_distribution(cleaned[label_col])
    print(f"Raw shape:     {raw_shape}")
    print(f"Cleaned shape: {cleaned.shape}")
    print(f"Features used ({len(numeric_df.columns)}): {list(numeric_df.columns)}")
    print(f"Class distribution (0=benign, 1=attack): {class_dist}")
    print(f"Saved cleaned dataset: {cleaned_path}")

    report = {
        "data_paths": [str(p) for p in data_paths],
        "label_column": label_col,
        "binary": bool(binary),
        "raw_shape": [int(raw_shape[0]), int(raw_shape[1])],
        "cleaned_shape": [int(cleaned.shape[0]), int(cleaned.shape[1])],
        "dropped_rows": int(rows_removed),
        "dropped_non_numeric_columns": int(len(dropped_str_cols)),
        "dropped_string_columns": dropped_str_cols,
        "encoded_columns": encoding_maps,
        "feature_count": int(len(numeric_df.columns)),
        "features": list(numeric_df.columns),
        "class_distribution": class_dist,
        "cleaned_file": str(cleaned_path),
    }

    save_json(report, processed_dir / "preprocessing_report.json")

    return report
