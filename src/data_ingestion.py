import logging
from io import StringIO
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import pandas as pd
import requests

from src.config import (
    get_category_from_label,
    map_label_to_binary,
    map_label_to_multiclass,
    settings,
)

logger = logging.getLogger(__name__)


def download_nsl_kdd(url: str, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading NSL-KDD from {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    save_path.write_text(response.text)
    logger.info(f"Saved to {save_path}")


def load_raw_data(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path, header=None, names=settings.column_names)
    test_df = pd.read_csv(test_path, header=None, names=settings.column_names)
    logger.info(f"Loaded train: {train_df.shape}, test: {test_df.shape}")
    return train_df, test_df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop(columns=[settings.difficulty_column], errors="ignore")

    for col in settings.categorical_features:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.lower()

    for col in settings.numerical_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if settings.target_column in df.columns:
        df[settings.target_column] = df[settings.target_column].astype(str).str.strip().str.lower()

    df = df.dropna()
    logger.info(f"Cleaned data shape: {df.shape}")
    return df


def add_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["label_binary"] = df[settings.target_column].apply(map_label_to_binary)
    df["label_multiclass"] = df[settings.target_column].apply(map_label_to_multiclass)
    df["attack_category"] = df[settings.target_column].apply(get_category_from_label)
    return df


def split_data(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify_column: str = "label_multiclass"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        random_state=random_state,
        stratify=df[stratify_column]
    )

    val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=temp_df[stratify_column]
    )

    logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: Path
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(output_dir / "train.parquet", index=False)
    val_df.to_parquet(output_dir / "val.parquet", index=False)
    test_df.to_parquet(output_dir / "test.parquet", index=False)
    logger.info(f"Saved processed data to {output_dir}")


def load_processed_data(data_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_parquet(data_dir / "train.parquet")
    val_df = pd.read_parquet(data_dir / "val.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    return train_df, val_df, test_df


def prepare_features_targets(
    df: pd.DataFrame,
    target_type: Literal["binary", "multiclass"] = "multiclass"
) -> Tuple[pd.DataFrame, pd.Series]:
    target_col = "label_binary" if target_type == "binary" else "label_multiclass"
    X = df.drop(columns=[settings.target_column, "label_binary", "label_multiclass", "attack_category"], errors="ignore")
    y = df[target_col]
    return X, y


def ingest_pipeline(
    force_download: bool = False,
    data_dir: Optional[Path] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_dir = data_dir or settings.data_processed_dir
    raw_dir = settings.data_raw_dir

    train_file = raw_dir / "KDDTrain+.txt"
    test_file = raw_dir / "KDDTest+.txt"

    if force_download or not train_file.exists() or not test_file.exists():
        download_nsl_kdd(settings.nsl_kdd_train_url, train_file)
        download_nsl_kdd(settings.nsl_kdd_test_url, test_file)

    train_raw, test_raw = load_raw_data(train_file, test_file)
    combined = pd.concat([train_raw, test_raw], ignore_index=True)
    combined = clean_data(combined)
    combined = add_target_labels(combined)

    train_df, val_df, test_df = split_data(
        combined,
        train_ratio=settings.train_split,
        val_ratio=settings.val_split,
        test_ratio=settings.test_split,
        random_state=settings.random_state
    )

    save_processed_data(train_df, val_df, test_df, data_dir)
    return train_df, val_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_df, val_df, test_df = ingest_pipeline()
    print(f"Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")