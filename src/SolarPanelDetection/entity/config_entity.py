from pathlib import Path
from dataclasses import dataclass

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataPreparationConfig:
    features: list
    n_splits: int
    img_dir: Path
    mask_dir: Path
    dataframe_save_path: Path

@dataclass(frozen=True)
class TrainingConfig:
    features: list
    n_splits: int
    lgb_params: dict
    weights_dir: Path
    df_path: Path

@dataclass(frozen=True)
class EvaluationConfig:
    weights_dir: Path
    df_path: Path
    all_params: dict
    mlflow_uri: str

@dataclass(frozen=True)
class PredictionConfig:
    n_splits: int