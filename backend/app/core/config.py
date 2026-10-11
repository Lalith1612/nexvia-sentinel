"""
Sentinel Backend Configuration & Environment Settings
"""

from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Sentinel Real-Time Fraud Detection Platform"
    API_V1_STR: str = "/api/v1"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # Paths relative to repository root
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent
    ML_DIR: Path = PROJECT_ROOT / "ml"
    MODELS_DIR: Path = ML_DIR / "models"
    DATA_DIR: Path = ML_DIR / "data" / "processed"

    # Artifact paths
    FEATURE_PIPELINE_PATH: Path = MODELS_DIR / "feature_pipeline.joblib"
    XGB_MODEL_PATH: Path = MODELS_DIR / "xgboost_baseline.json"
    GNN_WEIGHTS_PATH: Path = MODELS_DIR / "gnn_graphsage.pt"
    GNN_CONFIG_PATH: Path = MODELS_DIR / "gnn_config.json"
    HYBRID_CONFIG_PATH: Path = MODELS_DIR / "hybrid_config.json"
    CALIBRATORS_PATH: Path = MODELS_DIR / "hybrid_calibrators.joblib"

    # Dataset paths for seeding historical graph snapshot
    TRAIN_SAMPLE_PATH: Path = DATA_DIR / "train_benchmark_sample.parquet"
    VAL_PATH: Path = DATA_DIR / "val.parquet"

    # Durable ledger SQLite database path
    LEDGER_DB_PATH: Path = PROJECT_ROOT / "backend" / "data" / "sentinel_ledger.db"

    # Operational defaults
    DEFAULT_WEIGHT_XGB: float = 0.78
    DEFAULT_WEIGHT_GNN: float = 0.22
    THRESHOLD_CRITICAL: float = 0.80
    THRESHOLD_HIGH: float = 0.65
    THRESHOLD_MEDIUM: float = 0.20
    THRESHOLD_LOW: float = 0.00

    # Server execution
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    WORKERS: int = 1  # Exactly 1 worker for in-memory graph prototype

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
