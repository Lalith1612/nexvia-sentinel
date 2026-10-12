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

    # Replay configuration (Phase 5.3A)
    MAX_DIRECT_REPLAY_RECORDS: int = 1000
    MAX_DATASET_REPLAY_RECORDS: int = 5000
    TEST_DATASET_PATH: Path = DATA_DIR / "test.parquet"

    # Server execution
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    WORKERS: int = 1  # Exactly 1 worker for in-memory graph prototype
    CORS_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ]

    # Real-Time Kafka Streaming (Phase 6)
    KAFKA_ENABLED: bool = False
    KAFKA_BOOTSTRAP_SERVERS: str = "localhost:9092"
    KAFKA_TRANSACTIONS_TOPIC: str = "sentinel.transactions"
    KAFKA_RESULTS_TOPIC: str = "sentinel.fraud-predictions"
    KAFKA_ALERTS_TOPIC: str = "sentinel.fraud-alerts"
    KAFKA_DLQ_TOPIC: str = "sentinel.transactions.dlq"
    KAFKA_CONSUMER_GROUP: str = "sentinel-inference-group"
    KAFKA_AUTO_OFFSET_RESET: str = "latest"
    KAFKA_MAX_RETRIES: int = 3
    KAFKA_RETRY_BACKOFF_MS: int = 200
    KAFKA_POLL_TIMEOUT_SEC: float = 1.0
    USE_MOCK_STREAMING: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
