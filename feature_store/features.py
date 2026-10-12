"""
Sentinel Feast Feature Store Definitions (Phase 3)
Defines Feast Entity and FeatureView for financial account historical activity.
Guarantees strict pre-authorization scoring compliance (zero post-settlement or target leakage).
"""

from datetime import timedelta
from pathlib import Path
from feast import Entity, Field, FeatureView, FileSource, ValueType
from feast.types import Float32, Int64, String

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"

# 1. Entity: Unique Bank Account (Customer 'C...' or Merchant 'M...')
account_id = Entity(
    name="account_id",
    value_type=ValueType.STRING,
    join_keys=["account_id"],
    description="Primary account identifier for sender or recipient"
)

# 2. Offline Data Source: Parquet storage of pre-authorization rolling aggregates
account_features_source = FileSource(
    name="account_features_source",
    path=str(DATA_DIR / "account_features.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp"
)

# 3. Feature View: Account Rolling 24-Hour Behavioral Profile
account_hourly_stats_view = FeatureView(
    name="account_hourly_stats",
    entities=[account_id],
    ttl=timedelta(days=14),
    schema=[
        Field(name="rolling_avg_amount_24h", dtype=Float32, description="Rolling average transaction amount in past 24 hours"),
        Field(name="rolling_txn_count_24h", dtype=Int64, description="Transaction frequency count in past 24 hours"),
        Field(name="rolling_max_amount_24h", dtype=Float32, description="Maximum transaction amount in past 24 hours"),
        Field(name="zero_balance_frequency_24h", dtype=Float32, description="Proportion of past transactions starting with zero balance"),
        Field(name="merchant_interaction_ratio_24h", dtype=Float32, description="Proportion of transfers routed to merchants in past 24 hours"),
    ],
    online=True,
    source=account_features_source,
    tags={"team": "fraud-analytics", "tier": "tier-1-online"}
)
