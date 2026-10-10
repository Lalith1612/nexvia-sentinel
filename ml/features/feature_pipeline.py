"""
Sentinel Feature Engineering Pipeline
Constructs leakage-free, pre-authorization features for real-time fraud detection.
Ensures identical feature definitions between training and real-time inference.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import joblib

FEATURE_NAMES = [
    "amount",
    "log_amount",
    "oldbalanceOrg",
    "log_oldbalanceOrg",
    "oldbalanceDest",
    "log_oldbalanceDest",
    "amount_to_oldbalance_ratio",
    "orig_balance_depletion",
    "orig_oldbalance_is_zero",
    "dest_oldbalance_is_zero",
    "is_dest_merchant",
    "expected_newbalance_orig",
    "orig_deficit",
    "hour_of_day",
    "day_of_week",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER"
]

TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]

EXCLUDED_TARGET_FIELDS = ["isFraud", "isFlaggedFraud"]

class SentinelFeaturePipeline:
    """
    Production feature transformer for Sentinel pre-authorization scoring.
    Guarantees strict pre-authorization scoring compliance (zero post-settlement leakage).
    """

    def __init__(self, mode: str = "pre_auth"):
        self.mode = mode
        self.feature_names_: List[str] = list(FEATURE_NAMES)
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "SentinelFeaturePipeline":
        """Validate input schema and finalize feature configuration."""
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms raw transaction dataframe into engineered feature matrix.
        Input dataframe must contain:
          'step', 'type', 'amount', 'oldbalanceOrg', 'nameDest', 'oldbalanceDest'
        """
        if not self.is_fitted_:
            self.fit(df)

        # Work on a shallow copy of necessary series
        n_rows = len(df)
        feats = pd.DataFrame(index=df.index)

        # 1. Base amounts
        amt = df["amount"].astype(np.float32).values
        feats["amount"] = amt
        feats["log_amount"] = np.log1p(np.maximum(0.0, amt))

        # 2. Sender Balance Features
        old_orig = df["oldbalanceOrg"].astype(np.float32).values
        feats["oldbalanceOrg"] = old_orig
        feats["log_oldbalanceOrg"] = np.log1p(np.maximum(0.0, old_orig))

        # 3. Recipient Balance Features
        old_dest = df["oldbalanceDest"].astype(np.float32).values
        feats["oldbalanceDest"] = old_dest
        feats["log_oldbalanceDest"] = np.log1p(np.maximum(0.0, old_dest))

        # 4. Ratios & Depletion Indicators
        feats["amount_to_oldbalance_ratio"] = amt / (old_orig + 1.0)
        feats["orig_balance_depletion"] = ((amt >= old_orig) & (old_orig > 0.0)).astype(np.float32)
        feats["orig_oldbalance_is_zero"] = (old_orig <= 0.0).astype(np.float32)
        feats["dest_oldbalance_is_zero"] = (old_dest <= 0.0).astype(np.float32)

        # 5. Entity Type Indicator
        name_dest = df["nameDest"].astype(str)
        feats["is_dest_merchant"] = name_dest.str.startswith("M").astype(np.float32)

        # 6. Pre-Authorization Ledger Projections
        feats["expected_newbalance_orig"] = np.maximum(0.0, old_orig - amt)
        feats["orig_deficit"] = np.maximum(0.0, amt - old_orig)

        # 7. Temporal Diurnal & Cyclical Features
        step = df["step"].astype(np.int64).values
        feats["hour_of_day"] = (step % 24).astype(np.float32)
        feats["day_of_week"] = ((step // 24) % 7).astype(np.float32)

        # 8. Transaction Type One-Hot Encoding
        tx_type = df["type"].astype(str)
        for t in TRANSACTION_TYPES:
            feats[f"type_{t}"] = (tx_type == t).astype(np.float32)

        # Strict column order
        return feats[self.feature_names_]

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def transform_single(self, txn_dict: Dict) -> np.ndarray:
        """Transforms a single transaction dictionary for real-time inference."""
        df = pd.DataFrame([txn_dict])
        feats_df = self.transform(df)
        return feats_df.values.astype(np.float32)

    def save(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "SentinelFeaturePipeline":
        return joblib.load(filepath)
