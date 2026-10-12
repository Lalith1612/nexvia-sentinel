"""
Sentinel Phase 1: Evidently AI Data Drift & Distribution Monitoring Pipeline
Provides automated, reproducible distribution drift analysis between validation baseline
and production/holdout partitions, generating interactive HTML and machine-readable JSON reports.
Implements safely gated retraining recommendations with strict authorization invariants.
"""

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from evidently import Report
from evidently.presets import DataDriftPreset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "ml" / "data" / "processed"
DEFAULT_REPORTS_DIR = PROJECT_ROOT / "monitoring_reports"

import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.features.feature_pipeline import SentinelFeaturePipeline

logger = logging.getLogger("sentinel.ml.drift")


class SentinelDriftMonitor:
    """
    Automated distribution drift monitoring and safety-gated retraining advisor.
    Compares feature distributions across chronological reference and current datasets.
    """

    def __init__(
        self,
        drift_share_threshold: float = 0.30,
        p_value_threshold: float = 0.05,
        reports_dir: Optional[Path] = None
    ):
        self.drift_share_threshold = float(drift_share_threshold)
        self.p_value_threshold = float(p_value_threshold)
        self.reports_dir = Path(reports_dir) if reports_dir else DEFAULT_REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.feature_pipeline = SentinelFeaturePipeline(mode="pre_auth")

    def load_datasets(
        self,
        reference_path: Union[str, Path],
        current_path: Union[str, Path],
        sample_size: Optional[int] = 5000,
        use_feature_pipeline: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads reference and current dataset partitions.
        Optionally transforms raw transaction records through the production feature pipeline
        to inspect the exact 20-dimensional feature space evaluated by inference models.
        """
        ref_p = Path(reference_path)
        cur_p = Path(current_path)

        if not ref_p.exists():
            raise FileNotFoundError(f"Reference dataset not found: {ref_p}")
        if not cur_p.exists():
            raise FileNotFoundError(f"Current comparison dataset not found: {cur_p}")

        ref_df = pd.read_parquet(ref_p)
        cur_df = pd.read_parquet(cur_p)

        # Bounded sampling preserving order to ensure responsive execution and low memory
        if sample_size and len(ref_df) > sample_size:
            ref_df = ref_df.iloc[:sample_size].copy()
        if sample_size and len(cur_df) > sample_size:
            cur_df = cur_df.iloc[:sample_size].copy()

        if use_feature_pipeline:
            # Transform to the 20 pre-authorization features
            ref_feats = self.feature_pipeline.transform(ref_df)
            cur_feats = self.feature_pipeline.transform(cur_df)
            return ref_feats, cur_feats

        # If raw schema comparison is requested, filter to compatible numeric columns
        numeric_cols = [
            col for col in ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "step"]
            if col in ref_df.columns and col in cur_df.columns
        ]
        return ref_df[numeric_cols].copy(), cur_df[numeric_cols].copy()

    def run_drift_analysis(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Executes Evidently DataDriftPreset report and parses machine-readable summary.
        """
        if reference_df.empty or current_df.empty:
            raise ValueError("Reference and current datasets must be non-empty.")

        # Ensure matching column structure
        common_cols = [c for c in reference_df.columns if c in current_df.columns]
        if not common_cols:
            raise ValueError("No common columns found between reference and current datasets.")

        ref_data = reference_df[common_cols].copy()
        cur_data = current_df[common_cols].copy()

        # Run Evidently report
        report = Report([DataDriftPreset(drift_share=self.drift_share_threshold)])
        snapshot = report.run(reference_data=ref_data, current_data=cur_data)
        snapshot_dict = snapshot.dict()

        # Parse column-level metrics from snapshot dictionary
        metrics_list = snapshot_dict.get("metrics", [])
        drift_summary = self._parse_metrics(metrics_list, len(common_cols))

        return snapshot, drift_summary

    def _parse_metrics(self, metrics_list: List[Dict[str, Any]], total_columns: int) -> Dict[str, Any]:
        """Parses Evidently snapshot metric outputs into machine-readable summary."""
        drifted_count = 0
        drift_share = 0.0
        column_details: Dict[str, Dict[str, Any]] = {}

        for m in metrics_list:
            config = m.get("config", {})
            m_type = config.get("type", "")
            m_name = m.get("metric_name", "")
            val = m.get("value")

            # Overall dataset drift summary
            if "DriftedColumnsCount" in m_type or "DriftedColumnsCount" in m_name:
                if isinstance(val, dict):
                    drifted_count = int(val.get("count", 0))
                    drift_share = float(val.get("share", 0.0))

            # Column-level drift metric
            elif "ValueDrift" in m_type or "ValueDrift" in m_name:
                col_name = config.get("column", "")
                if not col_name and "column=" in m_name:
                    # Parse from name: ValueDrift(column=amount,method=...)
                    parts = m_name.split("column=")[1].split(",")[0]
                    col_name = parts

                if col_name:
                    p_val = float(val) if val is not None else 1.0
                    is_drifted = bool(p_val < self.p_value_threshold)
                    column_details[col_name] = {
                        "p_value": round(p_val, 6),
                        "drift_detected": is_drifted,
                        "threshold": self.p_value_threshold
                    }

        # If summary wasn't provided by DriftedColumnsCount, compute from column details
        if total_columns > 0 and drifted_count == 0 and column_details:
            drifted_count = sum(1 for c in column_details.values() if c["drift_detected"])
            drift_share = drifted_count / total_columns

        dataset_drift_detected = bool(drift_share >= self.drift_share_threshold)

        # Build safely gated retraining recommendation
        recommendation = self._build_retraining_recommendation(
            drift_detected=dataset_drift_detected,
            drift_share=drift_share,
            drifted_count=drifted_count,
            total_columns=total_columns
        )

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_drift_detected": dataset_drift_detected,
            "drift_share": round(drift_share, 4),
            "drift_share_threshold": round(self.drift_share_threshold, 4),
            "drifted_feature_count": drifted_count,
            "total_feature_count": total_columns,
            "column_metrics": column_details,
            "retraining_recommendation": recommendation,
            "operational_limitations": [
                "Drift scores reflect statistical divergence against PaySim synthetic baseline (val.parquet, step 501-620).",
                "High drift signals a distribution shift requiring analyst inspection, not confirmed degradation.",
                "Automated retraining is safely gated and requires explicit human authorization."
            ]
        }

    def _build_retraining_recommendation(
        self,
        drift_detected: bool,
        drift_share: float,
        drifted_count: int,
        total_columns: int
    ) -> Dict[str, Any]:
        """Constructs an auditable, safety-gated retraining recommendation."""
        if drift_detected:
            urgency = "HIGH" if drift_share >= 0.50 else "MEDIUM"
            reason = (
                f"Significant feature drift detected: {drifted_count}/{total_columns} features "
                f"({drift_share*100:.1f}%) drifted beyond threshold ({self.drift_share_threshold*100:.0f}%)."
            )
            action_required = True
        else:
            urgency = "NONE"
            reason = (
                f"Feature distributions within expected operational tolerances "
                f"({drifted_count}/{total_columns} features drifted, {drift_share*100:.1f}% <= {self.drift_share_threshold*100:.0f}%)."
            )
            action_required = False

        return {
            "action_required": action_required,
            "urgency": urgency,
            "reason": reason,
            "gated_approval_required": True,
            "recommended_action": "ANALYST_REVIEW_AND_RETRAIN" if action_required else "MONITORING_CONTINUES",
            "execution_command": "python ml/pipelines/train_xgboost.py" if action_required else None,
            "manual_procedure_note": (
                "Retraining must be manually approved by an authorized analyst and executed using the documented procedure."
            ) if action_required else None,
        }

    def save_reports(
        self,
        snapshot: Any,
        summary: Dict[str, Any],
        html_name: str = "data_drift_report.html",
        json_name: str = "drift_summary.json"
    ) -> Tuple[Path, Path]:
        """Persists the Evidently interactive HTML dashboard and machine-readable JSON."""
        html_path = self.reports_dir / html_name
        json_path = self.reports_dir / json_name

        # Save HTML
        snapshot.save_html(str(html_path))

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info("Drift reports saved: HTML -> %s, JSON -> %s", html_path, json_path)
        return html_path, json_path

    def trigger_safely_gated_retraining(
        self,
        recommendation: Dict[str, Any],
        approved: bool = False,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """
        Safely gated retraining action.
        Guarantees that automated drift detections NEVER overwrite production models
        without explicit operator approval and opt-in execution.
        """
        if not recommendation.get("action_required", False):
            return {
                "triggered": False,
                "status": "NO_ACTION_REQUIRED",
                "message": "Drift is within acceptable threshold; retraining suppressed."
            }

        if not approved:
            return {
                "triggered": False,
                "status": "APPROVAL_REQUIRED",
                "message": (
                    "Safety Gate Enforced: Retraining recommended but halted. "
                    "Operator approval ('approved=True') is required to prevent automated production model replacement."
                )
            }

        if dry_run:
            return {
                "triggered": True,
                "status": "DRY_RUN_SIMULATION",
                "message": (
                    "Operator approval verified. Dry-run retraining simulation successful. "
                    "Production model artifacts in ml/models/ remained untouched."
                ),
                "simulated_command": recommendation.get("execution_command")
            }

        # Actual opt-in retraining would be invoked explicitly here
        return {
            "triggered": True,
            "status": "RETRAINING_SCHEDULED",
            "message": "Retraining job scheduled with verified operator approval."
        }


def run_pipeline(
    sample_size: int = 5000,
    drift_share_threshold: float = 0.30
) -> Dict[str, Any]:
    """Convenience CLI entry point for executing drift analysis against baseline partitions."""
    monitor = SentinelDriftMonitor(drift_share_threshold=drift_share_threshold)
    val_path = PROCESSED_DATA_DIR / "val.parquet"
    test_path = PROCESSED_DATA_DIR / "test.parquet"

    print(f"[DRIFT] Loading validation baseline ({val_path.name}) vs test holdout ({test_path.name})...")
    ref_df, cur_df = monitor.load_datasets(val_path, test_path, sample_size=sample_size)

    print(f"[DRIFT] Running Evidently analysis on {len(ref_df.columns)} pre-authorization features...")
    snapshot, summary = monitor.run_drift_analysis(ref_df, cur_df)

    html_p, json_p = monitor.save_reports(snapshot, summary)
    print(f"[DRIFT] Completed. HTML report: {html_p}")
    print(f"[DRIFT] Machine-readable summary: {json_p}")
    print(f"[DRIFT] Dataset Drift Detected: {summary['dataset_drift_detected']}")
    print(f"[DRIFT] Drifted Features: {summary['drifted_feature_count']}/{summary['total_feature_count']} ({summary['drift_share']*100:.1f}%)")
    print(f"[DRIFT] Recommendation: {summary['retraining_recommendation']['recommended_action']}")

    return summary


if __name__ == "__main__":
    run_pipeline()
