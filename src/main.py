import argparse
import sys
import os

# Ensure src/ is on the path so sibling imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from profiler import load_data, detect_feature_types, profile_data
from drift_detector import run_drift_analysis
from model import prepare_features, train_model, evaluate_and_save
from mitigation import apply_all_mitigations
from utils import Timer


def print_drift_mitigation_summary(
    drift_df: pd.DataFrame,
    drop_cols: list[str],
    drift_time: float,
    mitigation_time: float,
) -> None:
    """Print combined drift detection & mitigation summary."""
    print("\n" + "=" * 70)
    print("DATA DRIFT DETECTION & MITIGATION SUMMARY")
    print("=" * 70)

    if drift_df.empty:
        print("No features analysed.")
    else:
        # Columns with detected drift (moderate or high)
        drifted = drift_df[drift_df["severity"].isin(["high", "moderate"])]
        if not drifted.empty:
            print(f"\nColumns with detected drift ({len(drifted)}):")
            for _, row in drifted.iterrows():
                col = row["column"]
                ftype = row["feature_type"]
                severity = row["severity"]
                if ftype == "numerical":
                    tests = f"KS={row['ks_stat']:.4f} (p={row['ks_pval']:.4f}), PSI={row['psi']:.4f}, Wasserstein={row['wasserstein']:.4f}"
                else:
                    tests = f"Chi2={row['chi2_stat']:.4f} (p={row['chi2_pval']:.4f}), JSD={row['jsd']:.4f}"
                print(f"  {col:30s}  type={ftype:<12s}  severity={severity:<10s}  {tests}")
        else:
            print("\nNo significant drift detected.")

    # Mitigation methods applied
    print(f"\nMitigation methods applied:")
    print(f"  - Data quality normalization (city abbreviation mapping, lowercase, hyphen removal)")
    if drop_cols:
        print(f"  - Dropped {len(drop_cols)} high-drift features: {drop_cols}")
    else:
        print(f"  - No features dropped (no high-drift columns)")

    # Runtime
    total = drift_time + mitigation_time
    print(f"\nRuntime: {total:.2f}s (detection: {drift_time:.2f}s, mitigation: {mitigation_time:.2f}s)")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="NAISC Singtel 2026 - Drift Detection & Churn Prediction")
    parser.add_argument("--train_data_filepath", required=True, help="Path to training CSV")
    parser.add_argument("--test_data_filepath", required=True, help="Path to test CSV")
    args = parser.parse_args()

    # --- Stage 1: Data Ingestion & Profiling ---
    train, test = load_data(args.train_data_filepath, args.test_data_filepath)
    feature_types = detect_feature_types(train)
    profile_data(train, feature_types, label="train")
    profile_data(test, feature_types, label="test")

    # --- Stage 2: Drift Detection ---
    with Timer("Drift detection") as drift_timer:
        drift_df = run_drift_analysis(train, test, feature_types)

    # --- Stage 3: Drift Mitigation ---
    with Timer("Drift mitigation") as mitigation_timer:
        train_m, test_m, sample_weights, drop_cols = apply_all_mitigations(
            train, test, drift_df, feature_types
        )

    # --- Print Drift & Mitigation Summary ---
    print_drift_mitigation_summary(drift_df, drop_cols, drift_timer.elapsed, mitigation_timer.elapsed)

    # --- Stage 4: Model Training & Prediction ---
    with Timer("Model training") as model_timer:
        X_train, y_train, X_test, customer_ids = prepare_features(
            train_m, test_m, feature_types, extra_drop_cols=drop_cols
        )
        lgbm = train_model(X_train, y_train, sample_weight=sample_weights)
        metrics = evaluate_and_save(
            lgbm, X_train, y_train, X_test, customer_ids, test_m, feature_types,
        )

    # --- Runtime Summary ---
    total_time = drift_timer.elapsed + mitigation_timer.elapsed + model_timer.elapsed
    print(
        f"\nTotal runtime: {total_time:.2f}s "
        f"(drift: {drift_timer.elapsed:.2f}s, "
        f"mitigation: {mitigation_timer.elapsed:.2f}s, "
        f"model: {model_timer.elapsed:.2f}s)"
    )


if __name__ == "__main__":
    main()
