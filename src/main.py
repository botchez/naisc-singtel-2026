import argparse
import shutil
import sys
import os

# Ensure src/ is on the path so sibling imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profiler import load_data, detect_feature_types, profile_data
from drift_detector import run_drift_analysis, print_drift_summary
from model import prepare_features, train_model, evaluate_and_save
from mitigation import apply_all_mitigations
from utils import Timer


def print_comparison(baseline_metrics: dict, mitigated_metrics: dict) -> None:
    """Print side-by-side AU-PRC comparison table."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(f"  {'Metric':<25s} {'Baseline':>10s} {'Mitigated':>10s} {'Delta':>10s}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")

    for key, label in [("train_auprc", "AU-PRC (train)"), ("test_auprc", "AU-PRC (test)")]:
        b = baseline_metrics.get(key)
        m = mitigated_metrics.get(key)
        if b is not None and m is not None:
            delta = m - b
            sign = "+" if delta >= 0 else ""
            print(f"  {label:<25s} {b:>10.4f} {m:>10.4f} {sign}{delta:>9.4f}")
        elif b is not None:
            print(f"  {label:<25s} {b:>10.4f} {'N/A':>10s} {'':>10s}")

    # Generalization gap
    for name, metrics in [("Baseline", baseline_metrics), ("Mitigated", mitigated_metrics)]:
        if "train_auprc" in metrics and "test_auprc" in metrics:
            gap = metrics["train_auprc"] - metrics["test_auprc"]
            print(f"  {'Gap (' + name + ')':<25s} {gap:>10.4f}")

    print("=" * 70)


def select_best_model(baseline_metrics: dict, mitigated_metrics: dict) -> str:
    """Select the model with the best test AU-PRC (or train if test unavailable)."""
    b_test = baseline_metrics.get("test_auprc")
    m_test = mitigated_metrics.get("test_auprc")

    if b_test is not None and m_test is not None:
        winner = "mitigated" if m_test >= b_test else "baseline"
    elif m_test is not None:
        winner = "mitigated"
    else:
        winner = "baseline"

    # Copy winner's files to canonical names
    shutil.copy(f"prediction_{winner}.csv", "prediction.csv")
    shutil.copy(f"model_{winner}.joblib", "model.joblib")
    print(f"\nBest model: {winner.upper()} -> prediction.csv, model.joblib")
    return winner


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

    # --- Stages 2-3: Drift Detection (observation only) ---
    with Timer("Drift detection") as drift_timer:
        drift_df = run_drift_analysis(train, test, feature_types)
    print_drift_summary(drift_df)

    # --- Stages 4-6: Drift Mitigation ---
    with Timer("Drift mitigation") as mitigation_timer:
        train_m, test_m, sample_weights, drop_cols = apply_all_mitigations(
            train, test, drift_df, feature_types
        )

    # --- Stage 7a: Baseline Model (no mitigation) ---
    with Timer("Baseline model") as baseline_timer:
        X_train_b, y_train_b, X_test_b, customer_ids_b = prepare_features(
            train, test, feature_types
        )
        lgbm_b = train_model(X_train_b, y_train_b)
        baseline_metrics = evaluate_and_save(
            lgbm_b, X_train_b, y_train_b, X_test_b, customer_ids_b, test, feature_types,
            label="baseline",
        )

    # --- Stage 7b: Mitigated Model (cleaned data, weighted, temporal dropped) ---
    with Timer("Mitigated model") as mitigated_timer:
        X_train_m, y_train_m, X_test_m, customer_ids_m = prepare_features(
            train_m, test_m, feature_types, extra_drop_cols=drop_cols
        )
        lgbm_m = train_model(X_train_m, y_train_m, sample_weight=sample_weights)
        mitigated_metrics = evaluate_and_save(
            lgbm_m, X_train_m, y_train_m, X_test_m, customer_ids_m, test_m, feature_types,
            label="mitigated",
        )

    # --- Stage 8: Comparison & Selection ---
    print_comparison(baseline_metrics, mitigated_metrics)
    select_best_model(baseline_metrics, mitigated_metrics)

    # --- Runtime Summary ---
    total_time = (
        drift_timer.elapsed + mitigation_timer.elapsed
        + baseline_timer.elapsed + mitigated_timer.elapsed
    )
    print(
        f"\nRuntime: {total_time:.2f}s "
        f"(drift: {drift_timer.elapsed:.2f}s, "
        f"mitigation: {mitigation_timer.elapsed:.2f}s, "
        f"baseline: {baseline_timer.elapsed:.2f}s, "
        f"mitigated: {mitigated_timer.elapsed:.2f}s)"
    )


if __name__ == "__main__":
    main()
