import argparse
import shutil
import sys
import os

# Ensure src/ is on the path so sibling imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profiler import load_data, detect_feature_types, profile_data
from drift_detector import run_drift_analysis, print_drift_summary
from model import prepare_features, train_model, evaluate_and_save
from mitigation import apply_all_mitigations, compute_temporal_weights
from domain_classifier import compute_domain_weights, combine_weights
from drift_mitigator import apply_feature_mitigations, print_mitigation_summary
from feature_engineer import engineer_features
from utils import Timer
import numpy as np


def print_comparison_grid(results: dict[str, dict]) -> str:
    """Print side-by-side AU-PRC comparison for all variants. Returns best variant name."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON GRID")
    print("=" * 80)
    print(f"  {'Variant':<25s} {'Train AU-PRC':>12s} {'Test AU-PRC':>12s} {'Gap':>8s}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*8}")

    best_name = None
    best_test = -1.0

    for name, metrics in results.items():
        train = metrics.get("train_auprc")
        test = metrics.get("test_auprc")
        if train is not None and test is not None:
            gap = train - test
            print(f"  {name:<25s} {train:>12.4f} {test:>12.4f} {gap:>8.4f}")
            if test > best_test:
                best_test = test
                best_name = name
        elif train is not None:
            print(f"  {name:<25s} {train:>12.4f} {'N/A':>12s} {'':>8s}")

    print("=" * 80)
    return best_name


def run_variant(
    name: str,
    X_train: "pd.DataFrame",
    y_train: "pd.Series",
    X_test: "pd.DataFrame",
    customer_ids: list[str],
    test: "pd.DataFrame",
    feature_types: dict[str, list[str]],
    sample_weight: np.ndarray | None = None,
) -> dict[str, float]:
    """Train and evaluate a single model variant."""
    lgbm = train_model(X_train, y_train, sample_weight=sample_weight)
    metrics = evaluate_and_save(
        lgbm, X_train, y_train, X_test, customer_ids, test, feature_types,
        label=name.lower().replace(" ", "_"),
    )
    return metrics


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

    # --- Stages 2-3: Drift Detection ---
    with Timer("Drift detection") as drift_timer:
        drift_df = run_drift_analysis(train, test, feature_types)
    print_drift_summary(drift_df)

    # --- Stage 4: Data Quality Normalization ---
    with Timer("Data quality normalization") as norm_timer:
        train_m, test_m, temporal_weights, drop_cols = apply_all_mitigations(
            train, test, drift_df, feature_types
        )

    # --- Stage 5: Feature-Level Drift Mitigation ---
    with Timer("Feature mitigation") as mitigation_timer:
        train_m, test_m, action_log = apply_feature_mitigations(
            train_m, test_m, drift_df, feature_types
        )
    print_mitigation_summary(action_log)

    # --- Stage 6: Domain Classifier ---
    with Timer("Domain classifier") as domain_timer:
        domain_weights, domain_auc = compute_domain_weights(
            train_m, test_m, feature_types, drift_df=drift_df
        )

    # --- Stage 7: Feature Engineering ---
    with Timer("Feature engineering") as fe_timer:
        train_fe, test_fe = engineer_features(train_m, test_m, feature_types, drift_df=drift_df)

    # --- Stage 7: Prepare features for both paths ---
    with Timer("Feature preparation") as prep_timer:
        # Without feature engineering
        X_train_nf, y_train_nf, X_test_nf, customer_ids = prepare_features(
            train_m, test_m, feature_types, extra_drop_cols=drop_cols
        )
        # With feature engineering
        X_train_fe, y_train_fe, X_test_fe, _ = prepare_features(
            train_fe, test_fe, feature_types, extra_drop_cols=drop_cols
        )

    # --- Stage 8: Train all 8 variants ---
    combined_weights = combine_weights(temporal_weights, domain_weights, len(train))

    weight_configs = [
        ("Baseline", None),
        ("Temporal", temporal_weights),
        ("Domain", domain_weights),
        ("Combined", combined_weights),
    ]

    results = {}

    with Timer("All model variants") as model_timer:
        for name, weights in weight_configs:
            # Without FE
            results[name] = run_variant(
                name.lower(), X_train_nf, y_train_nf, X_test_nf,
                customer_ids, test, feature_types, sample_weight=weights,
            )
            # With FE
            fe_name = f"{name} + FE"
            results[fe_name] = run_variant(
                f"{name.lower()}_fe", X_train_fe, y_train_fe, X_test_fe,
                customer_ids, test, feature_types, sample_weight=weights,
            )

    # --- Stage 9: Comparison & Selection ---
    best_name = print_comparison_grid(results)

    # Map display names back to file labels
    label_map = {}
    for name, _ in weight_configs:
        label_map[name] = name.lower()
        label_map[f"{name} + FE"] = f"{name.lower()}_fe"

    if best_name:
        best_label = label_map.get(best_name, best_name.lower().replace(" ", "_"))
        shutil.copy(f"prediction_{best_label}.csv", "prediction.csv")
        shutil.copy(f"model_{best_label}.joblib", "model.joblib")
        print(f"\nBest model: {best_name.upper()} -> prediction.csv, model.joblib")

    # --- Runtime Summary ---
    total_time = (
        drift_timer.elapsed + norm_timer.elapsed + mitigation_timer.elapsed
        + domain_timer.elapsed + fe_timer.elapsed + prep_timer.elapsed
        + model_timer.elapsed
    )
    print(
        f"\nRuntime: {total_time:.2f}s "
        f"(drift: {drift_timer.elapsed:.2f}s, "
        f"normalization: {norm_timer.elapsed:.2f}s, "
        f"mitigation: {mitigation_timer.elapsed:.2f}s, "
        f"domain: {domain_timer.elapsed:.2f}s, "
        f"feature_eng: {fe_timer.elapsed:.2f}s, "
        f"prep: {prep_timer.elapsed:.2f}s, "
        f"models: {model_timer.elapsed:.2f}s)"
    )


if __name__ == "__main__":
    main()
