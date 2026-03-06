import argparse
import sys
import os

# Ensure src/ is on the path so sibling imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from profiler import load_data, detect_feature_types, profile_data
from drift_detector import run_drift_analysis, print_drift_summary
from model import prepare_features, train_model, evaluate_and_save
from utils import Timer


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

    # --- Stage 7: Baseline Model (no mitigation) ---
    with Timer("Model training & prediction") as model_timer:
        X_train, y_train, X_test, customer_ids = prepare_features(train, test, feature_types)
        lgbm = train_model(X_train, y_train)
        metrics = evaluate_and_save(lgbm, X_train, y_train, X_test, customer_ids, test, feature_types)

    # --- Stage 8: Runtime Summary ---
    total_time = drift_timer.elapsed + model_timer.elapsed
    print(f"\nRuntime: {total_time:.2f}s (drift: {drift_timer.elapsed:.2f}s, model: {model_timer.elapsed:.2f}s)")


if __name__ == "__main__":
    main()
