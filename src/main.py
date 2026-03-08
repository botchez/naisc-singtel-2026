import argparse
import shutil
import sys
import os
import time

# Ensure src/ is on the path so sibling imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from profiler import load_data, detect_feature_types, profile_data
from drift_detector import run_drift_analysis
from model import prepare_features, train_model, evaluate_and_save, print_model_performance_summary
from mitigation import apply_all_mitigations
from utils import Timer


def print_drift_mitigation_summary(
    drift_df: pd.DataFrame,
    drop_cols: list[str],
) -> None:
    """Print combined drift detection & mitigation summary as a unified double-line table."""
    CW = (24, 14, 34, 20)
    FULL_W = sum(CW) + len(CW) - 1  # 85

    def full_row(text):
        return f"║ {text:<{FULL_W - 1}}║"

    def col_row(c1, c2, c3, c4):
        return f"║ {c1:<{CW[0]-2}} ║ {c2:<{CW[1]-2}} ║ {c3:<{CW[2]-2}} ║ {c4:<{CW[3]-2}} ║"

    def h_line(left, mid, right):
        return f"{left}{'═'*CW[0]}{mid}{'═'*CW[1]}{mid}{'═'*CW[2]}{mid}{'═'*CW[3]}{right}"

    def full_sep(left, right):
        return f"{left}{'═' * FULL_W}{right}"

    title = "DATA DRIFT DETECTION & MITIGATION SUMMARY"

    print("\n")
    print(full_sep("╔", "╗"))
    print(f"║{title:^{FULL_W}}║")

    if drift_df.empty:
        print(full_sep("╠", "╣"))
        print(full_row("No features analysed."))
        print(full_sep("╚", "╝"))
        return

    # Filter to only drifted features (high or moderate)
    drifted = drift_df[drift_df["severity"].isin(["high", "moderate"])].copy()

    if drifted.empty:
        print(full_sep("╠", "╣"))
        print(full_row("No high or moderate drift detected across all features."))
        print(full_sep("╚", "╝"))
        return

    drop_set = set(drop_cols)

    # Title → header (introduces column dividers)
    print(h_line("╠", "╦", "╣"))
    print(col_row("Column", "Type", "Drift Description", "Mitigation"))
    # Header → data (crossing)
    print(h_line("╠", "╬", "╣"))

    # Data rows
    for _, r in drifted.iterrows():
        col_name = r["column"][:CW[0] - 2]
        ftype = r["feature_type"]
        sev = r["severity"].capitalize()

        if ftype == "numerical":
            drift_desc = f"{sev}: PSI={r['psi']:.2f}, KS p={r['ks_pval']:.3f}"
        else:
            drift_desc = f"{sev}: JSD={r['jsd']:.2f}, Chi2 p={r['chi2_pval']:.3f}"
        drift_desc = drift_desc[:CW[2] - 2]

        normalized = ftype in ("categorical", "temporal")
        dropped = r["column"] in drop_set
        if normalized and dropped:
            mitigation = "Normalized+Dropped"
        elif dropped:
            mitigation = "Dropped"
        elif normalized:
            mitigation = "Normalized"
        else:
            mitigation = "Retained"

        print(col_row(col_name, ftype, drift_desc, mitigation))

    # Close table
    print(h_line("╚", "╩", "╝"))


def print_runtime_summary(
    drift_time: float,
    mitigation_time: float,
    model_time: float,
) -> None:
    """Print a separate runtime breakdown table using box-drawing characters."""
    total = drift_time + mitigation_time + model_time
    CW = (24, 14)
    W = sum(CW) + len(CW) - 1

    def h_line(left, mid, right):
        return f"{left}{'═'*CW[0]}{mid}{'═'*CW[1]}{right}"

    def full_sep(left, right):
        return f"{left}{'═' * W}{right}"

    def col_row(c1, c2):
        return f"║ {c1:<{CW[0]-2}} ║ {c2:<{CW[1]-2}} ║"

    title = "RUNTIME SUMMARY"

    print()
    print(full_sep("╔", "╗"))
    print(f"║{title:^{W}}║")
    print(h_line("╠", "╦", "╣"))
    print(col_row("Stage", "Time"))
    print(h_line("╠", "╬", "╣"))
    print(col_row("Drift Detection", f"{drift_time:.2f}s"))
    print(col_row("Drift Mitigation", f"{mitigation_time:.2f}s"))
    print(col_row("Model Training", f"{model_time:.2f}s"))
    print(h_line("╠", "╬", "╣"))
    print(col_row("Total", f"{total:.2f}s"))
    print(h_line("╚", "╩", "╝"))


def _animate_ascii_typewriter(lines: list[str], duration: float = 1.5) -> None:
    """Animate ASCII art with a typewriter reveal effect, left-to-right."""
    num_lines = len(lines)
    max_width = max(len(line) for line in lines)
    cols_per_frame = max(1, max_width // 30)
    num_frames = (max_width + cols_per_frame - 1) // cols_per_frame
    delay = duration / num_frames

    # Print blank lines to reserve space
    for _ in lines:
        sys.stdout.write("\n")
    sys.stdout.flush()

    for frame in range(num_frames):
        reveal_col = (frame + 1) * cols_per_frame
        highlight_start = reveal_col - cols_per_frame
        # Move cursor up to overwrite
        sys.stdout.write(f"\033[{num_lines}A")
        for line in lines:
            settled = line[:highlight_start]
            highlighted = line[highlight_start:reveal_col]
            padding = " " * max(0, max_width - reveal_col)
            sys.stdout.write(
                f"\033[0m{settled}\033[32m{highlighted}\033[0m{padding}\n"
            )
        sys.stdout.flush()
        time.sleep(delay)

    # Final frame: full text, colors reset
    sys.stdout.write(f"\033[{num_lines}A")
    for line in lines:
        sys.stdout.write(f"\033[0m{line}\n")
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="NAISC Singtel 2026 - Drift Detection & Churn Prediction")
    parser.add_argument("--train_data_filepath", required=True, help="Path to training CSV")
    parser.add_argument("--test_data_filepath", required=True, help="Path to test CSV")
    args = parser.parse_args()

    # --- Splash banner ---
    ascii_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "teamASCII.txt")
    if os.path.exists(ascii_path):
        with open(ascii_path) as f:
            lines = f.read().splitlines()
        print("\n")
        _animate_ascii_typewriter(lines)
        time.sleep(1)

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

    # --- Stage 4: Model Training & Prediction ---
    with Timer("Model training") as model_timer:
        # Baseline model (no mitigation)
        X_train_b, y_train_b, X_test_b, customer_ids = prepare_features(
            train, test, feature_types
        )
        lgbm_b = train_model(X_train_b, y_train_b)
        baseline_metrics = evaluate_and_save(
            lgbm_b, X_train_b, y_train_b, X_test_b, customer_ids, test, feature_types,
            tag="baseline",
        )

        # Mitigated model
        X_train_m, y_train_m, X_test_m, customer_ids = prepare_features(
            train_m, test_m, feature_types, extra_drop_cols=drop_cols
        )
        lgbm_m = train_model(X_train_m, y_train_m, sample_weight=sample_weights)
        mitigated_metrics = evaluate_and_save(
            lgbm_m, X_train_m, y_train_m, X_test_m, customer_ids, test_m, feature_types,
            tag="mitigated",
        )

        # Copy mitigated as primary output
        shutil.copy("prediction_mitigated.csv", "prediction.csv")
        shutil.copy("model_mitigated.joblib", "model.joblib")

    # --- All tables printed after all stages complete ---
    print_drift_mitigation_summary(drift_df, drop_cols)
    print_model_performance_summary(baseline_metrics, mitigated_metrics)
    print_runtime_summary(drift_timer.elapsed, mitigation_timer.elapsed, model_timer.elapsed)


if __name__ == "__main__":
    main()
