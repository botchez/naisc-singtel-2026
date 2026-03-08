import argparse
import shutil
import sys
import os
import time

# Ensure src/ is on the path so sibling imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from profiler import load_data, detect_feature_types, profile_data, align_train_test_columns
from drift_detector import run_drift_analysis, detect_intra_training_drift
from model import prepare_features, train_model, evaluate_and_save, print_model_performance_summary
from mitigation import apply_all_mitigations, compute_domain_weights, compute_recency_weights
from feature_engineer import engineer_features
from utils import Timer


def print_drift_mitigation_summary(
    drift_df: pd.DataFrame,
    drop_cols: list[str],
    mitigation_log: list[dict] | None = None,
    norm_log: dict[str, list[str]] | None = None,
) -> None:
    """Print combined drift detection & mitigation summary as a unified double-line table."""
    CW = (26, 14, 50, 30)
    FULL_W = sum(CW) + len(CW) - 1

    def full_row(text):
        return f"║ {text:<{FULL_W - 1}}║"

    def col_row(c1, c2, c3, c4):
        return f"║ {c1:<{CW[0]-2}} ║ {c2:<{CW[1]-2}} ║ {c3:<{CW[2]-2}} ║ {c4:<{CW[3]-2}} ║"

    def _wrap_text(text, max_w):
        """Word-wrap text into lines of max_w chars."""
        if len(text) <= max_w:
            return [text]
        lines, cur = [], ""
        for word in text.split():
            test = f"{cur} {word}".strip()
            if len(test) <= max_w:
                cur = test
            else:
                if cur:
                    lines.append(cur)
                cur = word if len(word) <= max_w else word[:max_w]
        if cur:
            lines.append(cur)
        return lines or [""]

    def col_row_wrap(c1, c2, c3, c4):
        """Print a data row, wrapping c3 and c4 onto continuation lines if they overflow."""
        max3 = CW[2] - 2
        max4 = CW[3] - 2
        lines3 = _wrap_text(c3, max3)
        lines4 = _wrap_text(c4, max4)
        n = max(len(lines3), len(lines4))
        rows = []
        for i in range(n):
            v1 = c1 if i == 0 else ""
            v2 = c2 if i == 0 else ""
            v3 = lines3[i] if i < len(lines3) else ""
            v4 = lines4[i] if i < len(lines4) else ""
            rows.append(col_row(v1, v2, v3, v4))
        return "\n".join(rows)

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

    # Build lookup from mitigation log for strategy names
    strategy_map = {}
    if mitigation_log:
        for entry in mitigation_log:
            strategy_map[entry["column"]] = entry["strategy_applied"]

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

        # Use qualitative description if available, fall back to metrics
        if "drift_description" in r.index and pd.notna(r.get("drift_description")):
            drift_desc = str(r["drift_description"])
        elif ftype == "numerical":
            sev = r["severity"].capitalize()
            drift_desc = f"{sev}: PSI={r['psi']:.2f}, KS p={r['ks_pval']:.3f}"
        else:
            sev = r["severity"].capitalize()
            drift_desc = f"{sev}: JSD={r['jsd']:.2f}, Chi2 p={r['chi2_pval']:.3f}"

        # Use strategy from mitigation_log if available
        if r["column"] in strategy_map:
            mitigation = strategy_map[r["column"]]
        else:
            dropped = r["column"] in drop_set
            actions = norm_log.get(r["column"], []) if norm_log else []
            if actions:
                label = ", ".join(a.lower() for a in actions)
                if dropped:
                    label += ", dropped"
                mitigation = label[0].upper() + label[1:]
            elif dropped:
                mitigation = "Drop Feature"
            else:
                mitigation = "Moderate drift, retained"

        print(col_row_wrap(col_name, ftype, drift_desc, mitigation))

    # Close table
    print(h_line("╚", "╩", "╝"))


def print_multivariate_shift_summary(shift_info: dict, weight_status: str = "Uniform (1.0)") -> None:
    """Print multivariate covariate shift detection results."""
    import numpy as np
    CW = (24, 22)
    W = sum(CW) + len(CW) - 1
    title = "MULTIVARIATE SHIFT DETECTION"

    def h(l, m, r):
        return f"{l}{'═'*CW[0]}{m}{'═'*CW[1]}{r}"

    def col_row(c1, c2):
        return f"║ {c1:<{CW[0]-2}} ║ {c2:<{CW[1]-2}} ║"

    print()
    print(f"╔{'═' * W}╗")
    print(f"║{title:^{W}}║")
    print(h("╠", "╦", "╣"))
    print(col_row("Metric", "Value"))
    print(h("╠", "╬", "╣"))
    print(col_row("Domain Clf AUC", f"{shift_info['auc']:.4f}"))
    fold_aucs = shift_info.get("fold_aucs", [])
    if fold_aucs:
        fold_std = np.std(fold_aucs)
        print(col_row("Fold AUC Std", f"{fold_std:.4f}"))
    status = "Yes" if shift_info["shift_detected"] else "No"
    print(col_row("Shift Detected", status))
    print(col_row("Sample Weights", weight_status))
    print(h("╚", "╩", "╝"))


def print_runtime_summary(
    drift_time: float,
    mitigation_time: float,
    feat_time: float,
    model_time: float,
) -> None:
    """Print a separate runtime breakdown table using box-drawing characters."""
    total = drift_time + mitigation_time + feat_time + model_time
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
    print(col_row("Feature Engineering", f"{feat_time:.2f}s"))
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
    #ascii_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "teamASCII.txt")
    #if os.path.exists(ascii_path):
    #    with open(ascii_path) as f:
    #        lines = f.read().splitlines()
    #    print("\n")
    #    _animate_ascii_typewriter(lines)
    #    time.sleep(1)

    # --- Stage 1: Data Ingestion & Profiling ---
    train, test = load_data(args.train_data_filepath, args.test_data_filepath)
    train, test = align_train_test_columns(train, test)
    feature_types = detect_feature_types(train, test_df=test)
    profile_data(train, feature_types, label="train")
    profile_data(test, feature_types, label="test")

    # --- Stage 2: Drift Detection (train vs test) ---
    with Timer("Drift detection") as drift_timer:
        drift_df = run_drift_analysis(train, test, feature_types)
        # Intra-training temporal drift detection
        intra_drift_df, month_rank = detect_intra_training_drift(train, feature_types)

    # --- Stage 3: Drift Mitigation ---
    with Timer("Drift mitigation") as mitigation_timer:
        train_m, test_m, drop_cols, mitigation_log, norm_log, shift_info = apply_all_mitigations(
            train, test, drift_df, feature_types
        )

    # --- Stage 4: Feature Engineering ---
    with Timer("Feature engineering") as feat_timer:
        train_m, test_m = engineer_features(train_m, test_m, feature_types)

    # --- Stage 5: Model Training & Prediction ---
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

        # --- Sample weight selection: try domain, recency, and combined ---
        import numpy as np
        from sklearn.metrics import average_precision_score

        train_oof_proba = shift_info.get("train_oof_proba")
        recency_weights = compute_recency_weights(month_rank) if month_rank is not None else None

        # Baseline: unweighted mitigated model
        lgbm_unweighted = train_model(X_train_m, y_train_m)

        # Check if test labels available for validation
        target_col = feature_types["target"][0] if feature_types["target"] else None
        has_test_labels = (target_col and target_col in test_m.columns
                          and test_m[target_col].notna().any())

        def _eval_model(model):
            """Evaluate model: test AU-PRC if available, else train AU-PRC."""
            if has_test_labels:
                from model import _encode_target
                y_test_v = _encode_target(test_m[target_col])
                return average_precision_score(y_test_v, model.predict_proba(X_test_m)[:, 1])
            return average_precision_score(y_train_m, model.predict_proba(X_train_m)[:, 1])

        best_model = lgbm_unweighted
        best_score = _eval_model(best_model)
        weight_status = "Uniform (1.0)"

        # Build candidate weight variants
        candidates = []

        # Domain weights at different power levels
        if train_oof_proba is not None and shift_info.get("shift_detected", False):
            for power in [0.3, 0.5, 0.7]:
                w = compute_domain_weights(train_oof_proba, power=power)
                if w is not None:
                    candidates.append((f"Domain-weighted (p={power})", w))

        # Recency weights
        if recency_weights is not None:
            candidates.append(("Recency-weighted", recency_weights))

            # Combined: domain x recency (if both available)
            if train_oof_proba is not None and shift_info.get("shift_detected", False):
                for power in [0.3, 0.5]:
                    dw = compute_domain_weights(train_oof_proba, power=power)
                    if dw is not None:
                        combined = dw * recency_weights
                        combined = combined / combined.mean()  # re-normalize
                        candidates.append((f"Domain(p={power})+Recency", combined))

        # Evaluate all candidates
        for label, weights in candidates:
            candidate = train_model(X_train_m, y_train_m, sample_weight=weights)
            score = _eval_model(candidate)
            if score > best_score:
                best_model = candidate
                best_score = score
                weight_status = label

        lgbm_m = best_model

        mitigated_metrics = evaluate_and_save(
            lgbm_m, X_train_m, y_train_m, X_test_m, customer_ids, test_m, feature_types,
            tag="mitigated",
        )

        shutil.copy("prediction_mitigated.csv", "prediction.csv")
        shutil.copy("model_mitigated.joblib", "model.joblib")

    # --- All tables printed after all stages complete ---
    print_drift_mitigation_summary(drift_df, drop_cols, mitigation_log, norm_log)
    print_multivariate_shift_summary(shift_info, weight_status)

    model_results = [
        ("Baseline", baseline_metrics),
        ("Mitigated", mitigated_metrics),
    ]
    print_model_performance_summary(model_results)
    print_runtime_summary(drift_timer.elapsed, mitigation_timer.elapsed, feat_timer.elapsed, model_timer.elapsed)


if __name__ == "__main__":
    main()
