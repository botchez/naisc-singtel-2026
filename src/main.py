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
from drift_detector import run_drift_analysis, detect_intra_training_drift, detect_label_shift
from model import prepare_features, train_model, evaluate_and_save, print_model_performance_summary
from mitigation import apply_all_mitigations, compute_domain_weights, compute_recency_weights
from feature_engineer import engineer_features
from utils import Timer, logger


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
    refine_time: float = 0.0,
) -> None:
    """Print a separate runtime breakdown table using box-drawing characters."""
    total = drift_time + mitigation_time + feat_time + refine_time + model_time
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
    if refine_time > 0:
        print(col_row("Mitigation Refinement", f"{refine_time:.2f}s"))
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

    # --- Early subsampling: cap training data for pipeline efficiency ---
    # Subsample BEFORE numeric coercion to avoid O(10M) per column.
    # Stages 2-5 operate on subsampled data; final model caps at 500K anyway.
    _MAX_PIPELINE_TRAIN = 500_000
    if len(train) > _MAX_PIPELINE_TRAIN:
        from utils import stratified_subsample_idx, _parse_month_labels

        _original_n = len(train)
        # Try temporal stratification to preserve month distribution
        _month_vals = None
        for tc in feature_types.get("temporal", []):
            if tc in train.columns:
                _month_vals = _parse_month_labels(train[tc])
                if _month_vals is not None:
                    break

        _pipeline_idx = stratified_subsample_idx(
            len(train), _MAX_PIPELINE_TRAIN,
            month_values=_month_vals, seed=42,
        )
        train = train.iloc[_pipeline_idx].reset_index(drop=True)
        logger.info(
            f"Early subsampling: {_MAX_PIPELINE_TRAIN:,}/{_original_n:,} training rows"
            f"{' (stratified by temporal)' if _month_vals is not None else ''}"
        )

    # Coerce columns classified as numerical but still stored as object dtype
    # (after subsampling so this runs on 500K rows, not 10M)
    for col in feature_types.get("numerical", []):
        if col in train.columns and not pd.api.types.is_numeric_dtype(train[col]):
            train[col] = pd.to_numeric(train[col], errors="coerce")
        if col in test.columns and not pd.api.types.is_numeric_dtype(test[col]):
            test[col] = pd.to_numeric(test[col], errors="coerce")

    # --- Stage 2: Drift Detection (train vs test) ---
    with Timer("Drift detection") as drift_timer:
        # Pre-subsample to 50K for drift analysis — statistical tests cap at 50K
        # internally anyway; passing 500K just wastes time on isna/dropna/to_numeric
        _DRIFT_SAMPLE = 50_000
        train_drift = train.sample(n=min(_DRIFT_SAMPLE, len(train)), random_state=42) if len(train) > _DRIFT_SAMPLE else train
        test_drift = test.sample(n=min(_DRIFT_SAMPLE, len(test)), random_state=42) if len(test) > _DRIFT_SAMPLE else test
        drift_df = run_drift_analysis(train_drift, test_drift, feature_types)
        label_shift_info = detect_label_shift(train_drift, test_drift, feature_types)
        # Keep full train for month_rank (used for recency weighting across all rows)
        intra_drift_df, month_rank = detect_intra_training_drift(train, feature_types)

    # --- Stage 3: Drift Mitigation ---
    with Timer("Drift mitigation") as mitigation_timer:
        train_m, test_m, drop_cols, mitigation_log, norm_log, shift_info = apply_all_mitigations(
            train, test, drift_df, feature_types
        )

    # --- Stage 4: Feature Engineering ---
    with Timer("Feature engineering") as feat_timer:
        train_m, test_m, new_feature_names = engineer_features(train_m, test_m, feature_types, drop_cols=drop_cols)

        # Post-engineering drift check: drop high-drift engineered features
        from drift_detector import detect_numerical_drift
        eng_dropped = []
        for col in new_feature_names:
            if col not in train_m.columns or col not in test_m.columns:
                continue
            drift = detect_numerical_drift(train_m[col], test_m[col])
            if drift["severity"] == "high":
                train_m = train_m.drop(columns=[col])
                test_m = test_m.drop(columns=[col])
                eng_dropped.append(col)
                drop_cols.append(col)
        if eng_dropped:
            logger.info(f"Dropped {len(eng_dropped)} high-drift engineered features: {eng_dropped}")

    # --- Stage 4b: Mitigation Refinement (SKIPPED for performance) ---
    # Eliminated: was doing up to 160 model trains + 160 full DataFrame copies
    # Stage 3's progressive mitigation + Stage 5's weight tournament cover this.
    refine_timer = Timer("Mitigation refinement (skipped)")
    refine_timer.elapsed = 0.0

    # --- Stage 5: Model Training & Prediction ---
    with Timer("Model training") as model_timer:
        import numpy as np

        # Baseline model (subsampled for comparison metrics only)
        _MAX_BASELINE_SAMPLE = 50_000
        train_baseline = train
        if len(train) > _MAX_BASELINE_SAMPLE:
            train_baseline = train.sample(n=_MAX_BASELINE_SAMPLE, random_state=42)
            logger.info(f"Baseline model: subsampling {_MAX_BASELINE_SAMPLE}/{len(train)} rows")
        X_train_b, y_train_b, X_test_b, customer_ids = prepare_features(
            train_baseline, test, feature_types
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

        # --- Feature selection: use subsampled model to prune zero-importance features ---
        from sklearn.metrics import average_precision_score
        from model import _encode_target

        _MAX_FEAT_SEL_SAMPLE = 50_000
        try:
            if len(X_train_m) > _MAX_FEAT_SEL_SAMPLE:
                rng_fs = np.random.RandomState(42)
                fs_idx = rng_fs.choice(len(X_train_m), size=_MAX_FEAT_SEL_SAMPLE, replace=False)
                fs_model = train_model(X_train_m.iloc[fs_idx], y_train_m.iloc[fs_idx])
            else:
                fs_model = train_model(X_train_m, y_train_m)
            importances = fs_model.feature_importances_
            zero_imp_cols = [c for c, imp in zip(X_train_m.columns, importances) if imp == 0]
            if zero_imp_cols and len(zero_imp_cols) < len(X_train_m.columns):
                X_train_m = X_train_m.drop(columns=zero_imp_cols)
                X_test_m = X_test_m.drop(columns=zero_imp_cols)
                logger.info(f"Dropped {len(zero_imp_cols)} zero-importance features: {zero_imp_cols}")
            elif zero_imp_cols:
                logger.warning(f"All {len(zero_imp_cols)} features have zero importance; keeping all")
        except Exception as e:
            logger.warning(f"Feature selection skipped: {e}")

        # --- Sample weight tournament: two-stage search ---
        train_oof_proba = shift_info.get("train_oof_proba")
        has_domain = train_oof_proba is not None and shift_info.get("shift_detected", False)
        has_recency = month_rank is not None

        # Resolve test labels for validation-guided selection
        target_col = feature_types["target"][0] if feature_types["target"] else None
        has_test_labels = (target_col and target_col in test_m.columns
                          and test_m[target_col].notna().any())
        y_test_m = _encode_target(test_m[target_col]) if has_test_labels else None

        # Subsample for weight tournament evaluations on large datasets
        from utils import stratified_subsample_idx

        _MAX_TOURNEY_SAMPLE = 50_000
        _tourney_idx = None
        _test_tourney_idx = None

        if len(X_train_m) > _MAX_TOURNEY_SAMPLE:
            _mr_train = month_rank.reindex(X_train_m.index) if month_rank is not None else None
            _tourney_idx = stratified_subsample_idx(
                len(X_train_m), _MAX_TOURNEY_SAMPLE,
                month_values=_mr_train, seed=42,
            )
            logger.info(f"Weight tournament: stratified subsampling {_MAX_TOURNEY_SAMPLE}/{len(X_train_m)} train rows")

        if has_test_labels and len(X_test_m) > _MAX_TOURNEY_SAMPLE:
            _test_tourney_idx = stratified_subsample_idx(
                len(X_test_m), _MAX_TOURNEY_SAMPLE,
                month_values=None, seed=42,
            )
            logger.info(f"Weight tournament: subsampling {_MAX_TOURNEY_SAMPLE}/{len(X_test_m)} test rows")

        def _eval_candidate(label, weights):
            """Train with weights, return (label, weights, test_auprc, train_auprc)."""
            if _tourney_idx is not None:
                X_sub = X_train_m.iloc[_tourney_idx]
                y_sub = y_train_m.iloc[_tourney_idx]
                w_sub = weights[_tourney_idx] if weights is not None else None
            else:
                X_sub, y_sub, w_sub = X_train_m, y_train_m, weights
            m = train_model(X_sub, y_sub, sample_weight=w_sub)
            train_proba = m.predict_proba(X_sub)[:, 1]
            train_sc = average_precision_score(y_sub, train_proba)
            if has_test_labels:
                if _test_tourney_idx is not None:
                    X_test_sub = X_test_m.iloc[_test_tourney_idx]
                    y_test_sub = y_test_m.iloc[_test_tourney_idx]
                else:
                    X_test_sub, y_test_sub = X_test_m, y_test_m
                test_proba = m.predict_proba(X_test_sub)[:, 1]
                test_sc = average_precision_score(y_test_sub, test_proba)
            else:
                test_sc = train_sc  # fallback to train if no test labels
            logger.info(f"  {label} -> train={train_sc:.4f} test={test_sc:.4f}")
            return label, weights, test_sc, train_sc

        def _build_blend(alpha, power, decay):
            """Build additive blend: alpha * domain^power + (1-alpha) * recency(decay)."""
            dw = compute_domain_weights(train_oof_proba, power=power)
            rw = compute_recency_weights(month_rank, decay=decay)
            if dw is None or rw is None:
                return None
            blended = alpha * dw + (1 - alpha) * rw
            return blended / blended.mean()

        def _build_geometric(alpha, power, decay):
            """Build geometric mean: domain^(alpha*power) * recency(decay)^(1-alpha)."""
            dw = compute_domain_weights(train_oof_proba, power=power)
            rw = compute_recency_weights(month_rank, decay=decay)
            if dw is None or rw is None:
                return None
            combined = dw ** alpha * rw ** (1 - alpha)
            return combined / combined.mean()

        # ── Weight tournament via Nelder-Mead optimization ──
        from scipy.optimize import minimize

        all_results = []  # collect all evaluated candidates
        _eval_cache = {}  # cache rounded params → result to skip duplicate trains

        # 1. Uniform baseline
        res_uniform = _eval_candidate("Uniform", None)
        all_results.append(res_uniform)

        best_label, best_weights, best_score, _ = res_uniform

        if has_domain and has_recency:
            def _make_cached_objective(builder, prefix):
                """Create a cached Nelder-Mead objective for a weight family."""
                evals = []

                def objective(params):
                    a, p, d = params
                    if not (0.0 <= a <= 1.0 and 0.1 <= p <= 2.0 and 0.01 <= d <= 0.80):
                        return 0.0  # bad score for out-of-bounds
                    # Round to 4 decimals — use rounded values for BOTH cache and weights
                    a, p, d = round(a, 4), round(p, 4), round(d, 4)
                    key = (prefix, a, p, d)
                    if key in _eval_cache:
                        return _eval_cache[key]
                    w = builder(a, p, d)
                    if w is None:
                        return 0.0
                    lbl = f"{prefix}(a={a:.4f},p={p:.4f},d={d:.4f})"
                    res = _eval_candidate(lbl, w)
                    evals.append(res)
                    score = -res[2]  # minimize negative test AU-PRC
                    _eval_cache[key] = score
                    return score

                return objective, evals

            # 2. Optimize additive blend
            blend_obj, blend_evals = _make_cached_objective(_build_blend, "Blend")
            logger.info("Weight tournament: optimizing Blend (Nelder-Mead)")
            minimize(blend_obj, x0=[0.5, 0.7, 0.15], method='Nelder-Mead',
                     options={'maxiter': 8, 'maxfev': 12, 'xatol': 0.02, 'fatol': 0.0005})
            all_results.extend(blend_evals)

            # 3. Optimize geometric mean
            geom_obj, geom_evals = _make_cached_objective(_build_geometric, "Geometric")
            logger.info("Weight tournament: optimizing Geometric (Nelder-Mead)")
            minimize(geom_obj, x0=[0.5, 0.7, 0.15], method='Nelder-Mead',
                     options={'maxiter': 8, 'maxfev': 12, 'xatol': 0.02, 'fatol': 0.0005})
            all_results.extend(geom_evals)

        elif has_domain:
            # Only domain weights available — optimize power only
            for power in [0.3, 0.5, 0.7, 1.0]:
                w = compute_domain_weights(train_oof_proba, power=power)
                if w is not None:
                    all_results.append(_eval_candidate(f"Domain(p={power})", w))

        elif has_recency:
            # Only recency weights available — optimize decay only
            for decay in [0.10, 0.20, 0.40]:
                rw = compute_recency_weights(month_rank, decay=decay)
                if rw is not None:
                    all_results.append(_eval_candidate(f"Recency(d={decay})", rw))

        # Pick overall winner
        all_results.sort(key=lambda x: x[2], reverse=True)
        best_label, best_weights, best_score, _ = all_results[0]

        # Log full leaderboard
        logger.info(f"Weight tournament: {len(all_results)} total evaluations")
        logger.info("── Full leaderboard (top 20) ──")
        for i, (lbl, _, test_sc, train_sc) in enumerate(all_results[:20]):
            logger.info(f"  #{i+1:2d}  {test_sc:.4f} test | {train_sc:.4f} train | {lbl}")

        logger.info(f"Final winner: {best_label} (test AU-PRC={best_score:.4f})")
        weight_status = best_label

        # Train final model with winning weights (stratified subsample on large datasets)
        _MAX_FINAL_TRAIN = 500_000
        if len(X_train_m) > _MAX_FINAL_TRAIN:
            _mr_final = month_rank.reindex(X_train_m.index) if month_rank is not None else None
            _final_idx = stratified_subsample_idx(
                len(X_train_m), _MAX_FINAL_TRAIN,
                month_values=_mr_final, seed=42,
            )
            X_train_final = X_train_m.iloc[_final_idx]
            y_train_final = y_train_m.iloc[_final_idx]
            w_final = best_weights[_final_idx] if best_weights is not None else None
            logger.info(f"Final model: stratified subsampling {_MAX_FINAL_TRAIN}/{len(X_train_m)} rows")
        else:
            X_train_final, y_train_final = X_train_m, y_train_m
            w_final = best_weights

        lgbm_m = train_model(X_train_final, y_train_final, sample_weight=w_final)

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
    print_runtime_summary(drift_timer.elapsed, mitigation_timer.elapsed, feat_timer.elapsed, model_timer.elapsed, refine_timer.elapsed)


if __name__ == "__main__":
    main()
