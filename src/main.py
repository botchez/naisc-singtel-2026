import argparse
import shutil
import sys
import os
import time

# Ensure src/ is on the path so sibling imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import polars as pl
from polars_compat import to_numeric_coerce, is_numeric_col, to_pandas_for_lgbm
from profiler import load_data, detect_feature_types, profile_data, align_train_test_columns, downcast_dataframe
from drift_detector import run_drift_analysis
from lightgbm import LGBMClassifier
from model import prepare_features, train_model, evaluate_and_save, print_model_performance_summary
from mitigation import apply_all_mitigations
from feature_engineer import engineer_features
from utils import Timer, logger


def print_drift_mitigation_summary(
    drift_df: pl.DataFrame,
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

    if len(drift_df) == 0:
        print(full_sep("╠", "╣"))
        print(full_row("No features analysed."))
        print(full_sep("╚", "╝"))
        return

    # Filter to only drifted features (high or moderate)
    drifted = drift_df.filter(pl.col("severity").is_in(["high", "moderate"]))

    if len(drifted) == 0:
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
    for r in drifted.iter_rows(named=True):
        col_name = r["column"][:CW[0] - 2]
        ftype = r["feature_type"]

        # Use qualitative description if available, fall back to metrics
        if "drift_description" in r and r.get("drift_description") is not None:
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


def print_runtime_summary(
    ingest_time: float,
    drift_time: float,
    mitigation_time: float,
    feat_time: float,
    model_time: float,
    total_elapsed: float = 0.0,
) -> None:
    """Print a separate runtime breakdown table using box-drawing characters."""
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
    print(col_row("Data Ingestion", f"{ingest_time:.2f}s"))
    print(col_row("Drift Detection", f"{drift_time:.2f}s"))
    print(col_row("Drift Mitigation", f"{mitigation_time:.2f}s"))
    print(col_row("Feature Engineering", f"{feat_time:.2f}s"))
    print(col_row("Model Training", f"{model_time:.2f}s"))
    print(h_line("╠", "╬", "╣"))
    print(col_row("Total", f"{total_elapsed:.2f}s"))
    print(h_line("╚", "╩", "╝"))


def main():
    parser = argparse.ArgumentParser(description="NAISC Singtel 2026 - Drift Detection & Churn Prediction")
    parser.add_argument("--train_data_filepath", required=True, help="Path to training CSV")
    parser.add_argument("--test_data_filepath", required=True, help="Path to test CSV")
    args = parser.parse_args()

    _pipeline_start = time.perf_counter()

    # --- Stage 1: Data Ingestion & Profiling ---
    with Timer("Data ingestion & profiling") as ingest_timer:
        train, test = load_data(args.train_data_filepath, args.test_data_filepath)
        train, test = align_train_test_columns(train, test)
        feature_types = detect_feature_types(train, test_df=test)
        profile_data(train, feature_types, label="train")
        profile_data(test, feature_types, label="test")

        # Coerce columns classified as numerical but still stored as non-numeric dtype
        for col in feature_types.get("numerical", []):
            if col in train.columns and not is_numeric_col(train[col]):
                train = train.with_columns(to_numeric_coerce(train[col]).alias(col))
            if col in test.columns and not is_numeric_col(test[col]):
                test = test.with_columns(to_numeric_coerce(test[col]).alias(col))

        # Downcast numeric columns now that they have proper dtypes (Float64 -> Float32, etc.)
        mem_before = (train.estimated_size() + test.estimated_size()) / 1e6
        train = downcast_dataframe(train)
        test = downcast_dataframe(test)
        mem_after = (train.estimated_size() + test.estimated_size()) / 1e6
        if mem_before > mem_after + 0.1:
            logger.info(f"Downcasted numeric columns: {mem_before:.1f}MB -> {mem_after:.1f}MB")

        # --- Build a single decision sample for drift detection and feature selection.
        # Full data flows through mitigation/FE/final model. ---
        from utils import stratified_subsample_idx, _parse_month_labels

        _DECISION_SAMPLE_SIZE = 100_000
        _decision_idx = None
        _month_vals = None
        for tc in feature_types.get("temporal", []):
            if tc in train.columns:
                _month_vals = _parse_month_labels(train[tc])
                if _month_vals is not None:
                    break

        if len(train) > _DECISION_SAMPLE_SIZE:
            _decision_idx = stratified_subsample_idx(
                len(train), _DECISION_SAMPLE_SIZE,
                month_values=_month_vals, seed=42,
            )
            logger.info(
                f"Decision sample: {_DECISION_SAMPLE_SIZE:,}/{len(train):,} rows"
                f"{' (stratified by temporal)' if _month_vals is not None else ''}"
            )
        train_decision = train[_decision_idx] if _decision_idx is not None else train

        # --- Dynamic pipeline cap: keep runtime under 10 minutes ---
        # Based on benchmark: 2M rows × 200 cols ≈ 230s computation.
        # Budget 480s for computation (leaving ~2 min for CSV I/O).
        _RUNTIME_BUDGET_S = 300
        _COST_PER_ROW_COL = 8.0e-7  # seconds per row × column (from 2M×200 on 2-core Docker)
        n_cols = len(train.columns)
        _max_pipeline_rows = int(_RUNTIME_BUDGET_S / (_COST_PER_ROW_COL * max(n_cols, 1)))
        _max_pipeline_rows = max(_max_pipeline_rows, _DECISION_SAMPLE_SIZE)  # never below decision sample

        if len(train) > _max_pipeline_rows:
            _original_train_len = len(train)
            _pipeline_idx = stratified_subsample_idx(
                len(train), _max_pipeline_rows,
                month_values=_month_vals, seed=42,
            )
            train = train[_pipeline_idx]
            # Recompute month values and decision index on the capped train
            # (original _decision_idx pointed into pre-cap rows, now invalid)
            _month_vals = None
            for tc in feature_types.get("temporal", []):
                if tc in train.columns:
                    _month_vals = _parse_month_labels(train[tc])
                    if _month_vals is not None:
                        break
            if len(train) > _DECISION_SAMPLE_SIZE:
                _decision_idx = stratified_subsample_idx(
                    len(train), _DECISION_SAMPLE_SIZE,
                    month_values=_month_vals, seed=42,
                )
            else:
                _decision_idx = None
            train_decision = train[_decision_idx] if _decision_idx is not None else train
            logger.info(
                f"Pipeline cap: {_max_pipeline_rows:,}/{_original_train_len:,} rows "
                f"(dynamic, targeting <{_RUNTIME_BUDGET_S}s for {n_cols} cols)"
            )
        else:
            logger.info(f"Pipeline cap: no cap needed ({len(train):,} rows × {n_cols} cols fits within budget)")

    # --- Stage 2: Drift Detection (train vs test) ---
    with Timer("Drift detection") as drift_timer:
        # Use decision sample for drift analysis — statistical tests converge at 100K
        test_drift = test.sample(n=min(_DECISION_SAMPLE_SIZE, len(test)), seed=42) if len(test) > _DECISION_SAMPLE_SIZE else test
        drift_df = run_drift_analysis(train_decision, test_drift, feature_types)

    # --- Stage 3: Drift Mitigation ---
    with Timer("Drift mitigation") as mitigation_timer:
        train_m, test_m, drop_cols, mitigation_log, norm_log = apply_all_mitigations(
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
                train_m = train_m.drop(col)
                test_m = test_m.drop(col)
                eng_dropped.append(col)
                drop_cols.append(col)
        if eng_dropped:
            logger.info(f"Dropped {len(eng_dropped)} high-drift engineered features: {eng_dropped}")

    # --- Stage 5: Model Training & Prediction ---
    with Timer("Model training") as model_timer:
        import numpy as np

        # Baseline model: raw features (no mitigation) — shows value of drift handling
        _MAX_BASELINE_SAMPLE = 100_000
        train_baseline = train.sample(n=min(_MAX_BASELINE_SAMPLE, len(train)), seed=42) if len(train) > _MAX_BASELINE_SAMPLE else train
        if len(train) > _MAX_BASELINE_SAMPLE:
            logger.info(f"Baseline model: subsampling {_MAX_BASELINE_SAMPLE}/{len(train)} rows")
        X_train_b, y_train_b, X_test_b, customer_ids = prepare_features(
            train_baseline, test, feature_types
        )
        lgbm_b = train_model(X_train_b, y_train_b)
        baseline_metrics = evaluate_and_save(
            lgbm_b, X_train_b, y_train_b, X_test_b, customer_ids, test, feature_types,
            tag="baseline",
        )

        # Mitigated model — prepare features with mitigation
        X_train_m, y_train_m, X_test_m, customer_ids = prepare_features(
            train_m, test_m, feature_types, extra_drop_cols=drop_cols
        )

        # --- Feature selection: use decision sample for importances ---
        from sklearn.metrics import average_precision_score
        from model import _encode_target

        try:
            if _decision_idx is not None and len(X_train_m) > _DECISION_SAMPLE_SIZE:
                fs_idx = _decision_idx[_decision_idx < len(X_train_m)]
                fs_model = train_model(X_train_m[fs_idx], y_train_m.gather(fs_idx.tolist()))
            else:
                fs_model = train_model(X_train_m, y_train_m)
            importances = fs_model.feature_importances_
            imp_by_col = dict(zip(X_train_m.columns, importances))

            # Drop zero-importance features
            zero_imp_cols = [c for c, imp in imp_by_col.items() if imp == 0]
            if zero_imp_cols and len(zero_imp_cols) < len(X_train_m.columns):
                X_train_m = X_train_m.drop(zero_imp_cols)
                X_test_m = X_test_m.drop(zero_imp_cols)
                logger.info(f"Dropped {len(zero_imp_cols)} zero-importance features: {zero_imp_cols}")
            elif zero_imp_cols:
                logger.warning(f"All {len(zero_imp_cols)} features have zero importance; keeping all")

            # Top-feature pairwise interactions (top-6 numeric by importance)
            remaining_imps = {c: imp_by_col[c] for c in X_train_m.columns if c in imp_by_col}
            sorted_cols = sorted(remaining_imps.keys(), key=lambda c: remaining_imps[c], reverse=True)
            top_cols = [c for c in sorted_cols[:8]
                        if is_numeric_col(X_train_m[c])]
            interaction_exprs_train = []
            interaction_exprs_test = []
            interaction_count = 0
            for i, ca in enumerate(top_cols):
                for cb in top_cols[i+1:]:
                    fname = f"{ca}_x_{cb}"
                    if fname not in X_train_m.columns:
                        interaction_exprs_train.append((pl.col(ca) * pl.col(cb)).alias(fname))
                        interaction_exprs_test.append((pl.col(ca) * pl.col(cb)).alias(fname))
                        interaction_count += 1
            if interaction_exprs_train:
                X_train_m = X_train_m.with_columns(interaction_exprs_train)
                X_test_m = X_test_m.with_columns(interaction_exprs_test)
            if interaction_count:
                logger.info(f"Added {interaction_count} top-feature interaction terms")
        except Exception as e:
            logger.warning(f"Feature selection skipped: {e}")

        from model import _native_cat_cols  # needed by Stage 5b self-training

        # --- Stage 5a: Iterative Mitigation (importance-aware) ---
        # Train a quick model to get importances, then re-mitigate features
        # that are both high-importance AND still have residual drift.
        # This focuses compute on what actually affects predictions.
        from drift_detector import detect_numerical_drift
        from mitigation import rank_transform, quantile_match
        try:
            _iter_model = train_model(X_train_m, y_train_m)
            _iter_imps = dict(zip(X_train_m.columns, _iter_model.feature_importances_))
            _nonzero_imps = [v for v in _iter_imps.values() if v > 0]
            _imp_threshold = np.percentile(_nonzero_imps, 50) if _nonzero_imps else 0

            # Check top important numeric features for residual drift
            _iter_sorted = sorted(_iter_imps.items(), key=lambda x: x[1], reverse=True)
            _iter_count = 0
            for col, imp in _iter_sorted:
                if imp < _imp_threshold:
                    break
                if col not in X_train_m.columns or col not in X_test_m.columns:
                    continue
                if not is_numeric_col(X_train_m[col]):
                    continue
                drift = detect_numerical_drift(X_train_m[col], X_test_m[col])
                if drift["severity"] in ("high", "moderate"):
                    # Try quantile match first (preserves scale), fall back to rank transform
                    tr_qm, te_qm = quantile_match(X_train_m[col], X_test_m[col])
                    qm_drift = detect_numerical_drift(tr_qm, te_qm)
                    if qm_drift["severity"] == "low":
                        X_train_m = X_train_m.with_columns(tr_qm.alias(col))
                        X_test_m = X_test_m.with_columns(te_qm.alias(col))
                        logger.info(f"  Iterative mitigation: {col} (imp={imp:.0f}, {drift['severity']}) → quantile match → low")
                    else:
                        tr_rk, te_rk = rank_transform(X_train_m[col], X_test_m[col])
                        X_train_m = X_train_m.with_columns(tr_rk.alias(col))
                        X_test_m = X_test_m.with_columns(te_rk.alias(col))
                        logger.info(f"  Iterative mitigation: {col} (imp={imp:.0f}, {drift['severity']}) → rank transform")
                    _iter_count += 1
            if _iter_count:
                logger.info(f"Iterative mitigation: re-mitigated {_iter_count} high-importance drifted features")
            else:
                logger.info("Iterative mitigation: no high-importance features with residual drift")
        except Exception as e:
            logger.warning(f"Iterative mitigation skipped: {e}")

        # Train final model on full data with uniform weights
        lgbm_m = train_model(X_train_m, y_train_m)

        # --- Stage 5b: Self-Training (pseudo-labeling) ---
        # Use high-confidence test predictions to augment training data,
        # adapting the model to the test distribution without needing test labels.
        _ST_CONF_HIGH = 0.95   # P(churn) above this → pseudo-label 1
        _ST_CONF_LOW  = 0.05   # P(churn) below this → pseudo-label 0
        _ST_MAX_ITERS = 1      # single iteration to avoid confirmation bias

        X_test_lgbm = to_pandas_for_lgbm(X_test_m) if isinstance(X_test_m, pl.DataFrame) else X_test_m
        for col in _native_cat_cols:
            if col in X_test_lgbm.columns:
                X_test_lgbm[col] = X_test_lgbm[col].astype("category")

        for st_iter in range(_ST_MAX_ITERS):
            test_proba = lgbm_m.predict_proba(X_test_lgbm)[:, 1]

            # Select high-confidence predictions
            confident_mask = (test_proba >= _ST_CONF_HIGH) | (test_proba <= _ST_CONF_LOW)
            n_pseudo = confident_mask.sum()

            if n_pseudo < 10:
                logger.info(f"Self-training iter {st_iter+1}: only {n_pseudo} confident samples, stopping")
                break

            pseudo_labels = (test_proba[confident_mask] >= 0.5).astype(np.int64)
            pseudo_X = X_test_m.filter(pl.Series(values=confident_mask))

            # Concatenate with original training data
            X_aug = pl.concat([X_train_m, pseudo_X], how="vertical_relaxed")
            y_aug = pl.concat([y_train_m, pl.Series(name=y_train_m.name, values=pseudo_labels)])

            # Retrain on augmented data
            lgbm_m = train_model(X_aug, y_aug)

            pos_pseudo = pseudo_labels.sum()
            logger.info(
                f"Self-training iter {st_iter+1}: {n_pseudo} pseudo-labeled samples "
                f"({pos_pseudo} pos, {n_pseudo - pos_pseudo} neg), "
                f"augmented train size: {len(X_aug)}"
            )

        mitigated_metrics = evaluate_and_save(
            lgbm_m, X_train_m, y_train_m, X_test_m, customer_ids, test_m, feature_types,
            tag="mitigated",
        )

        shutil.copy("prediction_mitigated.csv", "prediction.csv")
        shutil.copy("model_mitigated.joblib", "model.joblib")

    # --- All tables printed after all stages complete ---
    print_drift_mitigation_summary(drift_df, drop_cols, mitigation_log, norm_log)
    model_results = [
        ("Baseline", baseline_metrics),
        ("Mitigated", mitigated_metrics),
    ]
    print_model_performance_summary(model_results)
    _pipeline_elapsed = time.perf_counter() - _pipeline_start
    print_runtime_summary(ingest_timer.elapsed, drift_timer.elapsed, mitigation_timer.elapsed, feat_timer.elapsed, model_timer.elapsed, _pipeline_elapsed)


if __name__ == "__main__":
    main()
