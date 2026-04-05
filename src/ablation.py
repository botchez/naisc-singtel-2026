"""Quick ablation test: measure AU-PRC contribution of each pipeline stage.

Runs 5 configurations:
  1. Baseline       — raw data, no mitigation, no FE, uniform weights
  2. +Mitigation    — mitigation only, no FE, uniform weights
  3. +FE            — no mitigation, FE only, uniform weights
  4. +Weights       — no mitigation, no FE, tournament weights
  5. Full pipeline  — mitigation + FE + weights (current pipeline)

Usage:
  python src/ablation.py --train_data_filepath public_data/train.csv --test_data_filepath public_data/test.csv
"""
import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import polars as pl
import numpy as np
from sklearn.metrics import average_precision_score
from polars_compat import to_numeric_coerce, is_numeric_col, to_pandas_for_lgbm
from profiler import load_data, detect_feature_types, align_train_test_columns, downcast_dataframe
from drift_detector import run_drift_analysis, detect_intra_training_drift, detect_numerical_drift
from lightgbm import LGBMClassifier
import model as model_mod
from model import prepare_features, train_model, _to_pandas_with_categoricals, _encode_target
from mitigation import apply_all_mitigations, compute_domain_weights, compute_recency_weights, normalize_data_quality
from feature_engineer import engineer_features
from utils import logger, _parse_month_labels, stratified_subsample_idx


def _run_config(X_train, y_train, X_test, test, feature_types, sample_weight=None):
    """Train LightGBM, predict, return test AU-PRC. Must be called right after prepare_features."""
    # Capture _native_cat_cols immediately (set by the most recent prepare_features call)
    cat_cols = list(model_mod._native_cat_cols)

    mdl = train_model(X_train, y_train, sample_weight=sample_weight)
    X_test_pd = _to_pandas_with_categoricals(X_test, cat_cols)
    proba = mdl.predict_proba(X_test_pd)[:, 1]

    target_cols = feature_types.get("target", [])
    target_col = target_cols[0] if target_cols else None
    if target_col and target_col in test.columns:
        y_test = _encode_target(test[target_col])
        if y_test is not None and y_test.n_unique() >= 2:
            return average_precision_score(y_test.to_numpy(), proba)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_filepath", required=True)
    parser.add_argument("--test_data_filepath", required=True)
    args = parser.parse_args()

    # === Stage 1: Load & profile (shared) ===
    train_orig, test_orig = load_data(args.train_data_filepath, args.test_data_filepath)
    train_orig, test_orig = align_train_test_columns(train_orig, test_orig)
    feature_types = detect_feature_types(train_orig, test_df=test_orig)

    for col in feature_types.get("numerical", []):
        if col in train_orig.columns and not is_numeric_col(train_orig[col]):
            train_orig = train_orig.with_columns(to_numeric_coerce(train_orig[col]).alias(col))
        if col in test_orig.columns and not is_numeric_col(test_orig[col]):
            test_orig = test_orig.with_columns(to_numeric_coerce(test_orig[col]).alias(col))

    train_orig = downcast_dataframe(train_orig)
    test_orig = downcast_dataframe(test_orig)

    # Month values for recency
    _month_vals = None
    for tc in feature_types.get("temporal", []):
        if tc in train_orig.columns:
            _month_vals = _parse_month_labels(train_orig[tc])
            if _month_vals is not None:
                break

    # === Stage 2: Drift detection ===
    _DSS = 100_000
    _decision_idx = None
    if len(train_orig) > _DSS:
        _decision_idx = stratified_subsample_idx(len(train_orig), _DSS, month_values=_month_vals, seed=42)
    train_decision = train_orig[_decision_idx] if _decision_idx is not None else train_orig
    test_drift = test_orig.sample(n=min(_DSS, len(test_orig)), seed=42) if len(test_orig) > _DSS else test_orig

    drift_df = run_drift_analysis(train_decision, test_drift, feature_types)
    intra_drift_df, month_rank = detect_intra_training_drift(train_orig, feature_types)

    # === Stage 3: Mitigation ===
    train_m, test_m, drop_cols, mitigation_log, norm_log, shift_info = apply_all_mitigations(
        train_orig, test_orig, drift_df, feature_types
    )

    # === FE on mitigated data ===
    train_mfe, test_mfe, new_features = engineer_features(train_m, test_m, feature_types, drop_cols=drop_cols)
    eng_dropped = []
    for col in new_features:
        if col not in train_mfe.columns or col not in test_mfe.columns:
            continue
        drift = detect_numerical_drift(train_mfe[col], test_mfe[col])
        if drift["severity"] == "high":
            train_mfe = train_mfe.drop(col)
            test_mfe = test_mfe.drop(col)
            eng_dropped.append(col)
    drop_cols_full = drop_cols + eng_dropped

    # === FE on raw data ===
    train_fe_only, test_fe_only, new_features_raw = engineer_features(train_orig, test_orig, feature_types, drop_cols=[])
    eng_dropped_raw = []
    for col in new_features_raw:
        if col not in train_fe_only.columns or col not in test_fe_only.columns:
            continue
        drift = detect_numerical_drift(train_fe_only[col], test_fe_only[col])
        if drift["severity"] == "high":
            train_fe_only = train_fe_only.drop(col)
            test_fe_only = test_fe_only.drop(col)
            eng_dropped_raw.append(col)

    # === Weight helpers ===
    train_oof_proba = shift_info.get("train_oof_proba")
    has_domain = train_oof_proba is not None and shift_info.get("shift_detected", False)
    has_recency = month_rank is not None

    def _find_best_weights(X_train_pl, y_train_pl):
        """Mini weight tournament — returns (label, weights, val_score)."""
        cat_cols = list(model_mod._native_cat_cols)
        _X_pd = _to_pandas_with_categoricals(X_train_pl, cat_cols)
        _y_np = y_train_pl.to_numpy()

        _rng = np.random.RandomState(42)
        _tm = _month_vals.to_numpy(allow_copy=True) if isinstance(_month_vals, pl.Series) else _month_vals
        _n = len(_y_np)
        _val_mask = np.zeros(_n, dtype=bool)
        if _tm is not None:
            for g in np.unique(_tm):
                g_idx = np.where(_tm == g)[0]
                n_val_g = max(1, int(len(g_idx) * 0.2))
                chosen = _rng.choice(g_idx, size=n_val_g, replace=False)
                _val_mask[chosen] = True
        else:
            chosen = _rng.choice(_n, size=max(1, int(_n * 0.2)), replace=False)
            _val_mask[chosen] = True

        _tr = np.where(~_val_mask)[0]
        _va = np.where(_val_mask)[0]

        def _eval(label, w):
            ws = w[_tr] if w is not None else None
            m = LGBMClassifier(verbosity=-1, objective="binary", is_unbalance=True, random_state=42, importance_type="gain")
            m.fit(_X_pd.iloc[_tr], _y_np[_tr], sample_weight=ws)
            sc = average_precision_score(_y_np[_va], m.predict_proba(_X_pd.iloc[_va])[:, 1])
            return label, w, sc

        cands = [_eval("Uniform", None)]
        if has_domain and has_recency:
            for power in [0.3, 0.5, 0.7, 1.0]:
                w = compute_domain_weights(train_oof_proba, power=power)
                if w is not None:
                    cands.append(_eval(f"Domain(p={power})", w))
            for decay in [0.10, 0.20, 0.40]:
                rw = compute_recency_weights(month_rank, decay=decay)
                if rw is not None:
                    cands.append(_eval(f"Recency(d={decay})", rw))

            from scipy.optimize import minimize as _minimize
            def _build_blend(a, p, d):
                dw = compute_domain_weights(train_oof_proba, power=p)
                rw = compute_recency_weights(month_rank, decay=d)
                if dw is None or rw is None: return None
                b = a * dw + (1 - a) * rw
                return b / b.mean()

            _cache = {}
            _blend_evals = []
            def blend_obj(params):
                a, p, d = round(params[0], 4), round(params[1], 4), round(params[2], 4)
                if not (0 <= a <= 1 and 0.1 <= p <= 2.0 and 0.01 <= d <= 0.80): return 0.0
                key = (a, p, d)
                if key in _cache: return _cache[key]
                w = _build_blend(a, p, d)
                if w is None: return 0.0
                res = _eval(f"Blend(a={a},p={p},d={d})", w)
                _blend_evals.append(res)
                score = -res[2]
                _cache[key] = score
                return score

            _minimize(blend_obj, x0=[0.5, 0.7, 0.15], method='Nelder-Mead',
                      options={'maxiter': 8, 'maxfev': 12, 'xatol': 0.02, 'fatol': 0.0005})
            cands.extend(_blend_evals)

        cands.sort(key=lambda x: x[2], reverse=True)
        return cands[0]

    # === Normalization-only data (no progressive mitigation, no domain clf) ===
    train_norm, test_norm, _ = normalize_data_quality(train_orig, test_orig, feature_types)

    # ================================================================
    print("\n" + "=" * 70)
    print("ABLATION TEST - Stage Contribution Analysis")
    print("=" * 70)

    ablation = {}

    # 1. Baseline
    t0 = time.perf_counter()
    X_tr, y_tr, X_te, _ = prepare_features(train_orig, test_orig, feature_types)
    score = _run_config(X_tr, y_tr, X_te, test_orig, feature_types)
    ablation["1. Baseline (raw)"] = (score, time.perf_counter() - t0)
    print(f"  1. Baseline (raw):              AU-PRC = {score:.4f}")

    # 2a. Normalization only (abbreviations, lowercase, fuzzy match — no progressive mitigation)
    t0 = time.perf_counter()
    X_tr, y_tr, X_te, _ = prepare_features(train_norm, test_norm, feature_types)
    score = _run_config(X_tr, y_tr, X_te, test_orig, feature_types)
    ablation["2a. Normalization only"] = (score, time.perf_counter() - t0)
    print(f"  2a. Normalization only:         AU-PRC = {score:.4f}")

    # 2b. Full mitigation (normalization + progressive + feature drops)
    t0 = time.perf_counter()
    X_tr, y_tr, X_te, _ = prepare_features(train_m, test_m, feature_types, extra_drop_cols=drop_cols)
    score = _run_config(X_tr, y_tr, X_te, test_orig, feature_types)
    ablation["2b. Full mitigation"] = (score, time.perf_counter() - t0)
    print(f"  2b. Full mitigation:            AU-PRC = {score:.4f}")

    # 3. +FE only (no mitigation)
    t0 = time.perf_counter()
    X_tr, y_tr, X_te, _ = prepare_features(train_fe_only, test_fe_only, feature_types, extra_drop_cols=eng_dropped_raw)
    score = _run_config(X_tr, y_tr, X_te, test_orig, feature_types)
    ablation["3. FE only (no mitigation)"] = (score, time.perf_counter() - t0)
    print(f"  3. FE only (no mitigation):     AU-PRC = {score:.4f}")

    # 4. +Weights only (no mitigation)
    t0 = time.perf_counter()
    X_tr, y_tr, X_te, _ = prepare_features(train_orig, test_orig, feature_types)
    wlabel, wbest, _ = _find_best_weights(X_tr, y_tr)
    score = _run_config(X_tr, y_tr, X_te, test_orig, feature_types, sample_weight=wbest)
    ablation["4. Weights only (no mitig.)"] = (score, time.perf_counter() - t0)
    print(f"  4. Weights only (no mitig.):    AU-PRC = {score:.4f}  ({wlabel})")

    # 5. Mitigation + Weights (no FE)
    t0 = time.perf_counter()
    X_tr, y_tr, X_te, _ = prepare_features(train_m, test_m, feature_types, extra_drop_cols=drop_cols)
    wlabel, wbest, _ = _find_best_weights(X_tr, y_tr)
    score = _run_config(X_tr, y_tr, X_te, test_orig, feature_types, sample_weight=wbest)
    ablation["5. Mitigation + Weights"] = (score, time.perf_counter() - t0)
    print(f"  5. Mitigation + Weights:        AU-PRC = {score:.4f}  ({wlabel})")

    # 6. Mitigation + FE (no weights)
    t0 = time.perf_counter()
    X_tr, y_tr, X_te, _ = prepare_features(train_mfe, test_mfe, feature_types, extra_drop_cols=drop_cols_full)
    score = _run_config(X_tr, y_tr, X_te, test_orig, feature_types)
    ablation["6. Mitigation + FE"] = (score, time.perf_counter() - t0)
    print(f"  6. Mitigation + FE:             AU-PRC = {score:.4f}")

    # 7. Full pipeline (mitigation + FE + feature selection + interactions + weights)
    t0 = time.perf_counter()
    X_tr, y_tr, X_te, _ = prepare_features(train_mfe, test_mfe, feature_types, extra_drop_cols=drop_cols_full)

    # Feature selection + interactions (match main pipeline)
    try:
        fs_model = train_model(X_tr, y_tr)
        importances = fs_model.feature_importances_
        imp_by_col = dict(zip(X_tr.columns, importances))
        zero_imp = [c for c, imp in imp_by_col.items() if imp == 0]
        if zero_imp and len(zero_imp) < len(X_tr.columns):
            X_tr = X_tr.drop(zero_imp)
            X_te = X_te.drop(zero_imp)
        remaining = {c: imp_by_col[c] for c in X_tr.columns if c in imp_by_col}
        top_cols = [c for c in sorted(remaining, key=lambda c: remaining[c], reverse=True)[:8]
                    if is_numeric_col(X_tr[c])]
        int_exprs_tr, int_exprs_te = [], []
        for i, ca in enumerate(top_cols):
            for cb in top_cols[i+1:]:
                fn = f"{ca}_x_{cb}"
                if fn not in X_tr.columns:
                    int_exprs_tr.append((pl.col(ca) * pl.col(cb)).alias(fn))
                    int_exprs_te.append((pl.col(ca) * pl.col(cb)).alias(fn))
        if int_exprs_tr:
            X_tr = X_tr.with_columns(int_exprs_tr)
            X_te = X_te.with_columns(int_exprs_te)
    except Exception:
        pass

    wlabel, wbest, _ = _find_best_weights(X_tr, y_tr)
    score = _run_config(X_tr, y_tr, X_te, test_orig, feature_types, sample_weight=wbest)
    ablation["7. Full pipeline"] = (score, time.perf_counter() - t0)
    print(f"  7. Full pipeline:               AU-PRC = {score:.4f}  ({wlabel})")

    # Summary
    base = ablation["1. Baseline (raw)"][0]
    norm = ablation["2a. Normalization only"][0]
    mit = ablation["2b. Full mitigation"][0]
    print("\n" + "-" * 70)
    print("CONTRIBUTION (delta from baseline):")
    print("-" * 70)
    for name, (sc, elapsed) in ablation.items():
        delta = sc - base
        print(f"  {name:<35s} {sc:.4f}  ({'+' if delta >= 0 else ''}{delta:.4f})  [{elapsed:.1f}s]")
    print("-" * 70)
    print(f"\nMITIGATION BREAKDOWN:")
    print(f"  Normalization (abbrev/case/fuzzy):  +{norm - base:.4f}")
    print(f"  Progressive (stats-based):          +{mit - norm:.4f}")
    print(f"  Total mitigation:                   +{mit - base:.4f}")
    print()
    print(f"INCREMENTAL (on top of full mitigation):")
    print("-" * 70)
    for name in ["5. Mitigation + Weights", "6. Mitigation + FE", "7. Full pipeline"]:
        sc = ablation[name][0]
        delta = sc - mit
        print(f"  {name:<35s} {sc:.4f}  ({'+' if delta >= 0 else ''}{delta:.4f})")
    print("-" * 70)


if __name__ == "__main__":
    main()
