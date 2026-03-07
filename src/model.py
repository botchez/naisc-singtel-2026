import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import LabelEncoder
import joblib
from utils import logger


def prepare_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
    extra_drop_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, list[str]]:
    """Minimal preprocessing: encode categoricals, drop IDs, split X/y.

    Returns (X_train, y_train, X_test, customer_ids).
    """
    target_col = feature_types["target"][0] if feature_types["target"] else "ChurnStatus"
    id_col = feature_types["id"][0] if feature_types["id"] else "CustomerID"

    customer_ids = test[id_col].tolist()

    # Encode target
    y_train = (train[target_col] == "Yes").astype(int)

    # Columns to use as features
    drop_cols = feature_types["id"] + feature_types["target"]
    if extra_drop_cols:
        drop_cols = drop_cols + [c for c in extra_drop_cols if c not in drop_cols]
    feature_cols = [c for c in train.columns if c not in drop_cols]

    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()

    # Convert temporal columns to month number (1-12)
    for col in feature_types["temporal"]:
        if col in feature_cols:
            for df in [X_train, X_test]:
                df[col] = pd.to_datetime(df[col], format="%y-%b", errors="coerce").dt.month
                df[col] = df[col].fillna(0).astype(int)

    # Encode categoricals with label encoding (fitted on train+test combined)
    cat_cols = [c for c in feature_types["categorical"] if c in feature_cols]
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined = pd.concat([X_train[col].astype(str), X_test[col].astype(str)], ignore_index=True)
        le.fit(combined)
        X_train[col] = le.transform(X_train[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        encoders[col] = le

    # Fill remaining NaN with median (numerical) or mode (categorical)
    for col in X_train.columns:
        if X_train[col].isna().any() or X_test[col].isna().any():
            if pd.api.types.is_numeric_dtype(X_train[col]):
                fill_val = X_train[col].median()
            else:
                fill_val = X_train[col].mode().iloc[0] if not X_train[col].mode().empty else 0
            X_train[col] = X_train[col].fillna(fill_val)
            X_test[col] = X_test[col].fillna(fill_val)

    logger.info(f"Features prepared: {X_train.shape[1]} columns")
    return X_train, y_train, X_test, customer_ids


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: np.ndarray | None = None,
) -> LGBMClassifier:
    """Train LightGBM with fixed hyperparameters."""
    model = LGBMClassifier(
        verbosity=-1,
        objective="binary",
        is_unbalance=True,
        random_state=42,
        importance_type="gain",
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)
    logger.info("Model trained")
    return model


def evaluate_and_save(
    model: LGBMClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    customer_ids: list[str],
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
    label: str = "baseline",
) -> dict[str, float]:
    """Predict, compute AU-PRC, save model and predictions."""
    # Train AU-PRC
    train_proba = model.predict_proba(X_train)[:, 1]
    train_auprc = average_precision_score(y_train, train_proba)

    # Test AU-PRC (only if target column exists in test)
    target_col = feature_types["target"][0] if feature_types["target"] else "ChurnStatus"
    test_proba = model.predict_proba(X_test)[:, 1]
    test_auprc = None
    if target_col in test.columns and test[target_col].notna().any():
        y_test = (test[target_col] == "Yes").astype(int)
        test_auprc = average_precision_score(y_test, test_proba)

    # Save prediction and model with label
    pred_file = f"prediction_{label}.csv"
    model_file = f"model_{label}.joblib"
    pred_df = pd.DataFrame({"CustomerID": customer_ids, "probability_score": test_proba})
    pred_df.to_csv(pred_file, index=False)
    logger.info(f"Saved {pred_file}")

    joblib.dump(model, model_file)
    logger.info(f"Saved {model_file}")

    # Print metrics
    metrics = {"train_auprc": train_auprc}
    print("\n" + "=" * 70)
    print(f"MODEL PERFORMANCE ({label.upper()})")
    print("=" * 70)
    print(f"  AU-PRC (train): {train_auprc:.4f}")
    if test_auprc is not None:
        print(f"  AU-PRC (test):  {test_auprc:.4f}")
        metrics["test_auprc"] = test_auprc
    else:
        print("  AU-PRC (test):  N/A (no labels in test set)")
    print("=" * 70)

    return metrics
