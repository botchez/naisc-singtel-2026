import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from utils import logger


def _prepare_domain_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
    drift_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Encode train+test into a numeric DataFrame for the domain classifier.

    Excludes temporal columns and high-JSD/high-severity categorical columns
    that trivially separate train from test (e.g. Month, Contract).
    """
    skip_cols = set(
        feature_types.get("id", [])
        + feature_types.get("target", [])
        + feature_types.get("temporal", [])
    )

    # Exclude categorical columns with extreme drift (JSD > 0.5 or unseen categories)
    if drift_df is not None and not drift_df.empty:
        for _, row in drift_df.iterrows():
            if row.get("severity") == "high" and row.get("feature_type") in ("categorical", "temporal"):
                jsd = row.get("jsd")
                unseen = row.get("unseen_in_test", "")
                if (jsd is not None and jsd > 0.1) or (unseen and len(str(unseen)) > 0):
                    skip_cols.add(row["column"])

    feature_cols = [c for c in train.columns if c not in skip_cols]

    combined = pd.concat(
        [train[feature_cols], test[feature_cols]], ignore_index=True
    )

    for col in combined.columns:
        if combined[col].dtype == object or not pd.api.types.is_numeric_dtype(combined[col]):
            le = LabelEncoder()
            combined[col] = le.fit_transform(combined[col].astype(str))
        combined[col] = combined[col].fillna(combined[col].median())

    return combined


def compute_domain_weights(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_types: dict[str, list[str]],
    drift_df: pd.DataFrame | None = None,
    n_splits: int = 5,
    clip_quantile: float = 0.99,
) -> tuple[np.ndarray, float]:
    """Train a domain classifier (train=0, test=1) and return per-row weights for training data.

    Uses cross-validation on the combined dataset to get out-of-fold predictions,
    avoiding overfitting. Training rows that look like test data get higher weight.

    Returns (weights, auc) where weights has len(train) and auc is the domain classifier AUC.
    """
    n_train = len(train)
    n_test = len(test)

    combined = _prepare_domain_features(train, test, feature_types, drift_df)
    labels = np.array([0] * n_train + [1] * n_test)

    oof_preds = np.zeros(len(combined))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(combined, labels)):
        clf = LGBMClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            verbosity=-1,
            random_state=42,
        )
        clf.fit(combined.iloc[train_idx], labels[train_idx])
        oof_preds[val_idx] = clf.predict_proba(combined.iloc[val_idx])[:, 1]

    auc = roc_auc_score(labels, oof_preds)

    # Extract predictions for training rows only
    train_preds = oof_preds[:n_train]

    # Use P(test) directly with power dampening to avoid extreme weights.
    # Raw odds ratio P/(1-P) can produce values like 0.002–5.4 which is too extreme.
    # Instead: use P(test)^dampening as a softer signal, then normalize.
    dampening = 0.5
    eps = 1e-6
    train_preds = np.clip(train_preds, eps, 1 - eps)
    raw_weights = np.power(train_preds, dampening)

    # Clip extreme weights at the specified quantile
    clip_val = np.quantile(raw_weights, clip_quantile)
    raw_weights = np.clip(raw_weights, None, clip_val)

    # Normalize so sum == len(train)
    weights = raw_weights * n_train / raw_weights.sum()

    logger.info(
        f"Domain classifier AUC: {auc:.4f} | "
        f"Weights: min={weights.min():.3f}, max={weights.max():.3f}, "
        f"mean={weights.mean():.3f}"
    )

    return weights, auc


def combine_weights(
    temporal_weights: np.ndarray | None,
    domain_weights: np.ndarray | None,
    n_train: int,
) -> np.ndarray:
    """Combine temporal and domain weights multiplicatively, then normalize."""
    if temporal_weights is not None and domain_weights is not None:
        combined = temporal_weights * domain_weights
    elif temporal_weights is not None:
        combined = temporal_weights.copy()
    elif domain_weights is not None:
        combined = domain_weights.copy()
    else:
        return np.ones(n_train)

    # Normalize so sum == n_train
    combined = combined * n_train / combined.sum()
    return combined
