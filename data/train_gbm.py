"""Feature engineering + LightGBM training for pulpie block classifier.

Reads training_data.csv, does feature selection, trains LightGBM,
saves model + selected feature list.
"""

import json
import os
import sys

import numpy as np
import pandas as pd

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DATA_DIR, "training_data_dom.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model_dom.txt")
FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.json")

def load_data():
    df = pd.read_csv(TRAIN_PATH)
    print(f"Loaded {len(df)} blocks from {TRAIN_PATH}")
    print(f"Label distribution: {df['label'].value_counts().to_dict()}")
    return df


def feature_engineering(df):
    """Prune features: low variance, high correlation, low importance."""
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols]
    y = df["label"]

    print(f"\nStarting features: {len(feature_cols)}")

    # 1. Remove near-zero variance features (threshold: 0.01)
    variances = X.var()
    low_var = variances[variances < 0.01].index.tolist()
    if low_var:
        print(f"  Removing {len(low_var)} low-variance features: {low_var}")
        feature_cols = [c for c in feature_cols if c not in low_var]
        X = X[feature_cols]

    # 2. Remove highly correlated features (>0.95)
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        correlated = upper.index[upper[col] > 0.95].tolist()
        if correlated:
            # Keep the one with higher correlation to label
            for c in correlated:
                corr_with_label_c = abs(X[c].corr(y))
                corr_with_label_col = abs(X[col].corr(y))
                drop = c if corr_with_label_col >= corr_with_label_c else col
                to_drop.add(drop)

    if to_drop:
        print(f"  Removing {len(to_drop)} high-correlation features: {sorted(to_drop)}")
        feature_cols = [c for c in feature_cols if c not in to_drop]
        X = X[feature_cols]

    print(f"  After pruning: {len(feature_cols)} features")
    return feature_cols


def train_model(df, feature_cols):
    """Train LightGBM with cross-validation."""
    import lightgbm as lgb
    from sklearn.model_selection import StratifiedKFold

    X = df[feature_cols].values
    y = df["label"].values

    # Class weight for imbalance
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"\nClass balance: {n_neg} neg / {n_pos} pos (scale={scale_pos_weight:.2f})")

    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos_weight,
    }

    # Cross-validation to find best num_rounds
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_cols)
    cv_result = lgb.cv(
        params, dtrain, num_boost_round=3000,
        nfold=5, stratified=True, seed=42,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)],
    )

    best_round = len(cv_result["valid binary_logloss-mean"])
    best_loss = cv_result["valid binary_logloss-mean"][-1]
    print(f"\nCV best round: {best_round}, loss: {best_loss:.4f}")

    # Train final model on all data
    model = lgb.train(params, dtrain, num_boost_round=best_round)

    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importance("gain")))
    importance = dict(sorted(importance.items(), key=lambda x: -x[1]))
    print("\nFeature importance (gain):")
    for feat, imp in importance.items():
        print(f"  {feat:<30} {imp:.1f}")

    # Evaluate on training set (sanity check)
    from sklearn.metrics import classification_report
    y_pred = (model.predict(X) > 0.5).astype(int)
    print(f"\nTraining set classification report:")
    print(classification_report(y, y_pred, target_names=["DISCARD", "KEEP"]))

    return model, importance


def main():
    df = load_data()
    feature_cols = feature_engineering(df)
    model, importance = train_model(df, feature_cols)

    # Save model
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Save selected features
    with open(FEATURES_PATH, "w") as f:
        json.dump({
            "features": feature_cols,
            "importance": importance,
        }, f, indent=2)
    print(f"Feature list saved to {FEATURES_PATH}")


if __name__ == "__main__":
    main()
