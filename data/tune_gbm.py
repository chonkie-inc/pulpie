"""Hyperparameter tuning for pulpie GBM classifier."""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(DATA_DIR, "training_data_dom.csv")
MODEL_PATH = os.path.join(DATA_DIR, "model_dom.txt")
FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.json")


def load_data():
    df = pd.read_csv(TRAIN_PATH)
    with open(FEATURES_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["features"]
    X = df[feature_cols].values
    y = df["label"].values
    return X, y, feature_cols


def tune():
    X, y, feature_cols = load_data()
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    configs = [
        # baseline
        {"num_leaves": 31, "learning_rate": 0.05, "min_data_in_leaf": 20,
         "max_depth": -1, "label": "baseline"},
        # more capacity
        {"num_leaves": 63, "learning_rate": 0.05, "min_data_in_leaf": 20,
         "max_depth": -1, "label": "leaves63"},
        # high capacity
        {"num_leaves": 127, "learning_rate": 0.05, "min_data_in_leaf": 20,
         "max_depth": -1, "label": "leaves127"},
        # high capacity + regularization
        {"num_leaves": 127, "learning_rate": 0.05, "min_data_in_leaf": 50,
         "max_depth": -1, "lambda_l1": 0.1, "lambda_l2": 1.0, "label": "leaves127_reg"},
        # high capacity + lower lr
        {"num_leaves": 127, "learning_rate": 0.02, "min_data_in_leaf": 30,
         "max_depth": -1, "label": "leaves127_lr02"},
        # very high capacity
        {"num_leaves": 255, "learning_rate": 0.05, "min_data_in_leaf": 50,
         "max_depth": -1, "label": "leaves255"},
        # deeper trees
        {"num_leaves": 127, "learning_rate": 0.05, "min_data_in_leaf": 20,
         "max_depth": 10, "label": "depth10_leaves127"},
    ]

    best_loss = float("inf")
    best_config = None
    best_rounds = 0

    for cfg in configs:
        label = cfg.pop("label")
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "scale_pos_weight": scale_pos_weight,
            **cfg,
        }

        dtrain = lgb.Dataset(X, label=y, feature_name=feature_cols)
        cv_result = lgb.cv(
            params, dtrain, num_boost_round=5000,
            nfold=5, stratified=True, seed=42,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )

        rounds = len(cv_result["valid binary_logloss-mean"])
        loss = cv_result["valid binary_logloss-mean"][-1]
        std = cv_result["valid binary_logloss-stdv"][-1]

        status = "*BEST*" if loss < best_loss else ""
        print(f"  {label:<25} rounds={rounds:>5}  loss={loss:.5f}±{std:.5f}  {status}", flush=True)

        if loss < best_loss:
            best_loss = loss
            best_config = {**params}
            best_rounds = rounds

        cfg["label"] = label  # restore

    print(f"\nBest config: loss={best_loss:.5f} rounds={best_rounds}")
    print(f"  {best_config}")

    # Train final model with best config
    print(f"\nTraining final model with {best_rounds} rounds...")
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_cols)
    model = lgb.train(best_config, dtrain, num_boost_round=best_rounds)

    # Evaluate
    y_pred = (model.predict(X) > 0.5).astype(int)
    print(f"\nTraining set:")
    print(classification_report(y, y_pred, target_names=["DISCARD", "KEEP"]))

    # Proper held-out eval via single fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        dtrain_fold = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        dval_fold = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain_fold)
        model_fold = lgb.train(
            best_config, dtrain_fold, num_boost_round=best_rounds,
            valid_sets=[dval_fold],
            callbacks=[lgb.log_evaluation(0)],
        )
        y_val_pred = (model_fold.predict(X_val) > 0.5).astype(int)
        print(f"Held-out fold validation:")
        print(classification_report(y_val, y_val_pred, target_names=["DISCARD", "KEEP"]))
        break  # just one fold for reporting

    # Save
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    # Update features json with best config info
    with open(FEATURES_PATH) as f:
        meta = json.load(f)
    meta["best_config"] = {k: v for k, v in best_config.items() if k != "verbosity"}
    meta["best_rounds"] = best_rounds
    meta["best_cv_loss"] = best_loss
    with open(FEATURES_PATH, "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    tune()
