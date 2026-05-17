"""Train GBM on combined WebMainBench + CC data.

Combines training_data_dom.csv (WebMainBench) with training_data_cc.csv (CC),
trains LightGBM with the same hyperparameters, and evaluates on WebMainBench
using ROUGE-5 F1 via the pulpie binary.
"""

import json
import os
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
WMB_PATH = os.path.join(DATA_DIR, "training_data_dom.csv")
CC_PATH = os.path.join(DATA_DIR, "training_data_cc.csv")
FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.json")
MODEL_PATH = os.path.join(DATA_DIR, "model_combined.txt")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cc-only', action='store_true', help='Train on CC data only')
    parser.add_argument('--wmb-only', action='store_true', help='Train on WebMainBench only (baseline)')
    parser.add_argument('--cv-rounds', action='store_true', help='Run CV to find best rounds instead of using 3083')
    args = parser.parse_args()

    # Load feature metadata
    with open(FEATURES_PATH) as f:
        meta = json.load(f)
    feature_cols = meta["features"]

    # Load datasets
    print("Loading datasets...", flush=True)
    df_wmb = pd.read_csv(WMB_PATH)
    print(f"  WebMainBench: {len(df_wmb):,} blocks ({df_wmb['label'].sum():,} keep / {(1 - df_wmb['label']).sum():,.0f} discard)", flush=True)

    if os.path.exists(CC_PATH):
        df_cc = pd.read_csv(CC_PATH)
        print(f"  CC:           {len(df_cc):,} blocks ({df_cc['label'].sum():,} keep / {(1 - df_cc['label']).sum():,.0f} discard)", flush=True)
    else:
        df_cc = pd.DataFrame()
        print("  CC: not found, skipping", flush=True)

    # Select data
    if args.cc_only:
        df = df_cc
        print(f"\n  Training on CC only: {len(df):,} blocks", flush=True)
    elif args.wmb_only:
        df = df_wmb
        print(f"\n  Training on WebMainBench only: {len(df):,} blocks", flush=True)
    else:
        df = pd.concat([df_wmb, df_cc], ignore_index=True)
        print(f"\n  Training on combined: {len(df):,} blocks", flush=True)

    # Ensure feature columns exist (CC may have extra columns not in selected features)
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  WARNING: Missing features: {missing}", flush=True)
        sys.exit(1)

    X = df[feature_cols].values
    y = df["label"].values

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"  KEEP: {n_pos:,} ({n_pos/len(y)*100:.1f}%)  DISCARD: {n_neg:,} ({n_neg/len(y)*100:.1f}%)", flush=True)
    print(f"  scale_pos_weight: {scale_pos_weight:.4f}", flush=True)

    config = {
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_leaves": 127,
        "learning_rate": 0.05,
        "min_data_in_leaf": 20,
        "max_depth": -1,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "scale_pos_weight": scale_pos_weight,
    }

    # Cross-validation to find best rounds (or use known best)
    if args.cv_rounds:
        print("\nRunning 5-fold CV to find best rounds...", flush=True)
        dtrain = lgb.Dataset(X, label=y, feature_name=feature_cols)
        cv_results = lgb.cv(
            config, dtrain, num_boost_round=10000,
            nfold=5, stratified=True, seed=42,
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
        )
        best_key = [k for k in cv_results.keys() if 'mean' in k and 'logloss' in k.lower()][0]
        best_rounds = len(cv_results[best_key])
        best_loss = min(cv_results[best_key])
        print(f"  Best rounds: {best_rounds}, CV loss: {best_loss:.5f}", flush=True)
    else:
        best_rounds = 3083
        print(f"\nUsing {best_rounds} rounds (from previous tuning)", flush=True)

    # Held-out evaluation (single fold)
    print("\nHeld-out fold evaluation:", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        dtrain_fold = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        dval_fold = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain_fold)
        model_fold = lgb.train(
            config, dtrain_fold, num_boost_round=best_rounds,
            valid_sets=[dval_fold],
            callbacks=[lgb.log_evaluation(0)],
        )
        y_val_pred = (model_fold.predict(X_val) > 0.5).astype(int)
        print(classification_report(y_val, y_val_pred, target_names=["DISCARD", "KEEP"]))
        break

    # Train final model on ALL data
    print(f"Training final model with {best_rounds} rounds on all data...", flush=True)
    dtrain = lgb.Dataset(X, label=y, feature_name=feature_cols)
    model = lgb.train(config, dtrain, num_boost_round=best_rounds)

    # Training set sanity check
    y_pred = (model.predict(X) > 0.5).astype(int)
    print("Training set:", flush=True)
    print(classification_report(y, y_pred, target_names=["DISCARD", "KEEP"]))

    # Save
    model.save_model(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}", flush=True)
    print(f"  Size: {os.path.getsize(MODEL_PATH) / 1024 / 1024:.1f} MB", flush=True)


if __name__ == "__main__":
    main()
