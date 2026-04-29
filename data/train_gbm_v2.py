"""Train GBM: 30 features (26 original + 4 ancestor booleans), max 1000 trees.

Strategy: manual 5-fold CV with train() instead of lgb.cv to avoid
callback issues. Test 4 configs, pick best, train final model.

Usage:
  python train_gbm_v2.py              # train on WMB (default)
  python train_gbm_v2.py --cc         # train on CC data
  python train_gbm_v2.py --combined   # train on WMB + CC
"""

import argparse
import json
import os
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, classification_report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES_PATH = os.path.join(DATA_DIR, "selected_features.json")

TOP_FEATURES = [
    "link_ratio",
    "page_total_blocks",
    "parent_tag_type",
    "page_total_link_ratio",
    "tag_type",
    "section_block_count",
    "page_total_text_len",
    "dom_depth",
    "section_link_density",
    "position",
    "tag_type_score",
    "section_heading_text_len",
    "blocks_since_heading",
    "word_count",
    "next_block_link_ratio",
    "prev_block_link_ratio",
    "text_to_tag_ratio",
    "semantic_ancestor",
    "blocks_until_heading",
    "page_heading_count",
    "parent_class_id_score",
    "next_block_text_len",
    "link_count",
    "text_len",
    "tag_count",
    "avg_word_length",
    "in_nav",
    "in_footer",
    "in_aside",
    "in_header",
]

MAX_ROUNDS = 1000


def evaluate_config(X, y, feature_cols, params, label):
    """Manual 5-fold CV with early stopping."""
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_losses = []
    fold_rounds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)
        dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=dtrain, free_raw_data=False)

        model = lgb.train(
            params, dtrain, num_boost_round=MAX_ROUNDS,
            valid_sets=[dval], valid_names=['val'],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )

        fold_losses.append(model.best_score['val']['binary_logloss'])
        fold_rounds.append(model.best_iteration)

    avg_loss = np.mean(fold_losses)
    avg_rounds = int(np.mean(fold_rounds))
    std_loss = np.std(fold_losses)
    return avg_loss, std_loss, avg_rounds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cc", action="store_true", help="Train on CC data only")
    parser.add_argument("--combined", action="store_true", help="Train on WMB + CC")
    args = parser.parse_args()

    if args.cc:
        train_path = os.path.join(DATA_DIR, "training_data_cc.csv")
        model_path = os.path.join(DATA_DIR, "model_dom_v3_cc.txt")
    elif args.combined:
        train_path = os.path.join(DATA_DIR, "training_data_cc.csv")
        model_path = os.path.join(DATA_DIR, "model_dom_v3_combined.txt")
    else:
        train_path = os.path.join(DATA_DIR, "training_data_dom.csv")
        model_path = os.path.join(DATA_DIR, "model_dom_v3.txt")

    t_start = time.time()
    print(f"Loading {train_path}...", flush=True)
    df = pd.read_csv(train_path)

    if args.combined:
        wmb_path = os.path.join(DATA_DIR, "training_data_dom.csv")
        print(f"Loading {wmb_path}...", flush=True)
        df_wmb = pd.read_csv(wmb_path)
        df = pd.concat([df, df_wmb], ignore_index=True)

    X = df[TOP_FEATURES].values
    y = df["label"].values
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"  {len(y)} blocks, {n_pos} keep / {n_neg} discard", flush=True)
    print(f"  {len(TOP_FEATURES)} features", flush=True)

    configs = [
        {"num_leaves": 255, "learning_rate": 0.1, "min_data_in_leaf": 20,
         "max_depth": -1, "label": "lr0.1_l255"},
        {"num_leaves": 255, "learning_rate": 0.2, "min_data_in_leaf": 20,
         "max_depth": -1, "label": "lr0.2_l255"},
        {"num_leaves": 511, "learning_rate": 0.15, "min_data_in_leaf": 30,
         "max_depth": -1, "label": "lr0.15_l511"},
        {"num_leaves": 511, "learning_rate": 0.2, "min_data_in_leaf": 30,
         "max_depth": -1, "label": "lr0.2_l511"},
    ]

    best_loss = float("inf")
    best_config = None
    best_rounds = 0
    best_label = ""

    print(f"\n{'='*70}", flush=True)
    print(f"TUNING (5-fold CV, max {MAX_ROUNDS} rounds, early stop 100)", flush=True)
    print(f"{'='*70}", flush=True)

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
            "num_threads": -1,
            **cfg,
        }

        t0 = time.time()
        avg_loss, std_loss, avg_rounds = evaluate_config(X, y, TOP_FEATURES, params, label)
        elapsed = time.time() - t0

        status = "*BEST*" if avg_loss < best_loss else ""
        print(f"  {label:<20} rounds={avg_rounds:>4}  loss={avg_loss:.5f}±{std_loss:.5f}  "
              f"({elapsed:.0f}s)  {status}", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_config = {**params}
            best_rounds = avg_rounds
            best_label = label

        cfg["label"] = label

    print(f"\nBest: {best_label} — loss={best_loss:.5f}, rounds={best_rounds}", flush=True)

    # Train final model on ALL data
    print(f"\nTraining final model: {best_rounds} rounds on all {len(y)} blocks...", flush=True)
    dtrain = lgb.Dataset(X, label=y, feature_name=TOP_FEATURES)
    model = lgb.train(best_config, dtrain, num_boost_round=best_rounds)

    n_trees = model.num_trees()
    print(f"  Final model: {n_trees} trees, {len(TOP_FEATURES)} features", flush=True)

    # Held-out eval
    print(f"\nHeld-out evaluation:", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        dtrain_fold = lgb.Dataset(X_train, label=y_train, feature_name=TOP_FEATURES)
        dval_fold = lgb.Dataset(X_val, label=y_val, feature_name=TOP_FEATURES, reference=dtrain_fold)
        model_fold = lgb.train(
            best_config, dtrain_fold, num_boost_round=best_rounds,
            valid_sets=[dval_fold],
            callbacks=[lgb.log_evaluation(0)],
        )
        y_val_pred = (model_fold.predict(X_val) > 0.5).astype(int)
        print(classification_report(y_val, y_val_pred, target_names=["DISCARD", "KEEP"]), flush=True)
        break

    # Save model
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}", flush=True)
    print(f"Total time: {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
