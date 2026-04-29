"""Fast CC model training: skip config search, use known-best params.

Uses 90/10 train/val split instead of 5-fold CV for speed on 5M rows.
"""

import os
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

TOP_FEATURES = [
    "link_ratio", "page_total_blocks", "parent_tag_type", "page_total_link_ratio",
    "tag_type", "section_block_count", "page_total_text_len", "dom_depth",
    "section_link_density", "position", "tag_type_score", "section_heading_text_len",
    "blocks_since_heading", "word_count", "next_block_link_ratio", "prev_block_link_ratio",
    "text_to_tag_ratio", "semantic_ancestor", "blocks_until_heading", "page_heading_count",
    "parent_class_id_score", "next_block_text_len", "link_count", "text_len",
    "tag_count", "avg_word_length", "in_nav", "in_footer", "in_aside", "in_header",
]

MAX_ROUNDS = 1000

PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "verbosity": -1,
    "num_leaves": 255,
    "learning_rate": 0.1,
    "min_data_in_leaf": 20,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "num_threads": -1,
}


def train_model(name, train_path, model_path, extra_dfs=None):
    t0 = time.time()
    print(f"\n{'='*60}", flush=True)
    print(f"Training: {name}", flush=True)
    print(f"{'='*60}", flush=True)

    print(f"Loading {train_path}...", flush=True)
    df = pd.read_csv(train_path)
    if extra_dfs:
        for p in extra_dfs:
            print(f"Loading {p}...", flush=True)
            df = pd.concat([df, pd.read_csv(p)], ignore_index=True)

    X = df[TOP_FEATURES].values
    y = df["label"].values
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    params = {**PARAMS, "scale_pos_weight": n_neg / max(n_pos, 1)}
    print(f"  {len(y)} blocks, {n_pos} keep / {n_neg} discard", flush=True)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=TOP_FEATURES)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=TOP_FEATURES, reference=dtrain)

    print(f"Training (max {MAX_ROUNDS} rounds, early stop 50)...", flush=True)
    model = lgb.train(
        params, dtrain, num_boost_round=MAX_ROUNDS,
        valid_sets=[dval], valid_names=["val"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    best_iter = model.best_iteration
    print(f"  Best iteration: {best_iter}, loss: {model.best_score['val']['binary_logloss']:.5f}", flush=True)

    y_pred = (model.predict(X_val) > 0.5).astype(int)
    print(classification_report(y_val, y_pred, target_names=["DISCARD", "KEEP"]), flush=True)

    print(f"Retraining on all data with {best_iter} rounds...", flush=True)
    dtrain_full = lgb.Dataset(X, label=y, feature_name=TOP_FEATURES)
    model_final = lgb.train(params, dtrain_full, num_boost_round=best_iter)
    model_final.save_model(model_path)
    print(f"  Saved: {model_path} ({model_final.num_trees()} trees)", flush=True)
    print(f"  Time: {time.time() - t0:.0f}s", flush=True)


def main():
    cc_path = os.path.join(DATA_DIR, "training_data_cc.csv")
    wmb_path = os.path.join(DATA_DIR, "training_data_dom.csv")

    train_model("CC only", cc_path, os.path.join(DATA_DIR, "model_dom_v3_cc.txt"))
    train_model("WMB + CC", cc_path, os.path.join(DATA_DIR, "model_dom_v3_combined.txt"),
                extra_dfs=[wmb_path])


if __name__ == "__main__":
    main()
