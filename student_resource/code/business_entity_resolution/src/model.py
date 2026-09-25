"""
LightGBM Model Training and Decision Threshold Optimization for Macro F0.5.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import time
from evaluate import compute_macro_f05
from features import FEATURE_COLS


def train_lgbm(feature_df, feature_cols=None, n_estimators=350, learning_rate=0.06):
    """Train LightGBM binary classification model on labelled candidate pairs."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS

    X = feature_df[feature_cols].values.astype(np.float32)
    y = feature_df['label'].values.astype(np.int32)

    pos = int(y.sum())
    neg = len(y) - pos
    print(f"  Training LightGBM on {len(X):,} pairs ({pos:,} positives, {neg:,} negatives) ...", flush=True)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 127,
        'max_depth': 8,
        'learning_rate': learning_rate,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': -1,
        'n_jobs': -1,
        'scale_pos_weight': 1.0, # balanced probabilities
    }

    ds = lgb.Dataset(X, label=y, feature_name=feature_cols, free_raw_data=False)
    t0 = time.time()
    model = lgb.train(
        params, ds,
        num_boost_round=n_estimators,
        callbacks=[lgb.log_evaluation(100)]
    )
    print(f"  LightGBM trained in {time.time()-t0:.1f}s", flush=True)

    # Feature importance
    imp = model.feature_importance(importance_type='gain')
    imp_df = pd.DataFrame({'feature': feature_cols, 'importance': imp}).sort_values('importance', ascending=False)
    print("\n  Top Feature Importances (Gain):")
    for _, r in imp_df.head(8).iterrows():
        print(f"    {r['feature']:20s}: {r['importance']:,.1f}")

    return model


def predict_probabilities(model, feature_df, feature_cols=None):
    """Predict matching probabilities P(match) for candidate pairs."""
    if feature_cols is None:
        feature_cols = FEATURE_COLS
    if len(feature_df) == 0:
        return np.array([])
    X = feature_df[feature_cols].values.astype(np.float32)
    return model.predict(X)


def optimize_threshold_f05(val_df, probs, gt_dict, val_s1_ids, thresholds=None):
    """
    Sweep decision thresholds on validation split to find optimal threshold maximizing macro F0.5.
    Returns: (best_threshold, best_f05, sweep_results_df)
    """
    if thresholds is None:
        thresholds = np.arange(0.35, 0.90, 0.05)

    val_df = val_df[['s1_id', 's23_id']].copy()
    val_df['prob'] = probs

    results = []
    best_th = 0.65
    best_score = 0.0

    print("\n  Evaluating Decision Thresholds for Macro F0.5 ...", flush=True)

    for th in thresholds:
        preds = {sid: set() for sid in val_s1_ids}
        above = val_df[val_df['prob'] >= th]
        for sid, mid in zip(above['s1_id'], above['s23_id']):
            preds[sid].add(mid)

        f05 = compute_macro_f05(preds, gt_dict, val_s1_ids)
        n_matches = len(above)
        results.append({'threshold': th, 'f05': f05, 'n_predicted_matches': n_matches})

        if f05 > best_score:
            best_score = f05
            best_th = th

        print(f"    Threshold {th:.2f} -> Macro F0.5 = {f05:.4f} ({f05*100:.2f}%)  [predicted {n_matches:,} matches]")

    sweep_df = pd.DataFrame(results)
    print(f"\n  [*] Optimal Threshold: {best_th:.2f} -> Best Validation Macro F0.5: {best_score:.4f} ({best_score*100:.2f}%)", flush=True)
    return best_th, best_score, sweep_df


def apply_decision_threshold(df_with_probs, threshold, all_s1_ids):
    """
    Apply decision threshold to get final entity resolution matching dictionary.
    Returns: dict {s1_id: set(matched_s23_ids)}
    """
    preds = {sid: set() for sid in all_s1_ids}
    if len(df_with_probs) > 0:
        above = df_with_probs[df_with_probs['prob'] >= threshold]
        for sid, mid in zip(above['s1_id'], above['s23_id']):
            preds[sid].add(mid)
    return preds
