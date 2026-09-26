"""
=============================================================================
 ML CHALLENGE 2026 — HIGH-PRECISION MODEL TRAINING (V10 ARCHITECTURE)
=============================================================================
Trains a deep LightGBM booster on 67 engineered features combining:
  - 43 core name, address, number, and postal code signals (V4)
  - 16 IDF distinctiveness, unshared token penalty, and series signals (V9)
  - 8 novel geographic, numeric sequence, and distributional cosine signals (V10)

Evaluates on the exact 24,999-entity validation holdout (zero data leakage)
and calibrates decision threshold for Macro F0.5.
"""

import os
import sys
import gc
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from collections import defaultdict

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocess import (
    load_ground_truth, normalize_name, normalize_address,
    create_validation_split
)
from features_v4 import (
    clean_name_v3, extract_clean_numbers, extract_postal_code,
    char_ngrams, NAME_STOP_WORDS, INDIC_RE
)
from features_v9 import get_idf_dict
from features_v10 import (
    compute_pair_features_v10, FEATURE_COLS_V10,
    extract_leading_number, extract_us_state, extract_fr_dept
)
from evaluate import parse_ground_truth, compute_f05_per_entity


def build_country_inverted_index(s23_df):
    token_index = defaultdict(list)
    prefix_index = defaultdict(list)
    for idx, (nc, _) in enumerate(zip(s23_df['clean_name'], s23_df['clean_address'])):
        if not nc:
            continue
        tokens = [t for t in nc.split() if t not in NAME_STOP_WORDS and len(t) >= 2]
        seen_t = set()
        for t in tokens:
            if t not in seen_t:
                token_index[t].append(idx)
                seen_t.add(t)
        for t in tokens:
            pfx = t[:4]
            if len(pfx) >= 3:
                prefix_index[pfx].append(idx)
                break
    return token_index, prefix_index


def retrieve_candidates(nc, ac, token_index, prefix_index, s23_records, max_candidates=250):
    if not nc:
        return []
    tokens = [t for t in nc.split() if t not in NAME_STOP_WORDS and len(t) >= 2]
    c_counts = defaultdict(int)
    for t in tokens:
        postings = token_index.get(t, [])
        if len(postings) > 20000:
            continue
        for idx in postings:
            c_counts[idx] += 3

    if len(c_counts) < 30:
        for t in tokens:
            pfx = t[:4]
            if len(pfx) >= 3:
                postings = prefix_index.get(pfx, [])
                if len(postings) <= 15000:
                    for idx in postings:
                        c_counts[idx] += 1
                break

    if not c_counts:
        return []

    sorted_indices = sorted(c_counts.keys(), key=lambda i: c_counts[i], reverse=True)[:max_candidates]
    return [s23_records[i][0] for i in sorted_indices]


def main():
    parser = argparse.ArgumentParser(description="Train V10 67-Feature Model")
    parser.add_argument('--data-dir', default=os.path.abspath(os.path.join(SRC_DIR, '..', '..', '..', 'dataset')))
    parser.add_argument('--cache-dir', default=os.path.abspath(os.path.join(SRC_DIR, '..', '..', '..', 'cache')))
    parser.add_argument('--n-train-s1', type=int, default=75000, help="Number of S1 entities for training (default: 75000)")
    parser.add_argument('--n-val-s1', type=int, default=24999, help="Number of S1 entities for validation (default: 24999)")
    parser.add_argument('--max-neg-per-entity', type=int, default=6, help="Hard negatives mined per S1 (default: 6)")
    parser.add_argument('--save-model-name', default='lgbm_model_v10.pkl')
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 80, flush=True)
    print(" HIGH-PRECISION BUSINESS ENTITY RESOLUTION PIPELINE (V10 TRAINING)", flush=True)
    print("=" * 80, flush=True)
    print(f" Data Dir:            {args.data_dir}")
    print(f" Cache Dir:           {args.cache_dir}")
    print(f" Train S1 Entities:   {args.n_train_s1:,}")
    print(f" Val S1 Entities:     {args.n_val_s1:,}")
    print(f" Max Negatives/S1:    {args.max_neg_per_entity}")
    print(f" Features:            {len(FEATURE_COLS_V10)} engineered features (V10)")
    print("=" * 80, flush=True)

    # 1. Load Ground Truth & IDF Dictionary
    print("\n[Stage 1/5] Loading Ground Truth & Token IDF ...", flush=True)
    gt_df = load_ground_truth(args.data_dir)
    gt_map = parse_ground_truth(gt_df)
    print(f" Loaded {len(gt_map):,} GT entities")

    idf_dict = get_idf_dict()
    print(f" Loaded {len(idf_dict):,} token IDF entries from cache")

    # 2. Load S1 reference and split
    print("\n[Stage 2/5] Loading S1 Reference Data ...", flush=True)
    s1_all = pd.read_csv(os.path.join(args.data_dir, 'train', 'train_source1.tsv'), sep='\t', dtype=str)
    s1_all = s1_all.fillna({'business_name': '', 'business_address': '', 'country': ''})
    print(f" Loaded {len(s1_all):,} S1 records")

    np.random.seed(42)
    countries = ['US', 'India']
    train_s1_list = []
    val_s1_list = []

    for c in countries:
        s1_c = s1_all[s1_all['country'] == c].copy()
        s1_c = s1_c.sample(frac=1.0, random_state=42).reset_index(drop=True)
        c_ratio = len(s1_c) / len(s1_all)
        n_tr = int(args.n_train_s1 * c_ratio)
        n_va = int(args.n_val_s1 * c_ratio)

        val_s1_list.append(s1_c.iloc[:n_va])
        train_s1_list.append(s1_c.iloc[n_va:n_va + n_tr])
        print(f"   Country {c}: {n_tr:,} train, {n_va:,} val", flush=True)

    train_s1_df = pd.concat(train_s1_list, ignore_index=True)
    val_s1_df = pd.concat(val_s1_list, ignore_index=True)
    del s1_all, train_s1_list, val_s1_list
    gc.collect()

    # Normalization of S1
    print(" Normalizing S1 names & addresses with clean_name_v3 ...", flush=True)
    t_norm = time.time()
    train_s1_df['clean_name'] = [clean_name_v3(n) for n in train_s1_df['business_name']]
    train_s1_df['clean_address'] = [normalize_address(a) for a in train_s1_df['business_address']]
    val_s1_df['clean_name'] = [clean_name_v3(n) for n in val_s1_df['business_name']]
    val_s1_df['clean_address'] = [normalize_address(a) for a in val_s1_df['business_address']]
    print(f" S1 normalized in {time.time()-t_norm:.2f}s")

    # 3. Process Country Candidate Pools
    print("\n[Stage 3/5] Candidate Retrieval & 67-Feature Engineering (V10) ...", flush=True)
    train_X_chunks, train_y_chunks = [], []
    val_X_chunks, val_y_chunks = [], []
    val_pair_meta = []
    total_val_true_matches = 0
    total_val_found_matches = 0

    countries = ['US', 'India']
    for c in countries:
        print("\n" + "-" * 80)
        print(f" PROCESSING COUNTRY: {c}")
        print("-" * 80)

        tr_s1_c = train_s1_df[train_s1_df['country'] == c].reset_index(drop=True)
        va_s1_c = val_s1_df[val_s1_df['country'] == c].reset_index(drop=True)

        t_load = time.time()
        s2_df = pd.read_csv(os.path.join(args.data_dir, 'train', 'train_source2.tsv'), sep='\t', dtype=str)
        s3_df = pd.read_csv(os.path.join(args.data_dir, 'train', 'train_source3.tsv'), sep='\t', dtype=str)
        s2_c = s2_df[s2_df['country'] == c].fillna({'business_name': '', 'business_address': ''})
        s3_c = s3_df[s3_df['country'] == c].fillna({'business_name': '', 'business_address': ''})
        del s2_df, s3_df
        gc.collect()

        s23_c = pd.concat([s2_c, s3_c], ignore_index=True).drop_duplicates(subset=['entity_id'])
        del s2_c, s3_c
        gc.collect()
        print(f"   Loaded {len(s23_c):,} S2+S3 records for {c} in {time.time()-t_load:.2f}s")

        t_n23 = time.time()
        s23_c['clean_name'] = [clean_name_v3(n) for n in s23_c['business_name']]
        s23_c['clean_address'] = [normalize_address(a) for a in s23_c['business_address']]
        print(f"   Normalized in {time.time()-t_n23:.2f}s")

        s23_dict = {}
        for mid, cn, ca, rn in zip(s23_c['entity_id'], s23_c['clean_name'], s23_c['clean_address'], s23_c['business_name']):
            s23_dict[mid] = (cn, ca, str(rn) if pd.notna(rn) else '')

        c_index, pfx_index = build_country_inverted_index(s23_c)
        s23_records = list(zip(s23_c['entity_id'], s23_c['clean_name'], s23_c['clean_address']))
        del s23_c
        gc.collect()

        # Build Training Pairs
        print(f"   Mining training pairs for {len(tr_s1_c):,} {c} S1 entities ...", flush=True)
        c_train_X, c_train_y = [], []
        t_tr = time.time()
        for idx, row in tr_s1_c.iterrows():
            sid = row['entity_id']
            nc = row['clean_name']
            ac = row['clean_address']
            raw_n1 = str(row['business_name']) if pd.notna(row['business_name']) else ''
            true_targets = gt_map.get(sid, set())

            cands = retrieve_candidates(nc, ac, c_index, pfx_index, s23_records, max_candidates=100)
            cand_set = set(cands)

            # Pre-compute S1 attributes
            w1_toks = [w for w in nc.split() if w not in NAME_STOP_WORDS] if nc else []
            w1_set = set(w1_toks)
            aw1_set = set(ac.split()) if ac else set()
            raw_w1_set = set(raw_n1.lower().split()) if raw_n1 else set()
            nums1 = extract_clean_numbers(ac)
            has_a1 = bool(ac and ac != 'nan')
            n1_ngrams = char_ngrams(nc, 3)
            a1_ngrams = char_ngrams(ac, 3) if has_a1 else set()
            first_tok1 = w1_toks[0] if w1_toks else ''
            zip1 = extract_postal_code(ac)
            indic1 = bool(INDIC_RE.search(raw_n1)) if raw_n1 else False
            lead_num1 = extract_leading_number(ac)
            us_state1 = extract_us_state(ac)
            fr_dept1 = extract_fr_dept(ac)
            n1_4grams = char_ngrams(nc, 4)

            # Positive pairs
            for mid in true_targets:
                if mid in s23_dict:
                    n2, a2, raw_n2 = s23_dict[mid]
                    f = compute_pair_features_v10(
                        nc, ac, raw_n1, raw_w1_set,
                        n2, a2, raw_n2,
                        w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
                        first_tok1, zip1, indic1,
                        lead_num1, us_state1, fr_dept1, n1_4grams,
                        idf_dict=idf_dict
                    )
                    c_train_X.append(f)
                    c_train_y.append(1)

            # Hard Negative pairs
            neg_added = 0
            for mid in cands:
                if mid not in true_targets and mid in s23_dict:
                    n2, a2, raw_n2 = s23_dict[mid]
                    f = compute_pair_features_v10(
                        nc, ac, raw_n1, raw_w1_set,
                        n2, a2, raw_n2,
                        w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
                        first_tok1, zip1, indic1,
                        lead_num1, us_state1, fr_dept1, n1_4grams,
                        idf_dict=idf_dict
                    )
                    c_train_X.append(f)
                    c_train_y.append(0)
                    neg_added += 1
                    if neg_added >= args.max_neg_per_entity:
                        break

            if (idx + 1) % 25000 == 0 or (idx + 1) == len(tr_s1_c):
                print(f"     Train Progress: {idx+1:,}/{len(tr_s1_c):,} entities | {len(c_train_X):,} pairs ({time.time()-t_tr:.1f}s)", flush=True)

        train_X_chunks.append(np.array(c_train_X, dtype=np.float32))
        train_y_chunks.append(np.array(c_train_y, dtype=np.int32))
        del c_train_X, c_train_y
        gc.collect()

        # Build Validation Pairs
        print(f"   Evaluating validation candidates for {len(va_s1_c):,} {c} S1 entities ...", flush=True)
        c_val_X, c_val_y = [], []
        t_va = time.time()
        for idx, row in va_s1_c.iterrows():
            sid = row['entity_id']
            nc = row['clean_name']
            ac = row['clean_address']
            raw_n1 = str(row['business_name']) if pd.notna(row['business_name']) else ''
            true_targets = gt_map.get(sid, set())
            total_val_true_matches += len(true_targets)

            cands = retrieve_candidates(nc, ac, c_index, pfx_index, s23_records, max_candidates=250)
            cand_set = set(cands)
            total_val_found_matches += len(true_targets & cand_set)

            w1_toks = [w for w in nc.split() if w not in NAME_STOP_WORDS] if nc else []
            w1_set = set(w1_toks)
            aw1_set = set(ac.split()) if ac else set()
            raw_w1_set = set(raw_n1.lower().split()) if raw_n1 else set()
            nums1 = extract_clean_numbers(ac)
            has_a1 = bool(ac and ac != 'nan')
            n1_ngrams = char_ngrams(nc, 3)
            a1_ngrams = char_ngrams(ac, 3) if has_a1 else set()
            first_tok1 = w1_toks[0] if w1_toks else ''
            zip1 = extract_postal_code(ac)
            indic1 = bool(INDIC_RE.search(raw_n1)) if raw_n1 else False
            lead_num1 = extract_leading_number(ac)
            us_state1 = extract_us_state(ac)
            fr_dept1 = extract_fr_dept(ac)
            n1_4grams = char_ngrams(nc, 4)

            for mid in cands:
                if mid in s23_dict:
                    n2, a2, raw_n2 = s23_dict[mid]
                    f = compute_pair_features_v10(
                        nc, ac, raw_n1, raw_w1_set,
                        n2, a2, raw_n2,
                        w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
                        first_tok1, zip1, indic1,
                        lead_num1, us_state1, fr_dept1, n1_4grams,
                        idf_dict=idf_dict
                    )
                    c_val_X.append(f)
                    c_val_y.append(1 if mid in true_targets else 0)
                    val_pair_meta.append((sid, mid))

            if (idx + 1) % 5000 == 0 or (idx + 1) == len(va_s1_c):
                print(f"     Val Progress: {idx+1:,}/{len(va_s1_c):,} entities | {len(c_val_X):,} pairs ({time.time()-t_va:.1f}s)", flush=True)

        val_X_chunks.append(np.array(c_val_X, dtype=np.float32))
        val_y_chunks.append(np.array(c_val_y, dtype=np.int32))
        del c_val_X, c_val_y, c_index, pfx_index, s23_dict, tr_s1_c, va_s1_c
        gc.collect()

    val_blocking_recall = total_val_found_matches / max(1, total_val_true_matches)
    print("\n" + "=" * 80, flush=True)
    print(f" FULL V10 VALIDATION BLOCKING RECALL: {val_blocking_recall:.4f} ({val_blocking_recall*100:.2f}%)", flush=True)
    print(f"   Found {total_val_found_matches:,} / {total_val_true_matches:,} true match pairs", flush=True)
    print("=" * 80, flush=True)

    # 4. Train LightGBM Model
    print("\n[Stage 4/5] Training Deep LightGBM on 67 Engineered Features (V10) ...", flush=True)
    X_train = np.vstack(train_X_chunks)
    y_train = np.concatenate(train_y_chunks)
    del train_X_chunks, train_y_chunks
    gc.collect()

    pos_tr = int(y_train.sum())
    neg_tr = len(y_train) - pos_tr
    print(f" Training Matrix: {X_train.shape[0]:,} rows x {X_train.shape[1]} features", flush=True)
    print(f"   Positives: {pos_tr:,} | Negatives: {neg_tr:,} | Pos Ratio: {pos_tr/len(y_train)*100:.2f}%", flush=True)

    X_val = np.vstack(val_X_chunks)
    y_val = np.concatenate(val_y_chunks)
    del val_X_chunks, val_y_chunks
    gc.collect()

    pos_va = int(y_val.sum())
    neg_va = len(y_val) - pos_va
    print(f" Validation Matrix: {X_val.shape[0]:,} rows (Positives: {pos_va:,}, Negatives: {neg_va:,})", flush=True)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS_V10, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS_V10, reference=dtrain, free_raw_data=False)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 255,
        'max_depth': 12,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'lambda_l1': 0.1,
        'lambda_l2': 1.0,
        'verbose': -1,
        'n_jobs': 24,
    }

    t_tr_start = time.time()
    print(" Starting LightGBM training ...", flush=True)
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'val'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=True),
            lgb.log_evaluation(period=50)
        ]
    )
    print(f" LightGBM trained in {time.time()-t_tr_start:.1f}s (Best iteration: {model.best_iteration})", flush=True)

    # Feature Importance
    imp_gain = model.feature_importance(importance_type='gain')
    imp_split = model.feature_importance(importance_type='split')
    imp_df = pd.DataFrame({
        'feature': FEATURE_COLS_V10,
        'gain': imp_gain,
        'splits': imp_split
    }).sort_values('gain', ascending=False)
    print("\n Top 30 Most Informative Features (Gain):", flush=True)
    for rank, (_, r) in enumerate(imp_df.head(30).iterrows(), 1):
        is_v10 = "**" if r['feature'] in FEATURE_COLS_V10[59:] else ("* " if r['feature'] in FEATURE_COLS_V10[43:59] else "  ")
        print(f"   {rank:2d}. {is_v10} {r['feature']:26s} | Gain: {r['gain']:12,.1f} | Splits: {int(r['splits']):6d}", flush=True)

    # 5. Full Decision Threshold Sweep on Macro F0.5
    print("\n[Stage 5/5] Calibrating Decision Threshold for Macro F0.5 ...", flush=True)
    val_probs = model.predict(X_val, num_threads=24)

    all_val_s1_ids = list(val_s1_df['entity_id'])
    entity_cand_probs = {sid: [] for sid in all_val_s1_ids}
    for (sid, mid), p in zip(val_pair_meta, val_probs):
        entity_cand_probs[sid].append((mid, float(p)))

    best_th = 0.98
    best_f05 = 0.0

    for th in np.arange(0.90, 0.995, 0.01):
        th = round(th, 3)
        scores = []
        tot_tp, tot_fp, tot_fn = 0, 0, 0
        for sid in all_val_s1_ids:
            true_set = gt_map.get(sid, set())
            c_pairs = entity_cand_probs[sid]
            c_pairs_filt = [(mid, p) for mid, p in c_pairs if p >= th]
            c_pairs_filt.sort(key=lambda x: x[1], reverse=True)
            pred_set = {mid for mid, _ in c_pairs_filt[:6]}
            score = compute_f05_per_entity(pred_set, true_set)
            scores.append(score)
            tot_tp += len(pred_set & true_set)
            tot_fp += len(pred_set - true_set)
            tot_fn += len(true_set - pred_set)

        macro_f05 = np.mean(scores)
        prec = tot_tp / max(1, tot_tp + tot_fp)
        rec = tot_tp / max(1, tot_tp + tot_fn)
        print(f"   Threshold: {th:.3f} -> Macro F0.5: {macro_f05:.6f} | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | TP: {tot_tp:,} FP: {tot_fp:,} FN: {tot_fn:,}", flush=True)
        if macro_f05 > best_f05:
            best_f05 = macro_f05
            best_th = th

    # Global Conflict Resolution on validation
    print("\n Evaluating with Global 1-to-1 Conflict Resolution ...", flush=True)
    base_preds = {}
    prob_map = {}
    for sid in all_val_s1_ids:
        c_pairs = entity_cand_probs[sid]
        c_pairs_filt = [(mid, p) for mid, p in c_pairs if p >= best_th]
        c_pairs_filt.sort(key=lambda x: x[1], reverse=True)
        base_preds[sid] = [mid for mid, _ in c_pairs_filt[:6]]
        for mid, p in c_pairs_filt[:6]:
            prob_map[(sid, mid)] = p

    mid_to_sids = {}
    for sid, mids in base_preds.items():
        for mid in mids:
            mid_to_sids.setdefault(mid, []).append(sid)

    resolved_preds = {sid: list(mids) for sid, mids in base_preds.items()}
    conflicts_resolved = 0
    for mid, sids in mid_to_sids.items():
        if len(sids) > 1:
            conflicts_resolved += len(sids) - 1
            best_sid = max(sids, key=lambda s: prob_map[(s, mid)])
            for s in sids:
                if s != best_sid:
                    resolved_preds[s].remove(mid)

    resolved_scores = [compute_f05_per_entity(set(resolved_preds[sid]), gt_map.get(sid, set())) for sid in all_val_s1_ids]
    f05_resolved = np.mean(resolved_scores)
    print(f"   Conflicts resolved: {conflicts_resolved:,} duplicate slots")
    print(f"   Macro F0.5 after Conflict Resolution: {f05_resolved:.6f} (Delta: {f05_resolved - best_f05:+.6f})")

    # 6. Save V10 Model Checkpoint
    model_save_path = os.path.join(args.cache_dir, args.save_model_name)
    ckpt = {
        'model': model,
        'threshold': best_th,
        'val_f05': f05_resolved,
        'val_blocking_recall': val_blocking_recall,
        'feature_names': FEATURE_COLS_V10,
        'best_iteration': model.best_iteration,
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_train_s1': len(train_s1_df),
        'n_val_s1': len(val_s1_df)
    }
    with open(model_save_path, 'wb') as f:
        pickle.dump(ckpt, f)
    print(f"\n Saved V10 Model Checkpoint to: {model_save_path}")
    print(f" Total Pipeline Runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
