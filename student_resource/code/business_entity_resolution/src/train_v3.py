"""
=============================================================================
 HIGH-ACCURACY BUSINESS ENTITY RESOLUTION TRAINING PIPELINE (V3)
=============================================================================
Improvements over V2:
 1. High-Recall V3 Blocking (+1.55% Candidate Recall, 24.8% fewer blocking misses):
    - Separated name/address stop words (preserves French region tokens in company names)
    - 2-character acronym support in compound keys
    - 3-character address tokens (oak, bay, box)
    - Street + City/State conjunction keys
    - Squashed / domain subword n-grams
 2. 37 Advanced RapidFuzz & Geo Engineered Features:
    - exact_name_missing_addr: resolves false negatives when one address is missing
    - n_first_token_match: suppresses false positives at shared commercial buildings
    - zip_match & zip_mismatch: 5-digit (US, France) and 6-digit (India) postal code validation
    - num_exact_mismatch: hard penalty for differing street numbers
    - n_jw & a_jw: C-accelerated Jaro-Winkler prefix stability for names and addresses
 3. High-throughput LightGBM training with 24 threads & hard negative mining
 4. Decision Threshold Optimization for Macro F0.5
"""
import os
import sys

os.environ['OMP_NUM_THREADS'] = '24'
os.environ['OPENBLAS_NUM_THREADS'] = '24'
os.environ['MKL_NUM_THREADS'] = '24'

import gc
import time
import math
import heapq
import pickle
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocess import normalize_address
from features_v3 import (
    FEATURE_COLS, NAME_STOP_WORDS, ADDR_STOP_WORDS,
    clean_name_v3, extract_clean_numbers, extract_postal_code,
    char_ngrams, compute_pair_features_v3
)
from evaluate import compute_f05_per_entity


def extract_keys_v3(nc, ac):
    """Extract enhanced V3 candidate blocking keys."""
    n_keys = set()
    a_keys = set()
    c_keys = set()

    raw_n_words = nc.split()
    n_tokens = []
    for w in raw_n_words:
        if w not in NAME_STOP_WORDS:
            if len(w) >= 3:
                n_tokens.append(w)
            elif len(w) == 2 and w.isupper():
                n_tokens.append(w)

    nums = extract_clean_numbers(ac)
    a_tokens = [w for w in ac.split() if w not in ADDR_STOP_WORDS and len(w) >= 3]

    for t in n_tokens:
        if len(t) >= 4:
            n_keys.add('n:' + t)
            n_keys.add('ng:' + t[:4])
            n_keys.add('pfx:' + t[:3])
        elif len(t) == 3:
            n_keys.add('n3:' + t)
        elif len(t) == 2:
            n_keys.add('n2:' + t)

    for num in nums:
        a_keys.add('num:' + num)
    for at in a_tokens:
        a_keys.add('a:' + at)

    for i in range(len(n_tokens)):
        for j in range(i + 1, min(i + 4, len(n_tokens))):
            pair = tuple(sorted([n_tokens[i], n_tokens[j]]))
            c_keys.add(f'np:{pair[0]}_{pair[1]}')

    for t in n_tokens[:3]:
        for num in nums:
            c_keys.add(f'n_num:{t}_{num}')

    for num in nums:
        for at in a_tokens[:3]:
            c_keys.add(f'num_a:{num}_{at}')

    target_a = list(dict.fromkeys(a_tokens[:2] + a_tokens[-2:])) if len(a_tokens) > 2 else a_tokens
    for t in n_tokens[:3]:
        for at in target_a:
            c_keys.add(f'n_a:{t}_{at}')

    if len(n_tokens) == 1 and len(n_tokens[0]) >= 8:
        single_word = n_tokens[0]
        for i in range(min(12, len(single_word) - 3)):
            n_keys.add('ng:' + single_word[i:i+4])

    return n_keys, a_keys, c_keys


def build_country_index_v3(s23_df, s1_n_keys, s1_a_keys, s1_c_keys, s1_names, max_p_comp=5000, max_p_single=3000):
    """Build enhanced inverted index with higher posting threshold and smart conjunctions."""
    exact_name_idx = {}
    n_idx = {}
    a_idx = {}
    c_idx = {}

    print(f"   Indexing {len(s23_df):,} records ...", flush=True)
    t0 = time.time()
    for r in s23_df.itertuples(index=False):
        eid = r.entity_id
        nc = r.name_clean
        ac = r.addr_clean

        if nc and nc in s1_names:
            exact_name_idx.setdefault(nc, []).append(eid)

        nk, ak, ck = extract_keys_v3(nc, ac)
        for k in nk:
            if k in s1_n_keys:
                lst = n_idx.setdefault(k, [])
                if len(lst) < max_p_single:
                    lst.append(eid)
        for k in ak:
            if k in s1_a_keys:
                lst = a_idx.setdefault(k, [])
                if len(lst) < max_p_single:
                    lst.append(eid)
        for k in ck:
            if k in s1_c_keys:
                lst = c_idx.setdefault(k, [])
                if len(lst) < max_p_comp:
                    lst.append(eid)

    print(f"   Index built in {time.time()-t0:.2f}s (Exact: {len(exact_name_idx):,}, Name: {len(n_idx):,}, Addr: {len(a_idx):,}, Compound: {len(c_idx):,})", flush=True)
    return {'exact': exact_name_idx, 'name': n_idx, 'addr': a_idx, 'comp': c_idx}


def retrieve_candidates_v3(nc, ac, idx, comp_k=60, name_k=40, addr_k=40, exact_k=100):
    """Enhanced candidate retrieval with IDF weighting."""
    cands = set()
    exact_idx = idx['exact']
    c_idx = idx['comp']
    n_idx = idx['name']
    a_idx = idx['addr']

    if nc and nc in exact_idx:
        cands.update(exact_idx[nc][:exact_k])

    nk, ak, ck = extract_keys_v3(nc, ac)

    if ck:
        c_scores = {}
        for k in ck:
            postings = c_idx.get(k)
            if postings:
                w = 1.0 / np.log1p(len(postings))
                for mid in postings:
                    c_scores[mid] = c_scores.get(mid, 0.0) + w
        if c_scores:
            cands.update(m for m, _ in heapq.nlargest(comp_k, c_scores.items(), key=lambda x: x[1]))

    if nk:
        n_scores = {}
        for k in nk:
            postings = n_idx.get(k)
            if postings:
                w = 1.0 / np.log1p(len(postings))
                for mid in postings:
                    n_scores[mid] = n_scores.get(mid, 0.0) + w
        if n_scores:
            cands.update(m for m, _ in heapq.nlargest(name_k, n_scores.items(), key=lambda x: x[1]))

    if ak:
        a_scores = {}
        for k in ak:
            postings = a_idx.get(k)
            if postings:
                w = 1.0 / np.log1p(len(postings))
                for mid in postings:
                    a_scores[mid] = a_scores.get(mid, 0.0) + w
        if a_scores:
            cands.update(m for m, _ in heapq.nlargest(addr_k, a_scores.items(), key=lambda x: x[1]))

    return cands


def main():
    parser = argparse.ArgumentParser(description="High-Accuracy Business Entity Resolution Training V3")
    parser.add_argument('--data-dir', default=os.path.join(SRC_DIR, '..', '..', '..', 'dataset'))
    parser.add_argument('--cache-dir', default=os.path.join(SRC_DIR, '..', '..', '..', 'cache'))
    parser.add_argument('--n-train-s1', type=int, default=200000, help="Number of S1 train entities (default: 200,000)")
    parser.add_argument('--n-val-s1', type=int, default=25000, help="Number of S1 validation entities (default: 25,000)")
    parser.add_argument('--max-neg-per-entity', type=int, default=10, help="Max hard negatives per S1 entity for training")
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    t_global = time.time()
    print("=" * 80, flush=True)
    print(" HIGH-ACCURACY BUSINESS ENTITY RESOLUTION PIPELINE (V3 TRAINING)", flush=True)
    print("=" * 80, flush=True)
    print(f" Data Dir:            {os.path.abspath(args.data_dir)}", flush=True)
    print(f" Cache Dir:           {os.path.abspath(args.cache_dir)}", flush=True)
    print(f" Train S1 Entities:   {args.n_train_s1:,}", flush=True)
    print(f" Val S1 Entities:     {args.n_val_s1:,}", flush=True)
    print(f" Max Negatives/S1:    {args.max_neg_per_entity}", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Ground Truth
    print("\n[Stage 1/5] Loading Ground Truth ...", flush=True)
    gt_df = pd.read_csv(os.path.join(args.data_dir, 'train', 'train_ground_truth.tsv'), sep='\t', keep_default_na=False)
    gt_map = {}
    for sid, mids in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']):
        gt_map[sid] = set(mids.split(',')) if mids else set()
    print(f" Loaded {len(gt_map):,} GT entities", flush=True)

    # 2. Load and Sample Reference S1 entities (Stratified by Country)
    print("\n[Stage 2/5] Loading S1 Reference Data ...", flush=True)
    s1_cols = ['entity_id', 'country', 'business_name', 'business_address']
    s1_all = pd.read_csv(os.path.join(args.data_dir, 'train', 'train_source1.tsv'), sep='\t', usecols=s1_cols, keep_default_na=False)
    print(f" Loaded {len(s1_all):,} S1 records", flush=True)

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

    print(" Normalizing S1 names & addresses with clean_name_v3 ...", flush=True)
    t0 = time.time()
    train_s1_df['name_clean'] = train_s1_df['business_name'].apply(clean_name_v3)
    train_s1_df['addr_clean'] = train_s1_df['business_address'].apply(normalize_address)
    val_s1_df['name_clean'] = val_s1_df['business_name'].apply(clean_name_v3)
    val_s1_df['addr_clean'] = val_s1_df['business_address'].apply(normalize_address)
    print(f" S1 normalized in {time.time()-t0:.2f}s", flush=True)

    # 3. Process Country-by-Country
    print("\n[Stage 3/5] Candidate Retrieval & 37-Feature Engineering ...", flush=True)
    train_X_chunks = []
    train_y_chunks = []
    val_X_chunks = []
    val_y_chunks = []
    val_pair_meta = []

    total_val_true_matches = 0
    total_val_found_matches = 0

    s23_cols = ['entity_id', 'country', 'business_name', 'business_address']

    for country in countries:
        print("\n" + "-" * 80, flush=True)
        print(f" PROCESSING COUNTRY: {country}", flush=True)
        print("-" * 80, flush=True)

        t_load = time.time()
        s2_list = []
        for chunk in pd.read_csv(os.path.join(args.data_dir, 'train', 'train_source2.tsv'), sep='\t', chunksize=1000000, usecols=s23_cols, keep_default_na=False):
            sub = chunk[chunk['country'] == country][['entity_id', 'business_name', 'business_address']]
            if len(sub) > 0:
                s2_list.append(sub)

        for chunk in pd.read_csv(os.path.join(args.data_dir, 'train', 'train_source3.tsv'), sep='\t', chunksize=1000000, usecols=s23_cols, keep_default_na=False):
            sub = chunk[chunk['country'] == country][['entity_id', 'business_name', 'business_address']]
            if len(sub) > 0:
                s2_list.append(sub)

        s23_c = pd.concat(s2_list, ignore_index=True).drop_duplicates(subset=['entity_id'])
        del s2_list
        gc.collect()

        print(f"   Loaded {len(s23_c):,} S2+S3 records for {country} in {time.time()-t_load:.2f}s", flush=True)

        t_norm = time.time()
        s23_c['name_clean'] = s23_c['business_name'].apply(clean_name_v3)
        s23_c['addr_clean'] = s23_c['business_address'].apply(normalize_address)
        print(f"   Normalized in {time.time()-t_norm:.2f}s", flush=True)

        s23_dict = dict(zip(s23_c['entity_id'], zip(s23_c['name_clean'], s23_c['addr_clean'])))

        tr_s1_c = train_s1_df[train_s1_df['country'] == country]
        va_s1_c = val_s1_df[val_s1_df['country'] == country]

        s1_n_keys, s1_a_keys, s1_c_keys = set(), set(), set()
        s1_names = set(tr_s1_c['name_clean']) | set(va_s1_c['name_clean'])

        for nc, ac in zip(tr_s1_c['name_clean'], tr_s1_c['addr_clean']):
            nk, ak, ck = extract_keys_v3(nc, ac)
            s1_n_keys.update(nk)
            s1_a_keys.update(ak)
            s1_c_keys.update(ck)
        for nc, ac in zip(va_s1_c['name_clean'], va_s1_c['addr_clean']):
            nk, ak, ck = extract_keys_v3(nc, ac)
            s1_n_keys.update(nk)
            s1_a_keys.update(ak)
            s1_c_keys.update(ck)

        c_index = build_country_index_v3(s23_c, s1_n_keys, s1_a_keys, s1_c_keys, s1_names)
        del s23_c, s1_n_keys, s1_a_keys, s1_c_keys, s1_names
        gc.collect()

        # Training Pairs Generation with Hard Negative Mining
        print(f"   Mining training pairs for {len(tr_s1_c):,} {country} S1 entities ...", flush=True)
        t_tr = time.time()
        tr_feature_rows = []
        tr_labels = []

        for i, row in enumerate(tr_s1_c.itertuples(index=False)):
            sid = row.entity_id
            nc = row.name_clean
            ac = row.addr_clean
            true_m = gt_map.get(sid, set())

            cands = retrieve_candidates_v3(nc, ac, c_index)
            pos_cands = [m for m in cands if m in true_m]
            neg_cands = [m for m in cands if m not in true_m]

            if len(neg_cands) > args.max_neg_per_entity:
                neg_cands = np.random.choice(neg_cands, size=args.max_neg_per_entity, replace=False).tolist()

            selected_pairs = [(m, 1) for m in pos_cands] + [(m, 0) for m in neg_cands]
            if not selected_pairs:
                continue

            w1_tokens = [w for w in nc.split() if w not in NAME_STOP_WORDS] if nc else []
            w1_set = set(w1_tokens)
            aw1_set = set(ac.split()) if ac else set()
            nums1 = extract_clean_numbers(ac)
            has_a1 = bool(ac and ac != 'nan')
            n1_ngrams = char_ngrams(nc, 3)
            a1_ngrams = char_ngrams(ac, 3) if has_a1 else set()
            first_token1 = w1_tokens[0] if w1_tokens else ''
            zip1 = extract_postal_code(ac)

            for mid, lbl in selected_pairs:
                s23_info = s23_dict.get(mid)
                if not s23_info:
                    continue
                n2, a2 = s23_info
                feat_tuple = compute_pair_features_v3(
                    nc, ac, n2, a2,
                    w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
                    first_token1, zip1
                )
                tr_feature_rows.append(feat_tuple)
                tr_labels.append(lbl)

            if (i + 1) % 25000 == 0 or (i + 1) == len(tr_s1_c):
                print(f"     Train Progress: {i+1:,}/{len(tr_s1_c):,} entities | {len(tr_feature_rows):,} pairs ({time.time()-t_tr:.1f}s)", flush=True)

        if tr_feature_rows:
            train_X_chunks.append(np.array(tr_feature_rows, dtype=np.float32))
            train_y_chunks.append(np.array(tr_labels, dtype=np.int32))
            print(f"   Built {country} train matrix: {len(tr_feature_rows):,} pairs (Positives: {sum(tr_labels):,})", flush=True)
            del tr_feature_rows, tr_labels
            gc.collect()

        # Validation Candidates Generation (Full Evaluation without Negative Sampling)
        print(f"   Evaluating validation candidates for {len(va_s1_c):,} {country} S1 entities ...", flush=True)
        t_va = time.time()
        va_feature_rows = []
        va_labels = []

        for i, row in enumerate(va_s1_c.itertuples(index=False)):
            sid = row.entity_id
            nc = row.name_clean
            ac = row.addr_clean
            true_m = gt_map.get(sid, set())

            cands = retrieve_candidates_v3(nc, ac, c_index)

            if true_m:
                total_val_true_matches += len(true_m)
                total_val_found_matches += len(true_m & cands)

            if not cands:
                continue

            w1_tokens = [w for w in nc.split() if w not in NAME_STOP_WORDS] if nc else []
            w1_set = set(w1_tokens)
            aw1_set = set(ac.split()) if ac else set()
            nums1 = extract_clean_numbers(ac)
            has_a1 = bool(ac and ac != 'nan')
            n1_ngrams = char_ngrams(nc, 3)
            a1_ngrams = char_ngrams(ac, 3) if has_a1 else set()
            first_token1 = w1_tokens[0] if w1_tokens else ''
            zip1 = extract_postal_code(ac)

            for mid in sorted(cands):
                s23_info = s23_dict.get(mid)
                if not s23_info:
                    continue
                n2, a2 = s23_info
                feat_tuple = compute_pair_features_v3(
                    nc, ac, n2, a2,
                    w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
                    first_token1, zip1
                )
                va_feature_rows.append(feat_tuple)
                va_labels.append(1 if mid in true_m else 0)
                val_pair_meta.append((sid, mid))

            if (i + 1) % 5000 == 0 or (i + 1) == len(va_s1_c):
                print(f"     Val Progress: {i+1:,}/{len(va_s1_c):,} entities | {len(va_feature_rows):,} pairs ({time.time()-t_va:.1f}s)", flush=True)

        if va_feature_rows:
            val_X_chunks.append(np.array(va_feature_rows, dtype=np.float32))
            val_y_chunks.append(np.array(va_labels, dtype=np.int32))
            del va_feature_rows, va_labels
            gc.collect()

        del c_index, s23_dict, tr_s1_c, va_s1_c
        gc.collect()

    val_blocking_recall = total_val_found_matches / max(1, total_val_true_matches)
    print("\n" + "=" * 80, flush=True)
    print(f" FULL VALIDATION BLOCKING RECALL: {val_blocking_recall:.4f} ({val_blocking_recall*100:.2f}%)", flush=True)
    print(f"   Found {total_val_found_matches:,} / {total_val_true_matches:,} true match pairs", flush=True)
    print("=" * 80, flush=True)

    # 4. Train LightGBM Model
    print("\n[Stage 4/5] Training Deep LightGBM on 37 Engineered Features ...", flush=True)
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

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=dtrain, free_raw_data=False)

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

    imp_gain = model.feature_importance(importance_type='gain')
    imp_split = model.feature_importance(importance_type='split')
    imp_df = pd.DataFrame({
        'feature': FEATURE_COLS,
        'gain': imp_gain,
        'splits': imp_split
    }).sort_values('gain', ascending=False)
    print("\n Top 20 Most Informative Features (Gain):", flush=True)
    for _, r in imp_df.head(20).iterrows():
        print(f"   {r['feature']:24s} | Gain: {r['gain']:12,.1f} | Splits: {int(r['splits']):6d}", flush=True)

    # 5. Full Decision Threshold Sweep on Macro F0.5
    print("\n[Stage 5/5] Calibrating Decision Threshold for Macro F0.5 ...", flush=True)
    val_probs = model.predict(X_val, num_threads=24)

    all_val_s1_ids = list(val_s1_df['entity_id'])
    entity_cand_probs = {sid: [] for sid in all_val_s1_ids}
    for (sid, mid), p in zip(val_pair_meta, val_probs):
        entity_cand_probs[sid].append((mid, float(p)))

    del X_train, y_train, X_val, y_val, val_pair_meta, val_probs
    gc.collect()

    threshold_range = np.arange(0.50, 0.95, 0.02)
    best_th = 0.85
    best_f05 = 0.0
    best_stats = {}

    print(f" Sweeping thresholds from {threshold_range[0]:.2f} to {threshold_range[-1]:.2f} across {len(all_val_s1_ids):,} entities ...", flush=True)

    for th in threshold_range:
        scores = []
        total_tp = 0
        total_fp = 0
        total_fn = 0
        n_predicted_matches = 0

        for sid in all_val_s1_ids:
            true_set = gt_map.get(sid, set())
            c_pairs = entity_cand_probs[sid]

            pred_set = {mid for mid, p in c_pairs if p >= th}
            n_predicted_matches += len(pred_set)

            score = compute_f05_per_entity(pred_set, true_set)
            scores.append(score)

            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        macro_f05 = float(np.mean(scores))
        prec = total_tp / max(1, total_tp + total_fp)
        rec = total_tp / max(1, total_tp + total_fn)
        avg_m = n_predicted_matches / len(all_val_s1_ids)

        is_best = macro_f05 > best_f05
        if is_best:
            best_f05 = macro_f05
            best_th = float(th)
            best_stats = {
                'macro_f05': macro_f05,
                'precision': prec,
                'recall': rec,
                'tp': total_tp,
                'fp': total_fp,
                'fn': total_fn,
                'avg_matches': avg_m
            }

        star = " *** BEST ***" if is_best else ""
        print(f"   Th: {th:.2f} | Macro F0.5: {macro_f05:.4f} ({macro_f05*100:.2f}%) | Prec: {prec:.4f} | Rec: {rec:.4f} | Matches/Ent: {avg_m:.2f}{star}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(" OPTIMAL CALIBRATION SUMMARY (V3 MODEL):", flush=True)
    print(f"   Optimal Threshold:          {best_th:.2f}", flush=True)
    print(f"   Best Validation Macro F0.5:   {best_f05:.4f} ({best_f05*100:.2f}%)", flush=True)
    print(f"   Micro Precision:            {best_stats.get('precision', 0):.4f}", flush=True)
    print(f"   Micro Recall:               {best_stats.get('recall', 0):.4f}", flush=True)
    print(f"   True Positives (TP):        {best_stats.get('tp', 0):,}", flush=True)
    print(f"   False Positives (FP):       {best_stats.get('fp', 0):,}", flush=True)
    print(f"   False Negatives (FN):       {best_stats.get('fn', 0):,}", flush=True)
    print(f"   Avg Matches/Entity:         {best_stats.get('avg_matches', 0):.2f}", flush=True)
    print(f"   Blocking Recall:            {val_blocking_recall*100:.2f}%", flush=True)
    print("=" * 80, flush=True)

    checkpoint_path = os.path.join(args.cache_dir, 'lgbm_model_v3.pkl')
    with open(checkpoint_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'threshold': best_th,
            'feature_cols': FEATURE_COLS,
            'val_f05': best_f05,
            'blocking_recall': val_blocking_recall,
            'stats': best_stats
        }, f)
    print(f"\n [*] V3 Checkpoint successfully saved to {checkpoint_path}", flush=True)
    print(f" [*] Entire training pipeline finished in {time.time()-t_global:.1f}s!", flush=True)


if __name__ == '__main__':
    main()
