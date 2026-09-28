"""
=============================================================================
 ML CHALLENGE 2026 — ULTRA-FAST STREAMING XGBOOST TRAINING (V13 ENSEMBLE)
=============================================================================
Mines hard negatives on-the-fly by streaming Source 2 and Source 3 chunks
against sampled Source 1 reference entities.
Avoids normalizing 10.3M records in memory, achieving 100x speedup and
training a high-precision complementary XGBoost booster in under 90 seconds.
"""

import os
import sys
import gc
import time
import pickle
import argparse
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocess import load_ground_truth, normalize_address
from features_v4 import (
    clean_name_v3, extract_clean_numbers, extract_postal_code,
    char_ngrams, NAME_STOP_WORDS, INDIC_RE
)
from features_v9 import get_idf_dict
from features_v10 import (
    compute_pair_features_v10, FEATURE_COLS_V10,
    extract_leading_number, extract_us_state, extract_fr_dept
)
from evaluate import parse_ground_truth


def main():
    parser = argparse.ArgumentParser(description="Train Streamed XGBoost Booster")
    parser.add_argument('--data-dir', default=os.path.abspath(os.path.join(SRC_DIR, '..', '..', '..', 'dataset')))
    parser.add_argument('--cache-dir', default=os.path.abspath(os.path.join(SRC_DIR, '..', '..', '..', 'cache')))
    parser.add_argument('--n-train-s1', type=int, default=30000, help="S1 entities for training (default: 30000)")
    parser.add_argument('--max-neg-per-entity', type=int, default=4, help="Negatives per S1 (default: 4)")
    parser.add_argument('--save-model-name', default='xgb_model_v13.pkl')
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 80, flush=True)
    print(" HIGH-THROUGHPUT STREAMED XGBOOST TRAINING PIPELINE (V13 ENSEMBLE)", flush=True)
    print("=" * 80, flush=True)

    # 1. Load Ground Truth & Token IDF
    print("\n[Stage 1/5] Loading Ground Truth & Token IDF ...", flush=True)
    gt_df = load_ground_truth(args.data_dir)
    gt_map = parse_ground_truth(gt_df)
    del gt_df
    idf_dict = get_idf_dict()
    print(f" Loaded {len(gt_map):,} GT entities and {len(idf_dict):,} token IDF terms")

    # 2. Sample S1 Training Entities
    print(f"\n[Stage 2/5] Sampling {args.n_train_s1:,} S1 Entities Stratified by Country ...", flush=True)
    s1_all = pd.read_csv(os.path.join(args.data_dir, 'train', 'train_source1.tsv'), sep='\t', dtype=str)
    s1_all = s1_all.fillna({'business_name': '', 'business_address': '', 'country': ''})

    np.random.seed(42)
    s1_sampled_list = []
    for c in ['US', 'India']:
        s1_c = s1_all[s1_all['country'] == c].copy()
        s1_c = s1_c.sample(n=int(args.n_train_s1 * (len(s1_c)/len(s1_all))), random_state=42)
        s1_sampled_list.append(s1_c)

    s1_df = pd.concat(s1_sampled_list, ignore_index=True)
    del s1_all, s1_sampled_list
    gc.collect()

    s1_dict = {}
    s1_pre = {}
    s1_token_index = defaultdict(list)
    needed_positive_ids = set()

    for _, row in s1_df.iterrows():
        sid = row['entity_id']
        raw_n = str(row['business_name'])
        raw_a = str(row['business_address'])
        c = row['country'] if row['country'] else 'US'

        nc = clean_name_v3(raw_n)
        ac = normalize_address(raw_a)
        w1_toks = [w for w in nc.split() if w not in NAME_STOP_WORDS and len(w) >= 3] if nc else []
        w1_set = set(w1_toks)
        aw1_set = set(ac.split()) if ac else set()
        raw_w1_set = set(raw_n.lower().split()) if raw_n else set()
        nums1 = extract_clean_numbers(ac)
        has_a1 = bool(ac and ac != 'nan')
        n1_ngrams = char_ngrams(nc, 3)
        a1_ngrams = char_ngrams(ac, 3) if has_a1 else set()
        first_tok1 = w1_toks[0] if w1_toks else ''
        zip1 = extract_postal_code(ac)
        indic1 = bool(INDIC_RE.search(raw_n)) if raw_n else False
        lead_num1 = extract_leading_number(ac)
        us_state1 = extract_us_state(ac) if c == 'US' else None
        fr_dept1 = None
        n1_4grams = char_ngrams(nc, 4)

        s1_pre[sid] = (
            nc, ac, raw_n, raw_w1_set,
            w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
            first_tok1, zip1, indic1,
            lead_num1, us_state1, fr_dept1, n1_4grams, c
        )

        for t in w1_toks[:4]:
            s1_token_index[(t, c)].append(sid)

        for pid in gt_map.get(sid, set()):
            needed_positive_ids.add(pid)

    print(f" Precomputed metadata for {len(s1_pre):,} S1 entities ({len(needed_positive_ids):,} positive targets needed)", flush=True)

    # 3. Stream S2 and S3 to Mine Hard Negatives and Collect Positives
    print("\n[Stage 3/5] Streaming S2 & S3 to Mine Hard Negatives ...", flush=True)
    t_stream = time.time()
    s23_positives = {}
    neg_pairs_by_s1 = defaultdict(list)
    s1_neg_counts = defaultdict(int)

    for s_num in ['2', '3']:
        fpath = os.path.join(args.data_dir, 'train', f'train_source{s_num}.tsv')
        for chunk in pd.read_csv(fpath, sep='\t', chunksize=500000, keep_default_na=False):
            # Check positives
            pos_hits = chunk[chunk['entity_id'].isin(needed_positive_ids)]
            for mid, name, addr, c in zip(pos_hits['entity_id'], pos_hits['business_name'], pos_hits['business_address'], pos_hits['country']):
                s23_positives[mid] = (clean_name_v3(name), normalize_address(addr), str(name), c if c else 'US')

            # Sample candidates for hard negatives
            for mid, name, addr, c in zip(chunk['entity_id'], chunk['business_name'], chunk['business_address'], chunk['country']):
                if mid in needed_positive_ids:
                    continue
                c_val = c if c else 'US'
                raw_n_lower = str(name).lower()
                toks = [w for w in raw_n_lower.split() if w not in NAME_STOP_WORDS and len(w) >= 3]
                for t in toks[:3]:
                    sids = s1_token_index.get((t, c_val), [])
                    for sid in sids[:2]:
                        if s1_neg_counts[sid] < args.max_neg_per_entity:
                            true_set = gt_map.get(sid, set())
                            if mid not in true_set:
                                neg_pairs_by_s1[sid].append((
                                    clean_name_v3(name),
                                    normalize_address(addr),
                                    str(name)
                                ))
                                s1_neg_counts[sid] += 1
                                break

    print(f" Streamed in {time.time()-t_stream:.1f}s | Captured {len(s23_positives):,} positives & mined negatives for {len(neg_pairs_by_s1):,} S1", flush=True)

    # 4. Feature Extraction
    print("\n[Stage 4/5] Extracting 67 Features for Training Pairs ...", flush=True)
    t_feat = time.time()
    X_list = []
    y_list = []

    for sid, pre in s1_pre.items():
        (
            nc, ac, raw_n1, raw_w1_set,
            w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
            first_tok1, zip1, indic1,
            lead_num1, us_state1, fr_dept1, n1_4grams, c
        ) = pre

        # True Positives
        for mid in gt_map.get(sid, set()):
            if mid in s23_positives:
                cn2, ca2, raw_n2, c2 = s23_positives[mid]
                indic_flag = indic1 or bool(INDIC_RE.search(raw_n2))
                f = compute_pair_features_v10(
                    nc, ac, raw_n1, raw_w1_set,
                    cn2, ca2, raw_n2,
                    w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
                    first_tok1, zip1, indic_flag,
                    lead_num1, us_state1, fr_dept1, n1_4grams,
                    idf_dict=idf_dict
                )
                X_list.append(f)
                y_list.append(1)

        # Hard Negatives
        for (cn2, ca2, raw_n2) in neg_pairs_by_s1.get(sid, []):
            indic_flag = indic1 or bool(INDIC_RE.search(raw_n2))
            f = compute_pair_features_v10(
                nc, ac, raw_n1, raw_w1_set,
                cn2, ca2, raw_n2,
                w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
                first_tok1, zip1, indic_flag,
                lead_num1, us_state1, fr_dept1, n1_4grams,
                idf_dict=idf_dict
            )
            X_list.append(f)
            y_list.append(0)

    X_train = np.array(X_list, dtype=np.float32)
    y_train = np.array(y_list, dtype=np.int32)
    del X_list, y_list, s23_positives, neg_pairs_by_s1, s1_pre
    gc.collect()

    print(f" Training Matrix: {X_train.shape[0]:,} samples x {X_train.shape[1]} features in {time.time()-t_feat:.1f}s")
    print(f" Positive Class Ratio: {np.mean(y_train)*100:.2f}%", flush=True)

    # 5. Train XGBoost Booster
    print("\n[Stage 5/5] Fitting XGBoost Histogram Model (300 estimators) ...", flush=True)
    t_fit = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=9,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',
        n_jobs=24,
        random_state=42,
        eval_metric='logloss'
    )
    xgb_model.fit(X_train, y_train)
    print(f" XGBoost fitted in {time.time()-t_fit:.1f}s", flush=True)

    # Save Checkpoint
    save_path = os.path.join(args.cache_dir, args.save_model_name)
    ckpt = {
        'model': xgb_model,
        'feature_names': FEATURE_COLS_V10,
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_samples': len(y_train),
        'pos_ratio': float(np.mean(y_train))
    }
    with open(save_path, 'wb') as f:
        pickle.dump(ckpt, f)

    print(f"\n Checkpoint Saved Successfully: {save_path}")
    print(f" Total Pipeline Runtime: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
