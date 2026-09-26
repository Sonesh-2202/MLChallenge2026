#!/usr/bin/env python3
"""
Experiment E1: Multi-Dimensional Threshold & Margin Calibration
==============================================================
Evaluates the V3 model on the exact fixed 24,999 validation split.
Scores all candidate pairs with multiprocessing, then sweeps:
 1. Global threshold tau in [0.70, 0.96]
 2. Country-specific thresholds: (tau_US, tau_India)
 3. Relative confidence margin filters: p >= (top_p - delta)
 4. Singleton barrier thresholds
"""
import os
import sys
import gc
import time
import pickle
import heapq
import multiprocessing as mp
import numpy as np
import pandas as pd
from collections import defaultdict

# Windows OpenMP configuration
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

SRC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'student_resource', 'code', 'business_entity_resolution', 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocess import normalize_address
from features_v3 import (
    FEATURE_COLS, NAME_STOP_WORDS, ADDR_STOP_WORDS,
    clean_name_v3, extract_clean_numbers, extract_postal_code,
    char_ngrams, compute_pair_features_v3
)
from evaluate import compute_f05_per_entity

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'student_resource', 'dataset')
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'student_resource', 'cache')

# Global worker state
_G_INDEX = None
_G_S23 = None
_G_MODEL = None


def init_worker(index_data, s23_data, model_bytes):
    global _G_INDEX, _G_S23, _G_MODEL
    _G_INDEX = index_data
    _G_S23 = s23_data
    _G_MODEL = pickle.loads(model_bytes)


def extract_keys_v3(nc, ac):
    n_keys, a_keys, c_keys = set(), set(), set()
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


def retrieve_candidates_v3(nc, ac, idx, comp_k=60, name_k=40, addr_k=40, exact_k=100):
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


def process_val_chunk(s1_records):
    global _G_INDEX, _G_S23, _G_MODEL
    results = []
    for sid, nc, ac in s1_records:
        cands = retrieve_candidates_v3(nc, ac, _G_INDEX)
        if not cands:
            results.append((sid, []))
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

        features, valid_cands = [], []
        for mid in sorted(cands):
            s23_info = _G_S23.get(mid)
            if not s23_info: continue
            n2, a2 = s23_info
            feat = compute_pair_features_v3(
                nc, ac, n2, a2,
                w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
                first_token1, zip1
            )
            features.append(feat)
            valid_cands.append(mid)

        if not features:
            results.append((sid, []))
            continue

        X = np.array(features, dtype=np.float32)
        probs = _G_MODEL.predict(X)
        # Keep pairs with prob >= 0.60 to save memory while enabling sweeps
        viable = [(mid, round(float(p), 4)) for mid, p in zip(valid_cands, probs) if p >= 0.60]
        results.append((sid, viable))
    return results


def build_country_index(s23_df, s1_n_keys, s1_a_keys, s1_c_keys, s1_names, max_p_comp=5000, max_p_single=3000):
    exact_idx, n_idx, a_idx, c_idx = {}, {}, {}, {}
    for r in s23_df.itertuples(index=False):
        eid, nc, ac = r.entity_id, r.name_clean, r.addr_clean
        if nc and nc in s1_names:
            exact_idx.setdefault(nc, []).append(eid)
        nk, ak, ck = extract_keys_v3(nc, ac)
        for k in nk:
            if k in s1_n_keys:
                lst = n_idx.setdefault(k, [])
                if len(lst) < max_p_single: lst.append(eid)
        for k in ak:
            if k in s1_a_keys:
                lst = a_idx.setdefault(k, [])
                if len(lst) < max_p_single: lst.append(eid)
        for k in ck:
            if k in s1_c_keys:
                lst = c_idx.setdefault(k, [])
                if len(lst) < max_p_comp: lst.append(eid)
    return {'exact': exact_idx, 'name': n_idx, 'addr': a_idx, 'comp': c_idx}


def score_validation_set(num_workers=16):
    cached_scores_path = os.path.join(CACHE_DIR, 'val_e1_scored_pairs.pkl')
    if os.path.exists(cached_scores_path):
        print(f"Loading cached validation scores from {cached_scores_path} ...", flush=True)
        with open(cached_scores_path, 'rb') as f:
            return pickle.load(f)

    print("Scoring validation set with 16 workers ...", flush=True)
    t0 = time.time()

    model_path = os.path.join(CACHE_DIR, 'lgbm_model_v3.pkl')
    with open(model_path, 'rb') as f:
        ckpt = pickle.load(f)
    model_bytes = pickle.dumps(ckpt['model'])

    gt_df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv'),
                        sep='\t', keep_default_na=False)
    gt_map = {}
    for sid, mids in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']):
        gt_map[sid] = set(mids.split(',')) if mids else set()

    s1_all = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source1.tsv'),
                         sep='\t', usecols=['entity_id', 'country', 'business_name', 'business_address'],
                         keep_default_na=False)

    countries = ['US', 'India']
    val_s1_list = []
    n_val_s1 = 25000
    for c in countries:
        s1_c = s1_all[s1_all['country'] == c].copy()
        s1_c = s1_c.sample(frac=1.0, random_state=42).reset_index(drop=True)
        c_ratio = len(s1_c) / len(s1_all)
        n_va = int(n_val_s1 * c_ratio)
        val_s1_list.append(s1_c.iloc[:n_va])

    val_s1_df = pd.concat(val_s1_list, ignore_index=True)
    del s1_all, val_s1_list
    gc.collect()

    val_s1_df['name_clean'] = val_s1_df['business_name'].apply(clean_name_v3)
    val_s1_df['addr_clean'] = val_s1_df['business_address'].apply(normalize_address)

    all_entity_scores = {}
    entity_country = dict(zip(val_s1_df['entity_id'], val_s1_df['country']))
    s23_cols = ['entity_id', 'country', 'business_name', 'business_address']

    for country in countries:
        print(f"\nProcessing {country} partition ...", flush=True)
        va_s1_c = val_s1_df[val_s1_df['country'] == country]
        s1_records = list(zip(va_s1_c['entity_id'], va_s1_c['name_clean'], va_s1_c['addr_clean']))

        s2_list = []
        for chunk in pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source2.tsv'),
                                  sep='\t', chunksize=1000000, usecols=s23_cols, keep_default_na=False):
            sub = chunk[chunk['country'] == country][['entity_id', 'business_name', 'business_address']]
            if len(sub) > 0: s2_list.append(sub)
        for chunk in pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source3.tsv'),
                                  sep='\t', chunksize=1000000, usecols=s23_cols, keep_default_na=False):
            sub = chunk[chunk['country'] == country][['entity_id', 'business_name', 'business_address']]
            if len(sub) > 0: s2_list.append(sub)

        s23_c = pd.concat(s2_list, ignore_index=True).drop_duplicates(subset=['entity_id'])
        del s2_list; gc.collect()
        s23_c['name_clean'] = s23_c['business_name'].apply(clean_name_v3)
        s23_c['addr_clean'] = s23_c['business_address'].apply(normalize_address)
        s23_dict = dict(zip(s23_c['entity_id'], zip(s23_c['name_clean'], s23_c['addr_clean'])))

        s1_n_keys, s1_a_keys, s1_c_keys = set(), set(), set()
        s1_names = set(va_s1_c['name_clean'])
        for nc, ac in zip(va_s1_c['name_clean'], va_s1_c['addr_clean']):
            nk, ak, ck = extract_keys_v3(nc, ac)
            s1_n_keys.update(nk); s1_a_keys.update(ak); s1_c_keys.update(ck)

        c_index = build_country_index(s23_c, s1_n_keys, s1_a_keys, s1_c_keys, s1_names)
        del s23_c, s1_n_keys, s1_a_keys, s1_c_keys, s1_names; gc.collect()

        chunk_size = 250
        chunks = [s1_records[i:i + chunk_size] for i in range(0, len(s1_records), chunk_size)]
        print(f"  Dispatching {len(chunks)} chunks across {num_workers} workers ...", flush=True)

        with mp.Pool(processes=num_workers, initializer=init_worker,
                     initargs=(c_index, s23_dict, model_bytes)) as pool:
            for chunk_res in pool.imap_unordered(process_val_chunk, chunks, chunksize=2):
                for sid, pairs in chunk_res:
                    all_entity_scores[sid] = pairs

        del c_index, s23_dict, chunks; gc.collect()

    output_data = {
        'scores': all_entity_scores,
        'country': entity_country,
        'gt_map': gt_map,
        'val_ids': list(val_s1_df['entity_id'])
    }
    with open(cached_scores_path, 'wb') as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Scoring complete in {time.time()-t0:.1f}s. Saved to {cached_scores_path}", flush=True)
    return output_data


def evaluate_decision_rule(data, th_us, th_india, margin=None, max_matches=12, singleton_barrier=None):
    scores = data['scores']
    country = data['country']
    gt_map = data['gt_map']
    val_ids = data['val_ids']

    f05_list = []
    f05_us = []
    f05_india = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for sid in val_ids:
        c = country[sid]
        th = th_us if c == 'US' else th_india
        true_set = gt_map.get(sid, set())
        pairs = scores.get(sid, [])

        filtered = [mid for mid, p in pairs if p >= th]
        if filtered and margin is not None:
            top_p = max(p for mid, p in pairs if mid in filtered)
            filtered = [mid for mid, p in pairs if mid in filtered and p >= (top_p - margin)]

        if filtered and singleton_barrier is not None:
            top_p = max(p for mid, p in pairs if mid in filtered)
            if top_p < singleton_barrier:
                filtered = []

        filtered = filtered[:max_matches]
        pred_set = set(filtered)

        score = compute_f05_per_entity(pred_set, true_set)
        f05_list.append(score)
        if c == 'US':
            f05_us.append(score)
        else:
            f05_india.append(score)

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

    macro_f05 = np.mean(f05_list)
    macro_us = np.mean(f05_us)
    macro_india = np.mean(f05_india)
    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)

    return {
        'macro_f05': macro_f05,
        'macro_us': macro_us,
        'macro_india': macro_india,
        'precision': precision,
        'recall': recall,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
        'th_us': th_us,
        'th_india': th_india,
        'margin': margin,
        'barrier': singleton_barrier
    }


def main():
    print("=" * 80)
    print(" EXPERIMENT E1: THRESHOLD & MARGIN CALIBRATION")
    print("=" * 80)

    data = score_validation_set(num_workers=16)

    # 1. Baseline Verification
    print("\n--- Verifying V3 Baseline (Flat tau=0.94, no margin) ---", flush=True)
    base_res = evaluate_decision_rule(data, th_us=0.94, th_india=0.94, margin=None)
    print(f"  V3 Baseline: Macro F0.5 = {base_res['macro_f05']:.4f} (US: {base_res['macro_us']:.4f}, India: {base_res['macro_india']:.4f})")
    print(f"    Precision = {base_res['precision']*100:.2f}%, Recall = {base_res['recall']*100:.2f}% | TP={base_res['total_tp']:,} FP={base_res['total_fp']:,} FN={base_res['total_fn']:,}")

    # 2. Check Fast Pipeline V3 config (tau=0.94 with 0.20 margin filter)
    print("\n--- Testing Production Fast Pipeline V3 Rule (tau=0.94 with margin=0.20) ---", flush=True)
    pipe_res = evaluate_decision_rule(data, th_us=0.94, th_india=0.94, margin=0.20)
    print(f"  Production Pipeline: Macro F0.5 = {pipe_res['macro_f05']:.4f} (US: {pipe_res['macro_us']:.4f}, India: {pipe_res['macro_india']:.4f})")
    print(f"    Precision = {pipe_res['precision']*100:.2f}%, Recall = {pipe_res['recall']*100:.2f}% | TP={pipe_res['total_tp']:,} FP={pipe_res['total_fp']:,} FN={pipe_res['total_fn']:,}")

    # 3. Flat Threshold Sweep
    print("\n--- Grid Sweep: Flat Thresholds tau in [0.70, 0.96] ---", flush=True)
    best_flat = base_res
    flat_results = []
    for th in np.arange(0.70, 0.97, 0.01):
        th = round(th, 2)
        res = evaluate_decision_rule(data, th_us=th, th_india=th, margin=None)
        flat_results.append(res)
        if res['macro_f05'] > best_flat['macro_f05']:
            best_flat = res
        print(f"  tau={th:.2f} -> Macro F0.5={res['macro_f05']:.4f} (US={res['macro_us']:.4f}, IN={res['macro_india']:.4f}, P={res['precision']*100:.2f}%, R={res['recall']*100:.2f}%)")

    # 4. Country-Specific Threshold Grid Sweep
    print("\n--- 2D Grid Sweep: Independent (tau_US, tau_India) ---", flush=True)
    best_2d = best_flat
    grid_results = []
    us_range = np.arange(0.88, 0.96, 0.01)
    in_range = np.arange(0.85, 0.96, 0.01)

    for u_th in us_range:
        for i_th in in_range:
            u_th = round(u_th, 2)
            i_th = round(i_th, 2)
            res = evaluate_decision_rule(data, th_us=u_th, th_india=i_th, margin=None)
            grid_results.append(res)
            if res['macro_f05'] > best_2d['macro_f05']:
                best_2d = res

    print(f"\n  BEST Independent Thresholds: tau_US={best_2d['th_us']:.2f}, tau_India={best_2d['th_india']:.2f}")
    print(f"  Macro F0.5 = {best_2d['macro_f05']:.4f} (US: {best_2d['macro_us']:.4f}, India: {best_2d['macro_india']:.4f})")
    print(f"  Precision = {best_2d['precision']*100:.2f}%, Recall = {best_2d['recall']*100:.2f}% | TP={best_2d['total_tp']:,} FP={best_2d['total_fp']:,} FN={best_2d['total_fn']:,}")
    print(f"  Delta over V3 Baseline: {best_2d['macro_f05'] - base_res['macro_f05']:+.4f} ({(best_2d['macro_f05'] - base_res['macro_f05'])*100:+.2f}%)")

    # 5. Margin Filter Sweep on Top of Best 2D Thresholds
    print("\n--- Margin Filter Sweep on Best 2D Thresholds ---", flush=True)
    best_overall = best_2d
    for margin in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, None]:
        res = evaluate_decision_rule(data, th_us=best_2d['th_us'], th_india=best_2d['th_india'], margin=margin)
        margin_str = f"{margin:.2f}" if margin is not None else "None"
        print(f"  margin={margin_str} -> Macro F0.5={res['macro_f05']:.4f} (P={res['precision']*100:.2f}%, R={res['recall']*100:.2f}%)")
        if res['macro_f05'] > best_overall['macro_f05']:
            best_overall = res

    # 6. Save Best Configuration
    out_cfg = {
        'best_config': {
            'th_us': best_overall['th_us'],
            'th_india': best_overall['th_india'],
            'margin': best_overall['margin'],
            'macro_f05': best_overall['macro_f05'],
            'macro_us': best_overall['macro_us'],
            'macro_india': best_overall['macro_india'],
            'precision': best_overall['precision'],
            'recall': best_overall['recall'],
            'delta_over_v3': best_overall['macro_f05'] - base_res['macro_f05']
        }
    }
    cfg_path = os.path.join(CACHE_DIR, 'optimal_decision_rule_v3.json')
    import json
    with open(cfg_path, 'w') as f:
        json.dump(out_cfg, f, indent=2)
    print(f"\nSaved optimal decision rule to {cfg_path}")


if __name__ == '__main__':
    main()
