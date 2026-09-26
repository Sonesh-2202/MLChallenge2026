#!/usr/bin/env python3
"""
Phase 2: Comprehensive Error Analysis for V3 Model
====================================================
Runs leakage-safe validation evaluation with detailed breakdowns.
"""
import os
import sys
import gc
import json
import time
import pickle
import heapq
import numpy as np
import pandas as pd
from collections import defaultdict

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


def build_index(s23_df, s1_n_keys, s1_a_keys, s1_c_keys, s1_names, max_p_comp=5000, max_p_single=3000):
    exact_name_idx, n_idx, a_idx, c_idx = {}, {}, {}, {}
    for r in s23_df.itertuples(index=False):
        eid, nc, ac = r.entity_id, r.name_clean, r.addr_clean
        if nc and nc in s1_names:
            exact_name_idx.setdefault(nc, []).append(eid)
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
    return {'exact': exact_name_idx, 'name': n_idx, 'addr': a_idx, 'comp': c_idx}


def retrieve_candidates(nc, ac, idx, comp_k=60, name_k=40, addr_k=40, exact_k=100):
    cands = set()
    if nc and nc in idx['exact']:
        cands.update(idx['exact'][nc][:exact_k])
    nk, ak, ck = extract_keys_v3(nc, ac)
    for keys_set, index, k_val in [(ck, idx['comp'], comp_k), (nk, idx['name'], name_k), (ak, idx['addr'], addr_k)]:
        if keys_set:
            scores = {}
            for k in keys_set:
                postings = index.get(k)
                if postings:
                    w = 1.0 / np.log1p(len(postings))
                    for mid in postings:
                        scores[mid] = scores.get(mid, 0.0) + w
            if scores:
                cands.update(m for m, _ in heapq.nlargest(k_val, scores.items(), key=lambda x: x[1]))
    return cands


def main():
    t_start = time.time()
    print("=" * 80)
    print(" PHASE 2: COMPREHENSIVE ERROR ANALYSIS")
    print("=" * 80)

    model_path = os.path.join(CACHE_DIR, 'lgbm_model_v3.pkl')
    with open(model_path, 'rb') as f:
        ckpt = pickle.load(f)
    model = ckpt['model']
    threshold = ckpt['threshold']
    print(f"Model loaded. Threshold: {threshold:.4f}")

    gt_df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv'),
                        sep='\t', keep_default_na=False)
    gt_map = {}
    for sid, mids in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']):
        gt_map[sid] = set(mids.split(',')) if mids else set()

    s1_all = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source1.tsv'),
                         sep='\t', usecols=['entity_id', 'country', 'business_name', 'business_address'],
                         keep_default_na=False)
    print(f"Loaded {len(s1_all):,} S1 records")

    np.random.seed(42)
    countries = ['US', 'India']
    val_s1_list = []
    n_train_s1 = 200000
    n_val_s1 = 25000

    for c in countries:
        s1_c = s1_all[s1_all['country'] == c].copy()
        s1_c = s1_c.sample(frac=1.0, random_state=42).reset_index(drop=True)
        c_ratio = len(s1_c) / len(s1_all)
        n_va = int(n_val_s1 * c_ratio)
        val_s1_list.append(s1_c.iloc[:n_va])
        print(f"  Validation {c}: {n_va:,} entities")

    val_s1_df = pd.concat(val_s1_list, ignore_index=True)
    del s1_all, val_s1_list
    gc.collect()

    val_s1_df['name_clean'] = val_s1_df['business_name'].apply(clean_name_v3)
    val_s1_df['addr_clean'] = val_s1_df['business_address'].apply(normalize_address)
    print(f"Validation set: {len(val_s1_df):,} entities")

    results = {
        'split_info': {
            'seed': 42, 'n_val': len(val_s1_df),
            'n_val_by_country': val_s1_df['country'].value_counts().to_dict(),
            'threshold': float(threshold),
        },
        'by_country': {},
        'overall': {},
    }

    all_entity_results = []
    s23_cols = ['entity_id', 'country', 'business_name', 'business_address']

    for country in countries:
        print(f"\n{'='*60}")
        print(f" Processing {country}")
        print('='*60)

        va_s1_c = val_s1_df[val_s1_df['country'] == country].copy()
        va_s1_ids = list(va_s1_c['entity_id'])

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
        print(f"  Loaded {len(s23_c):,} S2+S3 records")

        s23_c['name_clean'] = s23_c['business_name'].apply(clean_name_v3)
        s23_c['addr_clean'] = s23_c['business_address'].apply(normalize_address)
        s23_dict = dict(zip(s23_c['entity_id'], zip(s23_c['name_clean'], s23_c['addr_clean'])))

        s1_n_keys, s1_a_keys, s1_c_keys = set(), set(), set()
        s1_names = set(va_s1_c['name_clean'])
        for nc, ac in zip(va_s1_c['name_clean'], va_s1_c['addr_clean']):
            nk, ak, ck = extract_keys_v3(nc, ac)
            s1_n_keys.update(nk); s1_a_keys.update(ak); s1_c_keys.update(ck)

        c_index = build_index(s23_c, s1_n_keys, s1_a_keys, s1_c_keys, s1_names)
        del s23_c, s1_n_keys, s1_a_keys, s1_c_keys, s1_names; gc.collect()

        total_cand_count = 0
        blocking_missed_pairs = 0
        blocking_total_true_pairs = 0

        for i, row in enumerate(va_s1_c.itertuples(index=False)):
            sid = row.entity_id
            nc = row.name_clean
            ac = row.addr_clean
            true_set = gt_map.get(sid, set())
            is_singleton = len(true_set) == 0
            s2_true = {m for m in true_set if m.startswith('S2-')}
            s3_true = {m for m in true_set if m.startswith('S3-')}

            cands = retrieve_candidates(nc, ac, c_index)
            total_cand_count += len(cands)

            if true_set:
                blocking_total_true_pairs += len(true_set)
                blocking_missed = true_set - cands
                blocking_missed_pairs += len(blocking_missed)
            else:
                blocking_missed = set()

            pair_probs = {}
            if cands:
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
                    s23_info = s23_dict.get(mid)
                    if not s23_info: continue
                    n2, a2 = s23_info
                    feat = compute_pair_features_v3(nc, ac, n2, a2,
                        w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
                        first_token1, zip1)
                    features.append(feat)
                    valid_cands.append(mid)
                if features:
                    X = np.array(features, dtype=np.float32)
                    probs = model.predict(X, num_threads=4)
                    pair_probs = dict(zip(valid_cands, probs))

            pred_set = {mid for mid, p in pair_probs.items() if p >= threshold}
            f05_raw = compute_f05_per_entity(pred_set, true_set)

            tp = pred_set & true_set
            fp = pred_set - true_set
            fn = true_set - pred_set
            fn_blocking_miss = blocking_missed
            fn_classifier_miss = set()
            fn_classifier_miss_probs = {}
            for m in (fn - fn_blocking_miss):
                p = pair_probs.get(m, -1.0)
                if p >= 0:
                    fn_classifier_miss.add(m)
                    fn_classifier_miss_probs[m] = float(p)

            entity_result = {
                'sid': sid, 'country': country, 'is_singleton': is_singleton,
                'n_true_matches': len(true_set), 'n_s2_true': len(s2_true), 'n_s3_true': len(s3_true),
                'n_candidates': len(cands), 'n_pred': len(pred_set), 'f05_raw': f05_raw,
                'n_tp': len(tp), 'n_fp': len(fp), 'n_fn': len(fn),
                'n_fn_blocking': len(fn_blocking_miss), 'n_fn_classifier': len(fn_classifier_miss),
                'fn_classifier_probs': fn_classifier_miss_probs,
                'name_clean': nc, 'addr_clean': ac,
            }
            if fp:
                entity_result['fp_examples'] = [{'mid': m, 'prob': float(pair_probs.get(m, -1)),
                    'name': s23_dict.get(m, ('',''))[0], 'addr': s23_dict.get(m, ('',''))[1]} for m in list(fp)[:3]]
            if fn_classifier_miss:
                entity_result['fn_classifier_examples'] = [{'mid': m, 'prob': float(pair_probs.get(m, -1)),
                    'name': s23_dict.get(m, ('',''))[0], 'addr': s23_dict.get(m, ('',''))[1]} for m in list(fn_classifier_miss)[:3]]
            if fn_blocking_miss:
                entity_result['fn_blocking_examples'] = [{'mid': m,
                    'name': s23_dict.get(m, ('',''))[0], 'addr': s23_dict.get(m, ('',''))[1]} for m in list(fn_blocking_miss)[:3]]

            all_entity_results.append(entity_result)
            if (i + 1) % 2000 == 0 or (i + 1) == len(va_s1_c):
                print(f"  Progress: {i+1:,}/{len(va_s1_c):,}", flush=True)

        c_entities = [e for e in all_entity_results if e['country'] == country]
        c_singleton = [e for e in c_entities if e['is_singleton']]
        c_nonsingleton = [e for e in c_entities if not e['is_singleton']]
        blocking_recall = 1.0 - blocking_missed_pairs / max(1, blocking_total_true_pairs)

        results['by_country'][country] = {
            'n_entities': len(c_entities), 'n_singletons': len(c_singleton),
            'n_non_singletons': len(c_nonsingleton),
            'macro_f05_raw': float(np.mean([e['f05_raw'] for e in c_entities])),
            'singleton_f05': float(np.mean([e['f05_raw'] for e in c_singleton])) if c_singleton else 1.0,
            'nonsingleton_f05': float(np.mean([e['f05_raw'] for e in c_nonsingleton])) if c_nonsingleton else 0.0,
            'blocking_recall': float(blocking_recall),
            'blocking_missed_pairs': blocking_missed_pairs, 'blocking_total_true_pairs': blocking_total_true_pairs,
            'avg_candidates': total_cand_count / max(1, len(c_entities)),
            'total_tp': sum(e['n_tp'] for e in c_entities), 'total_fp': sum(e['n_fp'] for e in c_entities),
            'total_fn': sum(e['n_fn'] for e in c_entities),
            'total_fn_blocking': sum(e['n_fn_blocking'] for e in c_entities),
            'total_fn_classifier': sum(e['n_fn_classifier'] for e in c_entities),
        }
        cs = results['by_country'][country]
        print(f"\n  {country}: F0.5={cs['macro_f05_raw']:.4f}, BlockRecall={cs['blocking_recall']:.4f}")
        print(f"    TP={cs['total_tp']}, FP={cs['total_fp']}, FN={cs['total_fn']} (block={cs['total_fn_blocking']}, clf={cs['total_fn_classifier']})")
        del c_index, s23_dict, va_s1_c; gc.collect()

    # Overall
    all_f05 = [e['f05_raw'] for e in all_entity_results]
    all_singleton = [e for e in all_entity_results if e['is_singleton']]
    all_nonsingleton = [e for e in all_entity_results if not e['is_singleton']]
    results['overall'] = {
        'n_entities': len(all_entity_results),
        'macro_f05': float(np.mean(all_f05)),
        'singleton_f05': float(np.mean([e['f05_raw'] for e in all_singleton])) if all_singleton else 1.0,
        'nonsingleton_f05': float(np.mean([e['f05_raw'] for e in all_nonsingleton])) if all_nonsingleton else 0.0,
        'total_tp': sum(e['n_tp'] for e in all_entity_results),
        'total_fp': sum(e['n_fp'] for e in all_entity_results),
        'total_fn': sum(e['n_fn'] for e in all_entity_results),
        'total_fn_blocking': sum(e['n_fn_blocking'] for e in all_entity_results),
        'total_fn_classifier': sum(e['n_fn_classifier'] for e in all_entity_results),
        'n_singleton_errors': sum(1 for e in all_singleton if e['n_fp'] > 0),
    }
    all_fn_probs = []
    for e in all_entity_results:
        all_fn_probs.extend(e['fn_classifier_probs'].values())
    if all_fn_probs:
        fn_arr = np.array(all_fn_probs)
        results['fn_prob_distribution'] = {
            'count': len(fn_arr), 'mean': float(fn_arr.mean()), 'median': float(np.median(fn_arr)),
            'p25': float(np.percentile(fn_arr, 25)), 'p75': float(np.percentile(fn_arr, 75)),
            'p90': float(np.percentile(fn_arr, 90)),
            'above_0.9': int((fn_arr >= 0.9).sum()), 'above_0.8': int((fn_arr >= 0.8).sum()),
            'above_0.7': int((fn_arr >= 0.7).sum()), 'above_0.5': int((fn_arr >= 0.5).sum()),
            'below_0.3': int((fn_arr < 0.3).sum()),
        }

    # Example collections
    fp_ex = []; fn_ex = []; bn_ex = []
    for e in all_entity_results:
        for fp in e.get('fp_examples', []):
            fp_ex.append({'sid': e['sid'], 's1_name': e['name_clean'], 's1_addr': e['addr_clean'], **fp})
        for fn in e.get('fn_classifier_examples', []):
            fn_ex.append({'sid': e['sid'], 's1_name': e['name_clean'], 's1_addr': e['addr_clean'], **fn})
        for bn in e.get('fn_blocking_examples', []):
            bn_ex.append({'sid': e['sid'], 's1_name': e['name_clean'], 's1_addr': e['addr_clean'], **bn})
    fp_ex.sort(key=lambda x: x['prob'], reverse=True)
    fn_ex.sort(key=lambda x: x['prob'], reverse=True)
    results['top_fp_examples'] = fp_ex[:30]
    results['top_fn_classifier_examples'] = fn_ex[:30]
    results['top_blocking_miss_examples'] = bn_ex[:30]

    out_path = os.path.join(CACHE_DIR, 'phase2_error_analysis.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    o = results['overall']
    print("\n" + "=" * 80)
    print(f" OVERALL: F0.5={o['macro_f05']:.4f} | TP={o['total_tp']:,} FP={o['total_fp']:,} FN={o['total_fn']:,}")
    print(f"   FN breakdown: blocking={o['total_fn_blocking']:,}, classifier={o['total_fn_classifier']:,}")
    if 'fn_prob_distribution' in results:
        fn = results['fn_prob_distribution']
        print(f"   Classifier FN probs: mean={fn['mean']:.3f} median={fn['median']:.3f} above_0.9={fn['above_0.9']:,} above_0.8={fn['above_0.8']:,}")
    print(f" Saved to: {out_path}")
    print(f" Runtime: {time.time()-t_start:.1f}s")


if __name__ == '__main__':
    main()
