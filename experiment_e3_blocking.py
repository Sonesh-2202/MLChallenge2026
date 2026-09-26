#!/usr/bin/env python3
"""
Experiment E3: High-Recall V4 Blocking Optimization
===================================================
Evaluates candidate blocking recall and candidate volume on the exact fixed
24,999 validation split (14,994 US, 10,005 India) against full 10.3M S2+S3 pool.

Key Improvements over V3 Blocking:
 1. Fix 2-letter token drop bug (previously checked w.isupper() on lowercase strings!)
 2. Squashed name compound keys: n_sq:token1token2 (western + oceanhawk -> westernoceanhawk)
 3. Character 4-grams on single long words (len >= 6 instead of len >= 8)
 4. Address token pair conjunctions for India (PIN code + locality, e.g. pin_a:110001_delhi)
 5. Street number pair conjunctions (e.g. num_num:29_13) for Indian plot/door numbers
"""
import os
import sys
import gc
import time
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
    NAME_STOP_WORDS, ADDR_STOP_WORDS,
    clean_name_v3, extract_clean_numbers, extract_postal_code
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'student_resource', 'dataset')

# Pure 2-letter non-meaningful words to skip
STOP_2LETTER = {'to', 'in', 'on', 'at', 'of', 'by', 'an', 'as', 'is', 'it', 'or', 'so', 'we', 'he', 'my', 'up', 'no'}


def extract_keys_v4(nc, ac):
    n_keys, a_keys, c_keys = set(), set(), set()
    raw_n_words = nc.split()
    n_tokens = []

    for w in raw_n_words:
        if w not in NAME_STOP_WORDS:
            if len(w) >= 3:
                n_tokens.append(w)
            elif len(w) == 2 and w not in STOP_2LETTER and w.isalnum():
                n_tokens.append(w)

    nums = extract_clean_numbers(ac)
    a_tokens = [w for w in ac.split() if w not in ADDR_STOP_WORDS and len(w) >= 3]
    pin = extract_postal_code(ac)

    # 1. Name keys
    for t in n_tokens:
        if len(t) >= 4:
            n_keys.add('n:' + t)
            n_keys.add('ng:' + t[:4])
            n_keys.add('pfx:' + t[:3])
        elif len(t) == 3:
            n_keys.add('n3:' + t)
        elif len(t) == 2:
            n_keys.add('n2:' + t)

    # 2. Address keys
    for num in nums:
        a_keys.add('num:' + num)
    for at in a_tokens:
        a_keys.add('a:' + at)
    if pin:
        a_keys.add('pin:' + pin)

    # 3. Compound name keys & squashed compound keys
    for i in range(len(n_tokens)):
        for j in range(i + 1, min(i + 4, len(n_tokens))):
            t1, t2 = n_tokens[i], n_tokens[j]
            pair = tuple(sorted([t1, t2]))
            c_keys.add(f'np:{pair[0]}_{pair[1]}')
            # Squashed compound key (e.g. westernoceanhawk)
            c_keys.add(f'nsq:{t1}{t2}')
            c_keys.add(f'nsq:{t2}{t1}')

    # 4. Name + number cross keys
    for t in n_tokens[:3]:
        for num in nums:
            c_keys.add(f'n_num:{t}_{num}')

    # 5. Number + address token cross keys
    for num in nums:
        for at in a_tokens[:3]:
            c_keys.add(f'num_a:{num}_{at}')

    # 6. Name + address token cross keys
    target_a = list(dict.fromkeys(a_tokens[:2] + a_tokens[-2:])) if len(a_tokens) > 2 else a_tokens
    for t in n_tokens[:3]:
        for at in target_a:
            c_keys.add(f'n_a:{t}_{at}')

    # 7. PIN code + address token cross keys (vital for Indic script records where name doesn't match!)
    if pin:
        for at in a_tokens[:3]:
            c_keys.add(f'pin_a:{pin}_{at}')
        for num in nums:
            c_keys.add(f'pin_num:{pin}_{num}')

    # 8. Multi-number conjunction (for Indian door/plot numbers like 29-13-21)
    if len(nums) >= 2:
        sorted_nums = sorted(nums)[:4]
        for i in range(len(sorted_nums)):
            for j in range(i + 1, len(sorted_nums)):
                c_keys.add(f'num2:{sorted_nums[i]}_{sorted_nums[j]}')

    # 9. Single word character n-grams (lowered threshold from 8 to 6 chars)
    if len(n_tokens) == 1 and len(n_tokens[0]) >= 6:
        single_word = n_tokens[0]
        for i in range(min(12, len(single_word) - 3)):
            n_keys.add('ng:' + single_word[i:i+4])

    return n_keys, a_keys, c_keys


def build_country_index_v4(s23_df, s1_n_keys, s1_a_keys, s1_c_keys, s1_names,
                           max_p_comp=6000, max_p_single=4000):
    exact_idx, n_idx, a_idx, c_idx = {}, {}, {}, {}
    for r in s23_df.itertuples(index=False):
        eid, nc, ac = r.entity_id, r.name_clean, r.addr_clean
        if nc and nc in s1_names:
            exact_idx.setdefault(nc, []).append(eid)
        nk, ak, ck = extract_keys_v4(nc, ac)
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


def retrieve_candidates_v4(nc, ac, idx, comp_k=70, name_k=45, addr_k=45, exact_k=100):
    cands = set()
    exact_idx = idx['exact']
    c_idx = idx['comp']
    n_idx = idx['name']
    a_idx = idx['addr']

    if nc and nc in exact_idx:
        cands.update(exact_idx[nc][:exact_k])

    nk, ak, ck = extract_keys_v4(nc, ac)

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
    print("=" * 80)
    print(" EXPERIMENT E3: HIGH-RECALL V4 BLOCKING OPTIMIZATION")
    print("=" * 80)

    t0 = time.time()
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

    total_true_all = 0
    total_found_all = 0
    total_cands_all = 0

    results_by_country = {}
    s23_cols = ['entity_id', 'country', 'business_name', 'business_address']

    for country in countries:
        print(f"\nEvaluating Country: {country} ...", flush=True)
        va_s1_c = val_s1_df[val_s1_df['country'] == country]

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

        print(f"  Indexing {len(s23_c):,} S2+S3 records ...", flush=True)
        s1_n_keys, s1_a_keys, s1_c_keys = set(), set(), set()
        s1_names = set(va_s1_c['name_clean'])
        for nc, ac in zip(va_s1_c['name_clean'], va_s1_c['addr_clean']):
            nk, ak, ck = extract_keys_v4(nc, ac)
            s1_n_keys.update(nk); s1_a_keys.update(ak); s1_c_keys.update(ck)

        c_index = build_country_index_v4(s23_c, s1_n_keys, s1_a_keys, s1_c_keys, s1_names)
        del s23_c, s1_n_keys, s1_a_keys, s1_c_keys, s1_names; gc.collect()

        c_true = 0
        c_found = 0
        c_cands = 0

        print(f"  Retrieving candidates for {len(va_s1_c):,} {country} entities ...", flush=True)
        for i, row in enumerate(va_s1_c.itertuples(index=False)):
            sid = row.entity_id
            nc = row.name_clean
            ac = row.addr_clean
            true_set = gt_map.get(sid, set())

            cands = retrieve_candidates_v4(nc, ac, c_index)
            c_cands += len(cands)

            if true_set:
                c_true += len(true_set)
                c_found += len(true_set & cands)

            if (i + 1) % 5000 == 0 or (i + 1) == len(va_s1_c):
                print(f"    Progress: {i+1:,}/{len(va_s1_c):,} entities", flush=True)

        rec = c_found / max(1, c_true)
        avg_c = c_cands / max(1, len(va_s1_c))
        results_by_country[country] = {
            'recall': rec,
            'true_total': c_true,
            'true_found': c_found,
            'missed': c_true - c_found,
            'avg_cands': avg_c
        }
        print(f"\n  {country} V4 Blocking Recall: {rec:.4f} ({rec*100:.2f}%) | Found {c_found:,}/{c_true:,} | Missed {c_true-c_found:,} | Avg Cands: {avg_c:.1f}")

        total_true_all += c_true
        total_found_all += c_found
        total_cands_all += c_cands
        del c_index; gc.collect()

    overall_recall = total_found_all / max(1, total_true_all)
    overall_avg_cands = total_cands_all / max(1, len(val_s1_df))

    # Compare with V3 Baseline
    v3_recall = 0.9540
    v3_missed = 3968

    print("\n" + "=" * 80)
    print(" EXPERIMENT E3 RESULTS: BLOCKING RECALL COMPARISON")
    print("=" * 80)
    print(f"V3 Baseline Blocking Recall: {v3_recall:.4f} (95.40%) | Missed: {v3_missed:,} true pairs | Avg Cands: 143.6")
    print(f"V4 Enhanced Blocking Recall: {overall_recall:.4f} ({overall_recall*100:.2f}%) | Missed: {total_true_all-total_found_all:,} true pairs | Avg Cands: {overall_avg_cands:.1f}")
    print(f"Recall Delta:                {overall_recall - v3_recall:+.4f} ({(overall_recall - v3_recall)*100:+.2f}%)")
    print(f"Missed Matches Eliminated:   {v3_missed - (total_true_all - total_found_all):,} ({((v3_missed - (total_true_all - total_found_all))/v3_missed)*100:.1f}% reduction in blocking misses!)")
    print(f"Total Runtime:               {time.time()-t0:.1f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
