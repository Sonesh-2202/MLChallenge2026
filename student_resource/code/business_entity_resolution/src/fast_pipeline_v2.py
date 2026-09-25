#!/usr/bin/env python3
"""
=============================================================================
 ULTRA-FAST PARALLEL BUSINESS ENTITY RESOLUTION PIPELINE (V2 PRODUCTION)
=============================================================================
Uses:
 - Country-Partitioned 3-Way Compound Inverted-Index Blocking (Name + Address + Conjunctions)
 - C-Accelerated RapidFuzz 30-Feature Engineering
 - Deep Regularized LightGBM (v2 checkpoint) with calibrated Macro F0.5 threshold
 - Relative Confidence Thresholding & Max Match Outlier Suppression
 - Partitioned Disk-Streaming (Zero-OOM, sub-second recovery)
 - Validated on official competition submission validator
"""
import os
import sys

# Crucial Windows OpenMP configuration
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import gc
import time
import pickle
import argparse
import subprocess
import heapq
import multiprocessing as mp
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocess import normalize_address, STOP_WORDS
from features_v2 import FEATURE_COLS, clean_name_enhanced, extract_clean_numbers, compute_pair_features, char_ngrams

# Global worker state
_G_INDEX = None
_G_S23 = None
_G_MODEL = None
_G_THRESHOLD = 0.70
_G_MAX_MATCHES_PER_ENTITY = 12


def init_worker(index_data, s23_data, model_bytes, threshold, max_matches=12):
    """Initialize worker process with index, lookup dict, and model."""
    global _G_INDEX, _G_S23, _G_MODEL, _G_THRESHOLD, _G_MAX_MATCHES_PER_ENTITY
    _G_INDEX = index_data
    _G_S23 = s23_data
    _G_MODEL = pickle.loads(model_bytes)
    _G_THRESHOLD = threshold
    _G_MAX_MATCHES_PER_ENTITY = max_matches


def extract_keys(nc, ac):
    """Extract 3-way compound blocking keys."""
    n_keys = set()
    a_keys = set()
    c_keys = set()

    n_tokens = [w for w in nc.split() if w not in STOP_WORDS and len(w) >= 3]
    nums = extract_clean_numbers(ac)
    a_tokens = [w for w in ac.split() if w not in STOP_WORDS and len(w) >= 4]

    for t in n_tokens:
        if len(t) >= 4:
            n_keys.add('n:' + t)
            n_keys.add('ng:' + t[:4])
        elif len(t) == 3:
            n_keys.add('n3:' + t)

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

    return n_keys, a_keys, c_keys


def process_s1_chunk_parallel(s1_records):
    """
    Process a chunk of S1 records in parallel:
     1. Retrieve candidates using 3-way compound blocking
     2. Vectorized 30-feature extraction with C-accelerated RapidFuzz
     3. Batch LightGBM prediction
     4. Filter matches by calibrated threshold + top-match relative confidence filtering
    s1_records: list of tuples (sid, name_clean, addr_clean)
    """
    idx = _G_INDEX
    s23_dict = _G_S23
    model = _G_MODEL
    threshold = _G_THRESHOLD
    max_m_cap = _G_MAX_MATCHES_PER_ENTITY

    exact_idx = idx['exact']
    c_idx = idx['comp']
    n_idx = idx['name']
    a_idx = idx['addr']

    chunk_candidates = []
    total_pairs = 0

    # 1. Candidate Retrieval
    for sid, nc, ac in s1_records:
        cands = set()

        if nc and nc in exact_idx:
            cands.update(exact_idx[nc][:100])

        nk, ak, ck = extract_keys(nc, ac)

        if ck:
            c_scores = {}
            for k in ck:
                postings = c_idx.get(k)
                if postings:
                    w = 1.0 / np.log1p(len(postings))
                    for mid in postings:
                        c_scores[mid] = c_scores.get(mid, 0.0) + w
            if c_scores:
                cands.update(m for m, _ in heapq.nlargest(40, c_scores.items(), key=lambda x: x[1]))

        if nk:
            n_scores = {}
            for k in nk:
                postings = n_idx.get(k)
                if postings:
                    w = 1.0 / np.log1p(len(postings))
                    for mid in postings:
                        n_scores[mid] = n_scores.get(mid, 0.0) + w
            if n_scores:
                cands.update(m for m, _ in heapq.nlargest(30, n_scores.items(), key=lambda x: x[1]))

        if ak:
            a_scores = {}
            for k in ak:
                postings = a_idx.get(k)
                if postings:
                    w = 1.0 / np.log1p(len(postings))
                    for mid in postings:
                        a_scores[mid] = a_scores.get(mid, 0.0) + w
            if a_scores:
                cands.update(m for m, _ in heapq.nlargest(30, a_scores.items(), key=lambda x: x[1]))

        cand_list = sorted(cands)
        chunk_candidates.append((sid, nc, ac, cand_list))
        total_pairs += len(cand_list)

    if total_pairs == 0:
        return [(sid, []) for sid, _, _, _ in chunk_candidates], [(sid, c) for sid, _, _, c in chunk_candidates]

    # 2. Build 30-Feature Matrix
    X = np.empty((total_pairs, len(FEATURE_COLS)), dtype=np.float32)
    pair_meta = []  # list of (sid, mid)
    pair_idx = 0

    for sid, nc, ac, cand_list in chunk_candidates:
        if not cand_list:
            continue
        w1_set = set(nc.split()) if nc else set()
        aw1_set = set(ac.split()) if ac else set()
        nums1 = extract_clean_numbers(ac)
        has_a1 = bool(ac and ac != 'nan')
        n1_ngrams = char_ngrams(nc, 3)
        a1_ngrams = char_ngrams(ac, 3) if has_a1 else set()

        for mid in cand_list:
            pair_meta.append((sid, mid))
            s23_info = s23_dict.get(mid)
            if s23_info is None:
                X[pair_idx, :] = 0.0
                pair_idx += 1
                continue
            n2, a2 = s23_info
            X[pair_idx, :] = compute_pair_features(nc, ac, n2, a2, w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams)
            pair_idx += 1

    # 3. Vectorized Prediction
    probs = model.predict(X, num_threads=1)

    # 4. Map matches >= threshold with outlier protection
    # sid -> list of (mid, prob)
    matches_map = {sid: [] for sid, _, _, _ in chunk_candidates}
    for (sid, mid), p in zip(pair_meta, probs):
        if p >= threshold:
            matches_map[sid].append((mid, float(p)))

    # Apply outlier guard: if an entity has matches, filter out matches that are much weaker than top match
    final_matches = []
    for sid, _, _, _ in chunk_candidates:
        cand_matches = matches_map[sid]
        if not cand_matches:
            final_matches.append((sid, []))
            continue

        # Sort by probability descending
        cand_matches.sort(key=lambda x: x[1], reverse=True)
        top_p = cand_matches[0][1]

        # Keep matches that are within 0.20 of top match and max cap
        filtered = [mid for mid, p in cand_matches if (top_p - p) <= 0.20][:max_m_cap]
        final_matches.append((sid, filtered))

    candidates_out = [(sid, cands) for sid, _, _, cands in chunk_candidates]
    return final_matches, candidates_out


def save_country_partition(country, matches_list, candidates_list, partition_dir):
    """Write intermediate country partition to disk."""
    os.makedirs(partition_dir, exist_ok=True)
    m_path = os.path.join(partition_dir, f'matches_{country}.tsv')
    c_path = os.path.join(partition_dir, f'candidates_{country}.tsv')

    with open(m_path, 'w', encoding='utf-8', newline='') as f:
        for sid, mids in matches_list:
            f.write(f"{sid}\t{','.join(sorted(mids)) if mids else ''}\n")

    with open(c_path, 'w', encoding='utf-8', newline='') as f:
        for sid, cids in candidates_list:
            f.write(f"{sid}\t{','.join(sorted(cids)) if cids else ''}\n")

    print(f"   Saved partition files for {country} to {partition_dir}", flush=True)


def build_country_index_s23(s23_df, s1_names, s1_n_keys, s1_a_keys, s1_c_keys, max_p=2500):
    """Build country index for test data."""
    exact_name_idx = {}
    n_idx = {}
    a_idx = {}
    c_idx = {}

    for r in s23_df.itertuples(index=False):
        eid = r.entity_id
        nc = r.name_clean
        ac = r.addr_clean

        if nc and nc in s1_names:
            exact_name_idx.setdefault(nc, []).append(eid)

        nk, ak, ck = extract_keys(nc, ac)
        for k in nk:
            if k in s1_n_keys:
                lst = n_idx.setdefault(k, [])
                if len(lst) < max_p:
                    lst.append(eid)
        for k in ak:
            if k in s1_a_keys:
                lst = a_idx.setdefault(k, [])
                if len(lst) < max_p:
                    lst.append(eid)
        for k in ck:
            if k in s1_c_keys:
                lst = c_idx.setdefault(k, [])
                if len(lst) < max_p:
                    lst.append(eid)

    return {'exact': exact_name_idx, 'name': n_idx, 'addr': a_idx, 'comp': c_idx}


def main():
    parser = argparse.ArgumentParser(description="Parallel Business Entity Resolution Pipeline V2")
    parser.add_argument('--data-dir', default=os.path.join(SRC_DIR, '..', '..', '..', 'dataset'))
    parser.add_argument('--output-dir', default=os.path.join(SRC_DIR, '..', '..', '..', 'output'))
    parser.add_argument('--cache-dir', default=os.path.join(SRC_DIR, '..', '..', '..', 'cache'))
    parser.add_argument('--n-workers', type=int, default=14, help="Number of worker processes (default: 14)")
    parser.add_argument('--chunk-size', type=int, default=2500, help="S1 records per worker task")
    parser.add_argument('--model-checkpoint', default='')
    args = parser.parse_args()

    t_global_start = time.time()
    print("=" * 78, flush=True)
    print(" ULTRA-FAST PARALLEL BUSINESS ENTITY RESOLUTION PIPELINE (V2 RUN)", flush=True)
    print("=" * 78, flush=True)
    print(f" Data Dir:    {os.path.abspath(args.data_dir)}", flush=True)
    print(f" Output Dir:  {os.path.abspath(args.output_dir)}", flush=True)
    print(f" Cache Dir:   {os.path.abspath(args.cache_dir)}", flush=True)
    print(f" Workers:     {args.n_workers} parallel processes", flush=True)
    print(f" Chunk Size:  {args.chunk_size} entities/chunk", flush=True)

    partition_dir = os.path.join(args.output_dir, 'partitions_v2')
    os.makedirs(partition_dir, exist_ok=True)

    # Load Model Checkpoint
    checkpoint_path = args.model_checkpoint or os.path.join(args.cache_dir, 'lgbm_model_v2.pkl')
    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Model checkpoint not found at {checkpoint_path}!", flush=True)
        sys.exit(1)

    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    model = ckpt['model']
    threshold = float(ckpt['threshold'])
    model_bytes = pickle.dumps(model)
    print(f"\n Loaded trained LightGBM model from {checkpoint_path}", flush=True)
    print(f"   Optimal Macro F0.5 Threshold: {threshold:.2f}", flush=True)
    if 'val_f05' in ckpt:
        print(f"   Validation Macro F0.5:        {ckpt['val_f05']*100:.2f}%", flush=True)
    if 'blocking_recall' in ckpt:
        print(f"   Validation Blocking Recall:   {ckpt['blocking_recall']*100:.2f}%", flush=True)

    # Load Complete Test Source 1
    print("\n" + "=" * 78, flush=True)
    print(" LOADING TEST SOURCE 1", flush=True)
    print("=" * 78, flush=True)
    s1_path = os.path.join(args.data_dir, 'test', 'test_source1.tsv')
    print(f" Reading {s1_path} ...", flush=True)
    t0 = time.time()
    test_s1 = pd.read_csv(s1_path, sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country'], keep_default_na=False)
    all_test_s1_ids = list(test_s1['entity_id'])
    print(f" Loaded {len(test_s1):,} test reference S1 entities in {time.time()-t0:.2f}s", flush=True)

    print(" Normalizing S1 names and addresses with domain stripper ...", flush=True)
    t0 = time.time()
    test_s1['name_clean'] = test_s1['business_name'].apply(clean_name_enhanced)
    test_s1['addr_clean'] = test_s1['business_address'].apply(normalize_address)
    print(f" Normalized {len(test_s1):,} S1 records in {time.time()-t0:.2f}s", flush=True)

    test_countries = sorted(test_s1['country'].unique())
    print(f" Test countries to process: {test_countries}", flush=True)

    # Process each country in parallel
    for country in test_countries:
        m_part = os.path.join(partition_dir, f'matches_{country}.tsv')
        c_part = os.path.join(partition_dir, f'candidates_{country}.tsv')

        s1_c = test_s1[test_s1['country'] == country]
        n_s1_c = len(s1_c)

        if os.path.exists(m_part) and os.path.exists(c_part):
            with open(m_part, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            if lines == n_s1_c:
                print(f"\n [*] Country {country} partition already exists ({lines:,} records). Skipping recomputation!", flush=True)
                continue

        print("\n" + "=" * 78, flush=True)
        print(f" PROCESSING COUNTRY: {country.upper()}", flush=True)
        print("=" * 78, flush=True)
        t_country_start = time.time()

        s1_records = list(zip(s1_c['entity_id'], s1_c['name_clean'], s1_c['addr_clean']))
        print(f"   S1 entities in {country}: {n_s1_c:,}", flush=True)

        # 1. Load S2 and S3 for this country
        print(f"   Loading test S2 & S3 for {country} ...", flush=True)
        t_load = time.time()
        s2_list, s3_list = [], []
        cols = ['entity_id', 'country', 'business_name', 'business_address']

        for chunk in pd.read_csv(os.path.join(args.data_dir, 'test', 'test_source2.tsv'), sep='\t', chunksize=500000, dtype=str, usecols=cols, keep_default_na=False):
            sub = chunk[chunk['country'] == country][['entity_id', 'business_name', 'business_address']]
            if len(sub) > 0:
                s2_list.append(sub)

        for chunk in pd.read_csv(os.path.join(args.data_dir, 'test', 'test_source3.tsv'), sep='\t', chunksize=500000, dtype=str, usecols=cols, keep_default_na=False):
            sub = chunk[chunk['country'] == country][['entity_id', 'business_name', 'business_address']]
            if len(sub) > 0:
                s3_list.append(sub)

        s23_c = pd.concat(s2_list + s3_list, ignore_index=True).drop_duplicates(subset=['entity_id'])
        del s2_list, s3_list
        gc.collect()

        print(f"   Loaded {len(s23_c):,} unique S2+S3 records in {time.time()-t_load:.2f}s", flush=True)

        # 2. Normalize S23 & Build 3-Way Compound Index
        print(f"   Normalizing S2+S3 ...", flush=True)
        t_norm = time.time()
        s23_c['name_clean'] = s23_c['business_name'].apply(clean_name_enhanced)
        s23_c['addr_clean'] = s23_c['business_address'].apply(normalize_address)
        print(f"   Normalized in {time.time()-t_norm:.2f}s", flush=True)

        print(f"   Extracting S1 query keys for {country} ...", flush=True)
        s1_names = set(s1_c['name_clean'])
        s1_n_keys, s1_a_keys, s1_c_keys = set(), set(), set()
        for nc, ac in zip(s1_c['name_clean'], s1_c['addr_clean']):
            nk, ak, ck = extract_keys(nc, ac)
            s1_n_keys.update(nk)
            s1_a_keys.update(ak)
            s1_c_keys.update(ck)

        print(f"   Building 3-way compound inverted index ...", flush=True)
        t_idx = time.time()
        idx_data = build_country_index_s23(s23_c, s1_names, s1_n_keys, s1_a_keys, s1_c_keys)
        s23_dict = dict(zip(s23_c['entity_id'], zip(s23_c['name_clean'], s23_c['addr_clean'])))
        print(f"   Index built in {time.time()-t_idx:.2f}s", flush=True)
        del s23_c, s1_names, s1_n_keys, s1_a_keys, s1_c_keys
        gc.collect()

        # 3. Create Chunks for Multiprocessing
        chunks = [s1_records[i:i + args.chunk_size] for i in range(0, n_s1_c, args.chunk_size)]
        print(f"   Partitioned {n_s1_c:,} entities into {len(chunks):,} parallel chunks", flush=True)

        # 4. Run Multiprocessing Pool
        print(f"   Spawning {args.n_workers} worker processes ...", flush=True)
        t_par = time.time()

        pool = mp.Pool(
            processes=args.n_workers,
            initializer=init_worker,
            initargs=(idx_data, s23_dict, model_bytes, threshold)
        )
        try:
            chunk_results = []
            async_results = [pool.apply_async(process_s1_chunk_parallel, (chunk,)) for chunk in chunks]

            completed = 0
            n_chunks = len(async_results)
            last_report = time.time()

            for ar in async_results:
                chunk_results.append(ar.get())
                completed += 1
                if completed % max(1, n_chunks // 20) == 0 or completed == n_chunks or time.time() - last_report >= 5:
                    pct = completed / n_chunks * 100
                    elap = time.time() - t_par
                    rate = (completed * args.chunk_size) / max(0.1, elap)
                    eta = (n_chunks - completed) * (elap / completed) if completed > 0 else 0
                    print(f"     Progress: {completed}/{n_chunks} chunks ({pct:.1f}%) | {rate:.0f} entities/s | ETA: {eta:.0f}s", flush=True)
                    last_report = time.time()
        finally:
            pool.terminate()
            pool.join()

        t_par_end = time.time()
        print(f"   Country {country} completed in {t_par_end - t_country_start:.2f}s!", flush=True)

        country_matches = [item for m_list, _ in chunk_results for item in m_list]
        country_cands = [item for _, c_list in chunk_results for item in c_list]

        save_country_partition(country, country_matches, country_cands, partition_dir)
        total_m = sum(len(m[1]) for m in country_matches)
        total_c = sum(len(c[1]) for c in country_cands)
        print(f"   Country {country} Total Matches: {total_m:,} | Candidates: {total_c:,}", flush=True)

        del idx_data, s23_dict, chunk_results, async_results, s1_records, s1_c, country_matches, country_cands
        gc.collect()

    del test_s1
    gc.collect()

    # Merge Partitions into Final Submission TSV Files
    print("\n" + "=" * 78, flush=True)
    print(" MERGING PARTITIONS INTO FINAL SUBMISSION FILES", flush=True)
    print("=" * 78, flush=True)

    matching_out = os.path.join(args.output_dir, 'matching_results.tsv')
    candidate_out = os.path.join(args.output_dir, 'candidate_pairs.tsv')

    print(" Reading partitions ...", flush=True)
    matches_lookup = {}
    candidates_lookup = {}

    for country in test_countries:
        m_part = os.path.join(partition_dir, f'matches_{country}.tsv')
        c_part = os.path.join(partition_dir, f'candidates_{country}.tsv')
        with open(m_part, 'r', encoding='utf-8') as f:
            for line in f:
                sid, sep, rest = line.partition('\t')
                matches_lookup[sid] = rest.rstrip('\r\n')
        with open(c_part, 'r', encoding='utf-8') as f:
            for line in f:
                sid, sep, rest = line.partition('\t')
                candidates_lookup[sid] = rest.rstrip('\r\n')

    print(f" Writing {matching_out} ...", flush=True)
    with open(matching_out, 'w', encoding='utf-8', newline='') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_test_s1_ids:
            f.write(f"{sid}\t{matches_lookup.get(sid, '')}\n")

    print(f" Writing {candidate_out} ...", flush=True)
    with open(candidate_out, 'w', encoding='utf-8', newline='') as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_test_s1_ids:
            f.write(f"{sid}\t{candidates_lookup.get(sid, '')}\n")

    n_total_entities = len(all_test_s1_ids)
    n_matches = sum(1 for v in matches_lookup.values() if v)
    n_singletons = sum(1 for v in matches_lookup.values() if not v)

    print(f"\n Final Submission Statistics:", flush=True)
    print(f"   Total Test S1 Entities: {n_total_entities:,}", flush=True)
    print(f"   Entities with Matches:  {n_matches:,}", flush=True)
    print(f"   Singletons (no match):  {n_singletons:,} ({n_singletons/n_total_entities*100:.2f}%)", flush=True)
    print(f"   Macro F0.5 Threshold:   {threshold:.2f}", flush=True)

    # Validate Submission
    print("\n" + "=" * 78, flush=True)
    print(" RUNNING OFFICIAL SUBMISSION VALIDATOR", flush=True)
    print("=" * 78, flush=True)

    validator_script = os.path.join(SRC_DIR, '..', '..', '..', 'utils', 'validate_submission.py')
    test_dir = os.path.join(args.data_dir, 'test')

    if os.path.exists(validator_script):
        cmd = [
            sys.executable, validator_script,
            '--matching', matching_out,
            '--candidate', candidate_out,
            '--test-dir', test_dir
        ]
        print(f" Running validator: {' '.join(cmd)}\n", flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout, flush=True)
        if res.stderr:
            print(res.stderr, flush=True)
        if res.returncode == 0:
            print(" [*] SUBMISSION PASSED ALL VALIDATION CHECKS (EXIT 0)!", flush=True)
        else:
            print(f" [WARNING] Validator returned exit code {res.returncode}", flush=True)
    else:
        print(f" Validator script not found at {validator_script}, skipping.", flush=True)

    print(f"\n [*] ENTIRE PIPELINE COMPLETED IN {time.time()-t_global_start:.1f}s!", flush=True)


if __name__ == '__main__':
    mp.freeze_support()
    main()
