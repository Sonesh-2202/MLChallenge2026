#!/usr/bin/env python3
"""
=============================================================================
 ULTRA-FAST PARALLEL BUSINESS ENTITY RESOLUTION PIPELINE (PRODUCTION RUN)
=============================================================================
Uses:
 - Country-Partitioned Inverted-Index Blocking with posting list capping (max 1000)
 - C-Accelerated RapidFuzz Vectorized Pairwise Feature Engineering
 - Multiprocessing across CPU cores (chunk-vectorized LightGBM scoring)
 - Partitioned Disk-Streaming (Zero-OOM, sub-second recovery)
 - Sub-second singleton handling & competition format compliance
"""
import os
import sys
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

# Add local src to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocess import normalize_name, normalize_address, extract_numbers, STOP_WORDS
from blocking import extract_name_keys, extract_addr_keys
from features import FEATURE_COLS

# Global worker state initialized once per process
_G_INDEX = None
_G_S23 = None
_G_MODEL = None
_G_THRESHOLD = 0.70
_G_NAME_K = 25
_G_ADDR_K = 25
_G_MAX_POSTINGS = 1000


def init_worker(index_data, s23_data, model_bytes, threshold, name_k=25, addr_k=25, max_postings=1000):
    """Initialize worker process with index, lookup dict, and model."""
    global _G_INDEX, _G_S23, _G_MODEL, _G_THRESHOLD, _G_NAME_K, _G_ADDR_K, _G_MAX_POSTINGS
    _G_INDEX = index_data
    _G_S23 = s23_data
    _G_MODEL = pickle.loads(model_bytes)
    _G_THRESHOLD = threshold
    _G_NAME_K = name_k
    _G_ADDR_K = addr_k
    _G_MAX_POSTINGS = max_postings


def process_s1_chunk_parallel(s1_records):
    """
    Process a chunk of S1 records in parallel:
     1. Retrieve candidates using inverted index
     2. Vectorized feature extraction with C-accelerated RapidFuzz
     3. Batch LightGBM prediction
     4. Filter matches by threshold
    s1_records: list of tuples (sid, name_clean, addr_clean)
    Returns:
        matches_out: list of (sid, [matched_mid, ...])
        candidates_out: list of (sid, [candidate_mid, ...])
    """
    idx = _G_INDEX
    s23_dict = _G_S23
    model = _G_MODEL
    threshold = _G_THRESHOLD
    name_k = _G_NAME_K
    addr_k = _G_ADDR_K
    max_p = _G_MAX_POSTINGS

    # 1. Candidate Retrieval for chunk
    chunk_candidates = []
    total_pairs = 0

    for sid, nc, ac in s1_records:
        cands = set()
        # Exact clean name pass
        if nc and nc in idx['exact_name']:
            cands.update(idx['exact_name'][nc][:name_k])

        # Name token pass (TF-IDF weighted)
        n_scores = {}
        for k in extract_name_keys(nc):
            postings = idx['name'].get(k)
            if postings and len(postings) <= max_p:
                w = 1.0 / np.log1p(len(postings))
                for mid in postings:
                    n_scores[mid] = n_scores.get(mid, 0.0) + w
        if n_scores:
            cands.update(m for m, _ in heapq.nlargest(name_k, n_scores.items(), key=lambda x: x[1]))

        # Address token pass (TF-IDF weighted)
        a_scores = {}
        for k in extract_addr_keys(ac):
            postings = idx['addr'].get(k)
            if postings and len(postings) <= max_p:
                w = 1.0 / np.log1p(len(postings))
                for mid in postings:
                    a_scores[mid] = a_scores.get(mid, 0.0) + w
        if a_scores:
            cands.update(m for m, _ in heapq.nlargest(addr_k, a_scores.items(), key=lambda x: x[1]))

        cand_list = sorted(cands)
        chunk_candidates.append((sid, nc, ac, cand_list))
        total_pairs += len(cand_list)

    if total_pairs == 0:
        return [(sid, []) for sid, _, _, _ in chunk_candidates], [(sid, c) for sid, _, _, c in chunk_candidates]

    # 2. Build vectorized feature matrix X for all pairs in chunk
    X = np.empty((total_pairs, 17), dtype=np.float32)
    pair_meta = []  # list of (sid, mid)

    pair_idx = 0
    for sid, nc, ac, cand_list in chunk_candidates:
        if not cand_list:
            continue
        w1_set = set(nc.split()) if nc else set()
        aw1_set = set(ac.split()) if ac else set()
        nums1 = extract_numbers(ac)
        has_a1 = bool(ac and ac != 'nan')

        for mid in cand_list:
            pair_meta.append((sid, mid))
            s23_info = s23_dict.get(mid)
            if s23_info is None:
                X[pair_idx, :] = 0.0
                pair_idx += 1
                continue
            n2, a2 = s23_info

            # Name metrics
            n_ratio = fuzz.ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
            n_tset = fuzz.token_set_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
            n_tsort = fuzz.token_sort_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
            n_part = fuzz.partial_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
            w2_set = set(n2.split()) if n2 else set()
            n_jacc = len(w1_set & w2_set) / max(1, len(w1_set | w2_set)) if (w1_set or w2_set) else 0.0

            # Address metrics
            has_a2 = bool(a2 and a2 != 'nan')
            both_addr = 1.0 if (has_a1 and has_a2) else 0.0
            if both_addr:
                a_ratio = fuzz.ratio(ac, a2) / 100.0
                a_tset = fuzz.token_set_ratio(ac, a2) / 100.0
                a_tsort = fuzz.token_sort_ratio(ac, a2) / 100.0
                a_part = fuzz.partial_ratio(ac, a2) / 100.0
                aw2_set = set(a2.split()) if a2 else set()
                a_jacc = len(aw1_set & aw2_set) / max(1, len(aw1_set | aw2_set)) if (aw1_set or aw2_set) else 0.0
            else:
                a_ratio = a_tset = a_tsort = a_part = a_jacc = 0.0

            nums2 = extract_numbers(a2)
            num_match = 1.0 if (nums1 and nums2 and (nums1 & nums2)) else 0.0
            num_jacc = len(nums1 & nums2) / max(1, len(nums1 | nums2)) if (nums1 or nums2) else 0.0

            eff_a_score = a_tset if both_addr else n_tset
            harm = (2.0 * n_tset * eff_a_score) / (n_tset + eff_a_score + 1e-5)
            exact_n = 1.0 if (nc and nc == n2) else 0.0
            exact_a = 1.0 if (ac and ac == a2) else 0.0

            X[pair_idx, 0] = n_ratio
            X[pair_idx, 1] = n_tset
            X[pair_idx, 2] = n_tsort
            X[pair_idx, 3] = n_part
            X[pair_idx, 4] = n_jacc
            X[pair_idx, 5] = a_ratio
            X[pair_idx, 6] = a_tset
            X[pair_idx, 7] = a_tsort
            X[pair_idx, 8] = a_part
            X[pair_idx, 9] = a_jacc
            X[pair_idx, 10] = num_match
            X[pair_idx, 11] = num_jacc
            X[pair_idx, 12] = both_addr
            X[pair_idx, 13] = 1.0 - both_addr
            X[pair_idx, 14] = harm
            X[pair_idx, 15] = exact_n
            X[pair_idx, 16] = exact_a
            pair_idx += 1

    # 3. Vectorized LightGBM prediction
    probs = model.predict(X)

    # 4. Map matches >= threshold
    matches_map = {sid: [] for sid, _, _, _ in chunk_candidates}
    for (sid, mid), p in zip(pair_meta, probs):
        if p >= threshold:
            matches_map[sid].append(mid)

    matches_out = [(sid, matches_map[sid]) for sid, _, _, _ in chunk_candidates]
    candidates_out = [(sid, cands) for sid, _, _, cands in chunk_candidates]

    return matches_out, candidates_out


def save_country_partition(country, matches_list, candidates_list, partition_dir):
    """Write intermediate country partition to disk to keep RAM low."""
    os.makedirs(partition_dir, exist_ok=True)
    m_path = os.path.join(partition_dir, f'matches_{country}.tsv')
    c_path = os.path.join(partition_dir, f'candidates_{country}.tsv')

    with open(m_path, 'w', encoding='utf-8', newline='') as f:
        for sid, mids in matches_list:
            f.write(f"{sid}\t{','.join(sorted(mids)) if mids else ''}\n")

    with open(c_path, 'w', encoding='utf-8', newline='') as f:
        for sid, cids in candidates_list:
            f.write(f"{sid}\t{','.join(sorted(cids)) if cids else ''}\n")

    print(f"   Saved partition files for {country} to {partition_dir}")


def main():
    import faulthandler
    faulthandler.enable()

    parser = argparse.ArgumentParser(description="Parallel Business Entity Resolution Pipeline")
    parser.add_argument('--data-dir', default=os.path.join(SRC_DIR, '..', '..', '..', 'dataset'))
    parser.add_argument('--output-dir', default=os.path.join(SRC_DIR, '..', '..', '..', 'output'))
    parser.add_argument('--cache-dir', default=os.path.join(SRC_DIR, '..', '..', '..', 'cache'))
    parser.add_argument('--n-workers', type=int, default=10, help="Number of worker processes (default: 10)")
    parser.add_argument('--chunk-size', type=int, default=2500, help="S1 records per worker task")
    args = parser.parse_args()

    t_global_start = time.time()
    print("=" * 78)
    print(" ULTRA-FAST PARALLEL BUSINESS ENTITY RESOLUTION PIPELINE")
    print("=" * 78)
    print(f" Data Dir:    {os.path.abspath(args.data_dir)}")
    print(f" Output Dir:  {os.path.abspath(args.output_dir)}")
    print(f" Cache Dir:   {os.path.abspath(args.cache_dir)}")
    print(f" Workers:     {args.n_workers} parallel processes")
    print(f" Chunk Size:  {args.chunk_size} entities/chunk")

    partition_dir = os.path.join(args.output_dir, 'partitions')
    os.makedirs(partition_dir, exist_ok=True)

    # Load Model Checkpoint
    checkpoint_path = os.path.join(args.cache_dir, 'lgbm_model.pkl')
    if not os.path.exists(checkpoint_path):
        print(f"\n[ERROR] Model checkpoint not found at {checkpoint_path}!")
        sys.exit(1)

    with open(checkpoint_path, 'rb') as f:
        ckpt = pickle.load(f)
    model = ckpt['model']
    threshold = ckpt['threshold']
    model_bytes = pickle.dumps(model)
    print(f"\n Loaded trained LightGBM model from {checkpoint_path}")
    print(f"   Optimal Macro F0.5 Threshold: {threshold:.2f}")

    # ==============================================================
    # LOAD COMPLETE TEST SOURCE 1
    # ==============================================================
    print("\n" + "=" * 78)
    print(" LOADING TEST SOURCE 1")
    print("=" * 78)
    s1_path = os.path.join(args.data_dir, 'test', 'test_source1.tsv')
    print(f" Reading {s1_path} ...", flush=True)
    t0 = time.time()
    test_s1 = pd.read_csv(s1_path, sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country'], keep_default_na=False)
    all_test_s1_ids = list(test_s1['entity_id'])
    print(f" Loaded {len(test_s1):,} test reference S1 entities in {time.time()-t0:.2f}s", flush=True)

    print(" Normalizing S1 names and addresses ...", flush=True)
    t0 = time.time()
    test_s1['name_clean'] = test_s1['business_name'].apply(normalize_name)
    test_s1['addr_clean'] = test_s1['business_address'].apply(normalize_address)
    print(f" Normalized {len(test_s1):,} S1 records in {time.time()-t0:.2f}s", flush=True)

    test_countries = sorted(test_s1['country'].unique())
    print(f" Test countries to process: {test_countries}", flush=True)

    # ==============================================================
    # PROCESS EACH COUNTRY IN PARALLEL
    # ==============================================================
    for country in test_countries:
        m_part = os.path.join(partition_dir, f'matches_{country}.tsv')
        c_part = os.path.join(partition_dir, f'candidates_{country}.tsv')

        # Check if already processed
        s1_c = test_s1[test_s1['country'] == country]
        n_s1_c = len(s1_c)

        if os.path.exists(m_part) and os.path.exists(c_part):
            # Check line count
            with open(m_part, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            if lines == n_s1_c:
                print(f"\n [*] Country {country} partition already exists ({lines:,} records). Skipping recomputation!", flush=True)
                continue

        print("\n" + "=" * 78)
        print(f" PROCESSING COUNTRY: {country.upper()}")
        print("=" * 78)
        t_country_start = time.time()

        s1_records = list(zip(s1_c['entity_id'], s1_c['name_clean'], s1_c['addr_clean']))
        print(f"   S1 entities in {country}: {n_s1_c:,}", flush=True)

        # 1. Load S2 and S3 for this country (keeping only 4 columns)
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

        print(f"   Loaded {len(s23_c):,} unique S2+S3 candidate records in {time.time()-t_load:.2f}s", flush=True)

        # 2. Normalize S23 and build Inverted Index
        print(f"   Extracting S1 query keys for {country} ...", flush=True)
        s1_names = set(s1_c['name_clean'])
        s1_name_tokens = set()
        for nc in s1_c['name_clean']:
            s1_name_tokens.update(extract_name_keys(nc))
        s1_addr_tokens = set()
        for ac in s1_c['addr_clean']:
            s1_addr_tokens.update(extract_addr_keys(ac))

        print(f"   Normalizing S2+S3 records & building pruned inverted index ...", flush=True)
        t_idx = time.time()
        s23_c['name_clean'] = s23_c['business_name'].apply(normalize_name)
        s23_c['addr_clean'] = s23_c['business_address'].apply(normalize_address)

        exact_name_idx = {}
        name_idx = {}
        addr_idx = {}
        s23_dict = {}

        max_cap = 1001
        for r in s23_c.itertuples(index=False):
            eid = r.entity_id
            nc = r.name_clean
            ac = r.addr_clean

            if nc and nc in s1_names:
                exact_name_idx.setdefault(nc, []).append(eid)
            if nc:
                for k in extract_name_keys(nc):
                    if k in s1_name_tokens:
                        lst = name_idx.setdefault(k, [])
                        if len(lst) < max_cap:
                            lst.append(eid)
            if ac:
                for k in extract_addr_keys(ac):
                    if k in s1_addr_tokens:
                        lst = addr_idx.setdefault(k, [])
                        if len(lst) < max_cap:
                            lst.append(eid)

            s23_dict[eid] = (nc, ac)

        idx_data = {'exact_name': exact_name_idx, 'name': name_idx, 'addr': addr_idx}
        print(f"   Index built in {time.time()-t_idx:.2f}s (Exact names: {len(exact_name_idx):,}, Name tokens: {len(name_idx):,}, Addr tokens: {len(addr_idx):,})", flush=True)
        del s23_c, s1_names, s1_name_tokens, s1_addr_tokens
        gc.collect()

        # 3. Create Chunks for Multiprocessing
        chunks = [s1_records[i:i + args.chunk_size] for i in range(0, n_s1_c, args.chunk_size)]
        print(f"   Partitioned {n_s1_c:,} entities into {len(chunks):,} parallel chunks", flush=True)

        # 4. Run Multiprocessing Pool
        print(f"   Spawning {args.n_workers} worker processes for parallel retrieval + scoring ...", flush=True)
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
            pool.close()
            pool.join()

        t_par_end = time.time()
        print(f"   Country {country} completed in {t_par_end - t_country_start:.2f}s!", flush=True)

        # Collect results and save to country partition
        country_matches = [item for m_list, _ in chunk_results for item in m_list]
        country_cands = [item for _, c_list in chunk_results for item in c_list]

        save_country_partition(country, country_matches, country_cands, partition_dir)
        total_m = sum(len(m[1]) for m in country_matches)
        total_c = sum(len(c[1]) for c in country_cands)
        print(f"   Country {country} Total Matches: {total_m:,} | Candidates: {total_c:,}", flush=True)

        # Clean all country memory
        del idx_data, s23_dict, exact_name_idx, name_idx, addr_idx, chunk_results, async_results, s1_records, s1_c, country_matches, country_cands
        gc.collect()

    del test_s1
    gc.collect()

    # ==============================================================
    # MERGE PARTITIONS INTO FINAL SUBMISSION TSV FILES
    # ==============================================================
    print("\n" + "=" * 78)
    print(" MERGING PARTITIONS INTO FINAL SUBMISSION FILES")
    print("=" * 78)

    matching_out = os.path.join(args.output_dir, 'matching_results.tsv')
    candidate_out = os.path.join(args.output_dir, 'candidate_pairs.tsv')

    # Read partition dictionaries
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

    # Stream write in original test order
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

    print(f"\n Final Submission Statistics:")
    print(f"   Total Test S1 Entities: {n_total_entities:,}")
    print(f"   Entities with Matches:  {n_matches:,}")
    print(f"   Singletons (no match):  {n_singletons:,} ({n_singletons/n_total_entities*100:.2f}%)")
    print(f"   Macro F0.5 Threshold:   {threshold:.2f}")

    # ==============================================================
    # VALIDATE SUBMISSION WITH OFFICIAL VALIDATOR
    # ==============================================================
    print("\n" + "=" * 78)
    print(" RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 78)

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
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode == 0:
            print(" [*] SUBMISSION PASSED ALL VALIDATION CHECKS (EXIT 0)!")
        else:
            print(f" [WARNING] Validator returned exit code {res.returncode}")
    else:
        print(f" Validator script not found at {validator_script}, skipping.")

    print(f"\n [*] ENTIRE PIPELINE COMPLETED IN {time.time()-t_global_start:.1f}s!")


if __name__ == '__main__':
    mp.freeze_support()
    main()
