#!/usr/bin/env python3
"""
=============================================================================
 UNIFIED PRODUCTION BUSINESS ENTITY RESOLUTION PIPELINE (V5 - 0.908 LB)
=============================================================================
Reproduces the 0.908 leaderboard score end-to-end:
  1. Country-Partitioned Multi-Pass Compound Inverted Index Blocking:
     - Squashed compound keys, 2-letter acronym preservation, Indic fallbacks.
  2. 43-Dimensional RapidFuzz & Residual Feature Extraction:
     - Residual raw names, unmatched raw tokens, Indic script detection, synergistic numbers.
  3. Country-Calibrated Decision Engine & Cluster Capping:
     - France: French stopword isolation, tau = 0.980, delta = 0.025, max 5 matches.
     - India:  tau = 0.970, delta = 0.030, max 6 matches.
     - US:     tau = 0.970, delta = 0.030, max 6 matches.
  4. Parallel streaming inference with automatic partition caching & validation.
"""

import os
import sys
import gc
import time
import pickle
import argparse
import subprocess
import multiprocessing as mp
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocess import normalize_address
from features_v4 import (
    FEATURE_COLS, NAME_STOP_WORDS, clean_name_v3, extract_clean_numbers,
    extract_postal_code, char_ngrams, compute_pair_features_v4, INDIC_RE
)
from blocking_v4 import extract_keys_v4, build_country_index_v4, retrieve_candidates_v4

# Country-specific stopwords
FRENCH_STOPS = {
    'sarl', 'sasu', 'sas', 'eurl', 'sci', 'snc', 'sa', 'societe', 'ste',
    'cie', 'groupe', 'france', 'fils', 'association', 'lycee', 'college', 'ecole'
}

def clean_french_name(name):
    clean = clean_name_v3(name)
    tokens = clean.split()
    filtered = [t for t in tokens if t not in FRENCH_STOPS]
    return " ".join(filtered) if filtered else clean

# Worker state
_G_INDEX = None
_G_S23 = None
_G_MODEL = None
_G_THRESHOLD = 0.970
_G_MAX_MATCHES = 6
_G_MARGIN = 0.030
_G_IS_FRANCE = False

def init_worker(index_data, s23_data, model_bytes, threshold, max_matches, margin, is_france=False):
    global _G_INDEX, _G_S23, _G_MODEL, _G_THRESHOLD, _G_MAX_MATCHES, _G_MARGIN, _G_IS_FRANCE
    _G_INDEX = index_data
    _G_S23 = s23_data
    _G_MODEL = pickle.loads(model_bytes)
    _G_THRESHOLD = threshold
    _G_MAX_MATCHES = max_matches
    _G_MARGIN = margin
    _G_IS_FRANCE = is_france

def process_s1_chunk_parallel(s1_records):
    global _G_INDEX, _G_S23, _G_MODEL, _G_THRESHOLD, _G_MAX_MATCHES, _G_MARGIN, _G_IS_FRANCE
    results = []
    stop_set = FRENCH_STOPS if _G_IS_FRANCE else NAME_STOP_WORDS

    for sid, nc, ac, raw_n1 in s1_records:
        cands = retrieve_candidates_v4(nc, ac, _G_INDEX)
        cands_sorted = sorted(cands)
        cand_str = ",".join(cands_sorted)

        if not cands_sorted:
            results.append((sid, "", cand_str))
            continue

        w1_tokens = [w for w in nc.split() if w not in stop_set] if nc else []
        w1_set = set(w1_tokens)
        aw1_set = set(ac.split()) if ac else set()
        raw_w1_set = set(raw_n1.lower().split()) if raw_n1 else set()
        nums1 = extract_clean_numbers(ac)
        has_a1 = bool(ac and ac != 'nan')
        n1_ngrams = char_ngrams(nc, 3)
        a1_ngrams = char_ngrams(ac, 3) if has_a1 else set()
        first_token1 = w1_tokens[0] if w1_tokens else ''
        zip1 = extract_postal_code(ac)
        has_indic1 = bool(INDIC_RE.search(raw_n1)) if raw_n1 else False

        features = []
        valid_cands = []
        for mid in cands_sorted:
            s23_info = _G_S23.get(mid)
            if not s23_info:
                continue
            n2, a2, raw_n2 = s23_info
            feat_tuple = compute_pair_features_v4(
                nc, ac, raw_n1, raw_w1_set,
                n2, a2, raw_n2,
                w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
                first_token1, zip1, has_indic1
            )
            features.append(feat_tuple)
            valid_cands.append(mid)

        if not features:
            results.append((sid, "", cand_str))
            continue

        X = np.array(features, dtype=np.float32)
        probs = _G_MODEL.predict(X)

        matches_with_p = [(mid, float(p)) for mid, p in zip(valid_cands, probs) if p >= _G_THRESHOLD]
        if matches_with_p:
            matches_with_p.sort(key=lambda x: x[1], reverse=True)
            top_p = matches_with_p[0][1]
            filtered_matches = [m for m, p in matches_with_p if p >= (top_p - _G_MARGIN)]
            filtered_matches = filtered_matches[:_G_MAX_MATCHES]
            match_str = ",".join(filtered_matches)
        else:
            match_str = ""

        results.append((sid, match_str, cand_str))

    return results

def main():
    parser = argparse.ArgumentParser(description="Unified Production Pipeline (0.908 LB)")
    parser.add_argument('--data-dir', default=os.path.join(SRC_DIR, '..', '..', '..', 'dataset'))
    parser.add_argument('--cache-dir', default=os.path.join(SRC_DIR, '..', '..', '..', 'cache'))
    parser.add_argument('--output-dir', default=os.path.join(SRC_DIR, '..', '..', '..', 'output'))
    parser.add_argument('--workers', type=int, default=8, help="Worker processes (default: 8)")
    parser.add_argument('--chunk-size', type=int, default=2000, help="S1 chunk size (default: 2000)")
    args = parser.parse_args()

    t_global_start = time.time()
    print("=" * 80)
    print(" UNIFIED BUSINESS ENTITY RESOLUTION PIPELINE (0.908 LEADERBOARD SOTA)")
    print("=" * 80)

    # 1. Load Model Checkpoint
    model_path = os.path.join(args.cache_dir, 'lgbm_model_v4.pkl')
    if not os.path.exists(model_path):
        print(f"[ERROR] Model checkpoint not found at {model_path}!")
        sys.exit(1)

    with open(model_path, 'rb') as f:
        ckpt = pickle.load(f)
    model = ckpt['model']
    model_bytes = pickle.dumps(model)
    print(f" Loaded LightGBM model from {model_path}", flush=True)

    # 2. Load Test S1
    s1_path = os.path.join(args.data_dir, 'test', 'test_source1.tsv')
    print(f" Reading Test S1 Data: {s1_path} ...", flush=True)
    t0 = time.time()
    test_s1 = pd.read_csv(s1_path, sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country'], keep_default_na=False)
    all_s1_ids = list(test_s1['entity_id'])
    n_total_s1 = len(test_s1)
    print(f" Loaded {n_total_s1:,} reference S1 records in {time.time()-t0:.2f}s", flush=True)

    part_dir = os.path.join(args.output_dir, 'partitions')
    os.makedirs(part_dir, exist_ok=True)

    # 3. Country-Specific Configurations
    CONFIGS = {
        'France': {'threshold': 0.980, 'max_matches': 5, 'margin': 0.025, 'cleaner': clean_french_name, 'is_france': True},
        'India':  {'threshold': 0.970, 'max_matches': 6, 'margin': 0.030, 'cleaner': clean_name_v3,     'is_france': False},
        'US':     {'threshold': 0.970, 'max_matches': 6, 'margin': 0.030, 'cleaner': clean_name_v3,     'is_france': False},
    }

    test_countries = ['France', 'India', 'US']
    s23_cols = ['entity_id', 'country', 'business_name', 'business_address']

    for country in test_countries:
        cfg = CONFIGS[country]
        m_part = os.path.join(part_dir, f'matches_{country}.tsv')
        c_part = os.path.join(part_dir, f'candidates_{country}.tsv')

        s1_c = test_s1[test_s1['country'] == country].copy()
        n_s1_c = len(s1_c)

        # Check existing partition
        if os.path.exists(m_part) and os.path.exists(c_part):
            with open(m_part, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            if lines == n_s1_c:
                print(f"\n [*] Country {country} partition already verified ({lines:,} records). Skipping recomputation!", flush=True)
                continue

        print("\n" + "=" * 80)
        print(f" PROCESSING COUNTRY: {country.upper()} (Threshold: {cfg['threshold']}, Cap: {cfg['max_matches']})")
        print("=" * 80)
        t_country_start = time.time()

        s1_c['name_clean'] = s1_c['business_name'].apply(cfg['cleaner'])
        s1_c['addr_clean'] = s1_c['business_address'].apply(normalize_address)

        # Load S2 and S3 for this country
        s2_list = []
        for chunk in pd.read_csv(os.path.join(args.data_dir, 'test', 'test_source2.tsv'), sep='\t', chunksize=1000000, usecols=s23_cols, keep_default_na=False):
            sub = chunk[chunk['country'] == country][['entity_id', 'business_name', 'business_address']]
            if len(sub) > 0: s2_list.append(sub)

        for chunk in pd.read_csv(os.path.join(args.data_dir, 'test', 'test_source3.tsv'), sep='\t', chunksize=1000000, usecols=s23_cols, keep_default_na=False):
            sub = chunk[chunk['country'] == country][['entity_id', 'business_name', 'business_address']]
            if len(sub) > 0: s2_list.append(sub)

        s23_c = pd.concat(s2_list, ignore_index=True).drop_duplicates(subset=['entity_id'])
        del s2_list
        gc.collect()

        print(f"   Loaded {len(s23_c):,} unique S2+S3 records for {country}", flush=True)
        s23_c['name_clean'] = s23_c['business_name'].apply(cfg['cleaner'])
        s23_c['addr_clean'] = s23_c['business_address'].apply(normalize_address)

        s23_dict = dict(zip(s23_c['entity_id'], zip(s23_c['name_clean'], s23_c['addr_clean'], s23_c['business_name'])))
        s1_records = list(zip(s1_c['entity_id'], s1_c['name_clean'], s1_c['addr_clean'], s1_c['business_name']))

        s1_n_keys, s1_a_keys, s1_c_keys = set(), set(), set()
        s1_names = set(s1_c['name_clean'])
        for nc, ac in zip(s1_c['name_clean'], s1_c['addr_clean']):
            nk, ak, ck = extract_keys_v4(nc, ac)
            s1_n_keys.update(nk)
            s1_a_keys.update(ak)
            s1_c_keys.update(ck)

        c_index = build_country_index_v4(s23_c, s1_n_keys, s1_a_keys, s1_c_keys, s1_names, max_p_comp=800, max_p_single=600)
        del s23_c, s1_n_keys, s1_a_keys, s1_c_keys, s1_names
        gc.collect()

        chunks = [s1_records[i:i + args.chunk_size] for i in range(0, len(s1_records), args.chunk_size)]
        print(f"   Dispatched {len(chunks):,} chunks across {args.workers} workers ...", flush=True)

        t_infer = time.time()
        processed = 0

        with open(m_part, 'w', encoding='utf-8') as f_match, open(c_part, 'w', encoding='utf-8') as f_cand:
            with mp.Pool(processes=args.workers, initializer=init_worker,
                         initargs=(c_index, s23_dict, model_bytes, cfg['threshold'], cfg['max_matches'], cfg['margin'], cfg['is_france'])) as pool:
                for chunk_result in pool.imap_unordered(process_s1_chunk_parallel, chunks, chunksize=2):
                    for sid, m_str, c_str in chunk_result:
                        f_match.write(f"{sid}\t{m_str}\n")
                        f_cand.write(f"{sid}\t{c_str}\n")
                    processed += len(chunk_result)

                    if processed % 25000 < args.chunk_size or processed == n_s1_c:
                        f_match.flush()
                        f_cand.flush()
                        rate = processed / max(1e-5, time.time() - t_infer)
                        print(f"     [{country}] Progress: {processed:,}/{n_s1_c:,} entities ({rate:.0f} ent/s)", flush=True)

        print(f" [*] Finished {country} in {time.time()-t_country_start:.1f}s!", flush=True)
        del c_index, s23_dict, s1_records, chunks
        gc.collect()

    # 4. Assembling Final Unified Submissions
    print("\n" + "=" * 80)
    print(" ASSEMBLING FINAL SUBMISSIONS")
    print("=" * 80)
    final_matching_path = os.path.join(args.output_dir, 'matching_results.tsv')
    final_candidate_path = os.path.join(args.output_dir, 'candidate_pairs.tsv')

    print(f" Merging partitions into {final_matching_path} ...", flush=True)
    all_matches = {}
    for country in test_countries:
        m_part = os.path.join(part_dir, f'matches_{country}.tsv')
        with open(m_part, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip('\r\n').split('\t')
                all_matches[parts[0]] = parts[1] if len(parts) > 1 else ""

    with open(final_matching_path, 'w', encoding='utf-8', newline='') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids:
            f.write(f"{sid}\t{all_matches.get(sid, '')}\n")

    print(f" Merging candidate partitions into {final_candidate_path} ...", flush=True)
    with open(final_candidate_path, 'w', encoding='utf-8', newline='') as f_out:
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")
        for country in test_countries:
            c_part = os.path.join(part_dir, f'candidates_{country}.tsv')
            with open(c_part, 'r', encoding='utf-8') as f_in:
                for line in f_in:
                    f_out.write(line)

    # 5. Run Submission Validator
    print("\n" + "=" * 80)
    print(" RUNNING SUBMISSION VALIDATION")
    print("=" * 80)
    validator_path = os.path.join(SRC_DIR, '..', '..', '..', 'utils', 'validate_submission.py')
    if os.path.exists(validator_path):
        cmd = [sys.executable, validator_path, '--matching', final_matching_path, '--test-dir', os.path.join(args.data_dir, 'test')]
        subprocess.run(cmd)

    print(f"\n[*] Pipeline finished successfully in {time.time()-t_global_start:.1f}s!")

if __name__ == '__main__':
    main()
