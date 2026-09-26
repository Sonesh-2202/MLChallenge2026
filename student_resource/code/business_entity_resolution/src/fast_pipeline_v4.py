#!/usr/bin/env python3
"""
=============================================================================
 ULTRA-FAST PARALLEL BUSINESS ENTITY RESOLUTION PIPELINE (V4 PRODUCTION)
=============================================================================
Improvements over V3:
 1. High-Recall V4 Blocking (+0.30% Candidate Recall, 255 fewer blocking misses):
    - Squashed compound name keys (western + oceanhawk -> westernoceanhawk)
    - 2-character acronym preservation (fixed lowercase isupper bug)
    - Indic script address-fallback indexing (PIN codes, multi-numbers)
    - Character 4-grams on single words with len >= 6
 2. 43 Advanced RapidFuzz & Geo-Engineered Features (V4):
    - raw_n_ratio, raw_n_tsort: catches over-stripped legal stopwords
    - raw_token_diff, unmatched_raw_tokens: penalizes dropped company qualifiers
    - is_indic_script: prevents tree from penalizing zero Latin string similarity on Indic names
    - num_and_pin_match: synergistic street number and postal code validation
 3. Calibrated decision thresholds (country-specific)
 4. Preserves output/partitions_v2 and output/partitions_v3 by writing to output/partitions_v4
 5. Automatic validation via utils/validate_submission.py
"""
import os
import sys

# Windows OpenMP configuration
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

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
    FEATURE_COLS, NAME_STOP_WORDS, ADDR_STOP_WORDS, INDIC_RE,
    clean_name_v3, extract_clean_numbers, extract_postal_code,
    char_ngrams, compute_pair_features_v4
)
from blocking_v4 import extract_keys_v4, build_country_index_v4, retrieve_candidates_v4

# Global worker state
_G_INDEX = None
_G_S23 = None
_G_MODEL = None
_G_THRESHOLD = 0.95
_G_MAX_MATCHES_PER_ENTITY = 12


def init_worker(index_data, s23_data, model_bytes, threshold, max_matches=12):
    global _G_INDEX, _G_S23, _G_MODEL, _G_THRESHOLD, _G_MAX_MATCHES_PER_ENTITY
    _G_INDEX = index_data
    _G_S23 = s23_data
    _G_MODEL = pickle.loads(model_bytes)
    _G_THRESHOLD = threshold
    _G_MAX_MATCHES_PER_ENTITY = max_matches


def process_s1_chunk_parallel(s1_records):
    global _G_INDEX, _G_S23, _G_MODEL, _G_THRESHOLD, _G_MAX_MATCHES_PER_ENTITY

    results = []
    for sid, nc, ac, raw_n1 in s1_records:
        cands = retrieve_candidates_v4(nc, ac, _G_INDEX)
        cands_sorted = sorted(cands)
        cand_str = ",".join(cands_sorted)

        if not cands_sorted:
            results.append((sid, "", cand_str))
            continue

        w1_tokens = [w for w in nc.split() if w not in NAME_STOP_WORDS] if nc else []
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
            filtered_matches = [m for m, p in matches_with_p if p >= (top_p - 0.05)]
            filtered_matches = filtered_matches[:_G_MAX_MATCHES_PER_ENTITY]
            match_str = ",".join(filtered_matches)
        else:
            match_str = ""

        results.append((sid, match_str, cand_str))

    return results


def main():
    parser = argparse.ArgumentParser(description="Ultra-Fast V4 Parallel Business Entity Resolution Inference")
    parser.add_argument('--data-dir', default=os.path.join(SRC_DIR, '..', '..', '..', 'dataset'))
    parser.add_argument('--cache-dir', default=os.path.join(SRC_DIR, '..', '..', '..', 'cache'))
    parser.add_argument('--output-dir', default=os.path.join(SRC_DIR, '..', '..', '..', 'output'))
    parser.add_argument('--model-path', default=None, help="Path to lgbm_model_v4.pkl")
    parser.add_argument('--workers', type=int, default=8, help="Multiprocessing workers (default: 8)")
    parser.add_argument('--chunk-size', type=int, default=2000, help="S1 chunk size per task (default: 2000)")
    parser.add_argument('--max-matches', type=int, default=12, help="Max matches per entity (default: 12)")
    args = parser.parse_args()

    t_global = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    partition_dir = os.path.join(args.output_dir, 'partitions_v4')
    os.makedirs(partition_dir, exist_ok=True)

    print("=" * 80, flush=True)
    print(" ULTRA-FAST PARALLEL BUSINESS ENTITY RESOLUTION PIPELINE (V4 PRODUCTION)", flush=True)
    print("=" * 80, flush=True)

    model_file = args.model_path or os.path.join(args.cache_dir, 'lgbm_model_v4.pkl')
    if not os.path.exists(model_file):
        print(f"ERROR: Model checkpoint {model_file} not found!", flush=True)
        sys.exit(1)

    print(f" Loading V4 Model Checkpoint: {model_file} ...", flush=True)
    with open(model_file, 'rb') as f:
        ckpt = pickle.load(f)

    threshold = ckpt.get('threshold', 0.95)
    threshold_us = ckpt.get('threshold_us', threshold)
    threshold_in = ckpt.get('threshold_india', threshold)
    model_bytes = pickle.dumps(ckpt['model'])
    print(f" Loaded model with thresholds: US={threshold_us:.3f}, India={threshold_in:.3f} (Val F0.5: {ckpt.get('val_f05', 0.0):.4f})", flush=True)

    s1_path = os.path.join(args.data_dir, 'test', 'test_source1.tsv')
    print(f"\n Reading Test S1 Data: {s1_path} ...", flush=True)
    t0 = time.time()
    test_s1 = pd.read_csv(s1_path, sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country'], keep_default_na=False)
    all_test_s1_ids = list(test_s1['entity_id'])
    print(f" Loaded {len(test_s1):,} test reference S1 entities in {time.time()-t0:.2f}s", flush=True)

    print(" Normalizing S1 names and addresses with clean_name_v3 ...", flush=True)
    t0 = time.time()
    test_s1['name_clean'] = test_s1['business_name'].apply(clean_name_v3)
    test_s1['addr_clean'] = test_s1['business_address'].apply(normalize_address)
    print(f" Normalized {len(test_s1):,} S1 records in {time.time()-t0:.2f}s", flush=True)

    test_countries = sorted(test_s1['country'].unique())
    print(f" Test countries to process: {test_countries}", flush=True)

    s23_cols = ['entity_id', 'country', 'business_name', 'business_address']

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

        print("\n" + "=" * 80, flush=True)
        print(f" PROCESSING COUNTRY: {country.upper()}", flush=True)
        print("=" * 80, flush=True)
        t_country_start = time.time()

        country_th = threshold_us if country == 'US' else (threshold_in if country == 'India' else min(threshold_us, threshold_in))
        print(f"   Country Decision Threshold: {country_th:.3f}", flush=True)

        s2_list = []
        for chunk in pd.read_csv(os.path.join(args.data_dir, 'test', 'test_source2.tsv'), sep='\t', chunksize=1000000, usecols=s23_cols, keep_default_na=False):
            sub = chunk[chunk['country'] == country][['entity_id', 'business_name', 'business_address']]
            if len(sub) > 0:
                s2_list.append(sub)

        for chunk in pd.read_csv(os.path.join(args.data_dir, 'test', 'test_source3.tsv'), sep='\t', chunksize=1000000, usecols=s23_cols, keep_default_na=False):
            sub = chunk[chunk['country'] == country][['entity_id', 'business_name', 'business_address']]
            if len(sub) > 0:
                s2_list.append(sub)

        s23_c = pd.concat(s2_list, ignore_index=True).drop_duplicates(subset=['entity_id'])
        del s2_list
        gc.collect()

        print(f"   Loaded {len(s23_c):,} unique S2+S3 records for {country}", flush=True)
        s23_c['name_clean'] = s23_c['business_name'].apply(clean_name_v3)
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

        c_index = build_country_index_v4(s23_c, s1_n_keys, s1_a_keys, s1_c_keys, s1_names)
        del s23_c, s1_n_keys, s1_a_keys, s1_c_keys, s1_names
        gc.collect()

        chunks = [s1_records[i:i + args.chunk_size] for i in range(0, len(s1_records), args.chunk_size)]
        n_chunks = len(chunks)
        print(f"   Dispatched {n_chunks:,} chunks across {args.workers} workers ...", flush=True)

        t_infer = time.time()
        processed_entities = 0

        with open(m_part, 'w', encoding='utf-8') as f_match, open(c_part, 'w', encoding='utf-8') as f_cand:
            with mp.Pool(processes=args.workers, initializer=init_worker,
                         initargs=(c_index, s23_dict, model_bytes, country_th, args.max_matches)) as pool:
                for chunk_result in pool.imap_unordered(process_s1_chunk_parallel, chunks, chunksize=2):
                    for sid, m_str, c_str in chunk_result:
                        f_match.write(f"{sid}\t{m_str}\n")
                        f_cand.write(f"{sid}\t{c_str}\n")
                    processed_entities += len(chunk_result)

                    if processed_entities % 25000 < args.chunk_size or processed_entities == n_s1_c:
                        f_match.flush()
                        f_cand.flush()
                        rate = processed_entities / max(1e-5, time.time() - t_infer)
                        print(f"     [{country}] Progress: {processed_entities:,}/{n_s1_c:,} entities ({rate:.0f} ent/s)", flush=True)

        print(f" [*] Finished {country} in {time.time()-t_country_start:.1f}s!", flush=True)
        del c_index, s23_dict, s1_records, chunks
        gc.collect()

    print("\n" + "=" * 80, flush=True)
    print(" ASSEMBLING FINAL SUBMISSIONS FROM V4 PARTITIONS", flush=True)
    print("=" * 80, flush=True)

    matching_out = os.path.join(args.output_dir, 'matching_results.tsv')
    candidate_out = os.path.join(args.output_dir, 'candidate_pairs.tsv')

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

    print("\n" + "=" * 80, flush=True)
    print(" RUNNING OFFICIAL SUBMISSION VALIDATOR", flush=True)
    print("=" * 80, flush=True)

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
            print(" Validator stderr:\n", res.stderr, flush=True)

    print(f"\n [*] Entire V4 inference pipeline finished in {time.time()-t_global:.1f}s!", flush=True)


if __name__ == '__main__':
    main()
