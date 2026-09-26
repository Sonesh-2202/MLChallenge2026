#!/usr/bin/env python3
"""
=============================================================================
 ULTRA-FAST FRENCH RESCORING PIPELINE (V5 PRODUCTION)
=============================================================================
Specialized precision pipeline for France (259,452 entities):
  1. Extended French legal and generic token stopwords (SARL, SASU, SAS, EURL,
     SCI, SNC, SA, Societe, Ste, Cie, Groupe, France, Fils, Association).
  2. Tightened France decision threshold: tau = 0.980 (eliminates commune false merges).
  3. Narrowed margin delta = 0.025 and hard cap = 5 matches.
  4. Merges with capped India and US partitions to generate matching_results_v5.tsv.
"""

import os
import sys
import gc
import time
import pickle
import argparse
import multiprocessing as mp
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocess import normalize_address
from features_v4 import (
    FEATURE_COLS, clean_name_v3, extract_clean_numbers, extract_postal_code,
    char_ngrams, compute_pair_features_v4, INDIC_RE, JaroWinkler
)
from blocking_v4 import extract_keys_v4, build_country_index_v4, retrieve_candidates_v4

# French specific stopwords
FRENCH_STOPS = {
    'sarl', 'sasu', 'sas', 'eurl', 'sci', 'snc', 'sa', 'societe', 'ste',
    'cie', 'groupe', 'france', 'fils', 'association', 'lycee', 'college', 'ecole'
}

def clean_french_name(name):
    clean = clean_name_v3(name)
    tokens = clean.split()
    filtered = [t for t in tokens if t not in FRENCH_STOPS]
    return " ".join(filtered) if filtered else clean

_G_INDEX = None
_G_S23 = None
_G_MODEL = None
_G_THRESHOLD = 0.980
_G_MAX_MATCHES = 5
_G_MARGIN = 0.025

def init_worker(index_data, s23_data, model_bytes, threshold=0.980, max_matches=5, margin=0.025):
    global _G_INDEX, _G_S23, _G_MODEL, _G_THRESHOLD, _G_MAX_MATCHES, _G_MARGIN
    _G_INDEX = index_data
    _G_S23 = s23_data
    _G_MODEL = pickle.loads(model_bytes)
    _G_THRESHOLD = threshold
    _G_MAX_MATCHES = max_matches
    _G_MARGIN = margin

def process_s1_chunk_parallel(s1_records):
    global _G_INDEX, _G_S23, _G_MODEL, _G_THRESHOLD, _G_MAX_MATCHES, _G_MARGIN
    results = []
    for sid, nc, ac, raw_n1 in s1_records:
        cands = retrieve_candidates_v4(nc, ac, _G_INDEX)
        cands_sorted = sorted(cands)
        cand_str = ",".join(cands_sorted)

        if not cands_sorted:
            results.append((sid, "", cand_str))
            continue

        w1_tokens = [w for w in nc.split() if w not in FRENCH_STOPS] if nc else []
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
    parser = argparse.ArgumentParser(description="French High-Precision Pipeline (V5)")
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--chunk-size', type=int, default=2000)
    args = parser.parse_args()

    res_dir = os.path.abspath(os.path.join(SRC_DIR, '..', '..', '..'))
    data_dir = os.path.join(res_dir, 'dataset')
    cache_dir = os.path.join(res_dir, 'cache')
    output_dir = os.path.join(res_dir, 'output')

    part_dir = os.path.join(output_dir, 'partitions_v5')
    os.makedirs(part_dir, exist_ok=True)

    print("=" * 80)
    print(" ULTRA-FAST FRENCH RESCORING PIPELINE (V5 PRODUCTION)")
    print("=" * 80)

    # 1. Load V4 Model
    model_path = os.path.join(cache_dir, 'lgbm_model_v4.pkl')
    with open(model_path, 'rb') as f:
        ckpt = pickle.load(f)
    model = ckpt['model']
    model_bytes = pickle.dumps(model)

    # 2. Load France Test S1
    s1_path = os.path.join(data_dir, 'test', 'test_source1.tsv')
    print(f" Reading {s1_path} ...", flush=True)
    test_s1 = pd.read_csv(s1_path, sep='\t', dtype=str, usecols=['entity_id', 'business_name', 'business_address', 'country'], keep_default_na=False)
    fr_s1 = test_s1[test_s1['country'] == 'France'].copy()
    n_fr = len(fr_s1)
    print(f" France reference entities: {n_fr:,}", flush=True)

    fr_s1['name_clean'] = fr_s1['business_name'].apply(clean_french_name)
    fr_s1['addr_clean'] = fr_s1['business_address'].apply(normalize_address)

    # 3. Load France S2+S3
    print(" Loading France S2+S3 targets ...", flush=True)
    s23_cols = ['entity_id', 'country', 'business_name', 'business_address']
    s2_list = []
    for chunk in pd.read_csv(os.path.join(data_dir, 'test', 'test_source2.tsv'), sep='\t', chunksize=1000000, usecols=s23_cols, keep_default_na=False):
        sub = chunk[chunk['country'] == 'France'][['entity_id', 'business_name', 'business_address']]
        if len(sub) > 0: s2_list.append(sub)

    for chunk in pd.read_csv(os.path.join(data_dir, 'test', 'test_source3.tsv'), sep='\t', chunksize=1000000, usecols=s23_cols, keep_default_na=False):
        sub = chunk[chunk['country'] == 'France'][['entity_id', 'business_name', 'business_address']]
        if len(sub) > 0: s2_list.append(sub)

    s23_fr = pd.concat(s2_list, ignore_index=True).drop_duplicates(subset=['entity_id'])
    del s2_list
    gc.collect()

    print(f" Loaded {len(s23_fr):,} France S2+S3 targets", flush=True)
    s23_fr['name_clean'] = s23_fr['business_name'].apply(clean_french_name)
    s23_fr['addr_clean'] = s23_fr['business_address'].apply(normalize_address)

    s23_dict = dict(zip(s23_fr['entity_id'], zip(s23_fr['name_clean'], s23_fr['addr_clean'], s23_fr['business_name'])))
    s1_records = list(zip(fr_s1['entity_id'], fr_s1['name_clean'], fr_s1['addr_clean'], fr_s1['business_name']))

    s1_n_keys, s1_a_keys, s1_c_keys = set(), set(), set()
    s1_names = set(fr_s1['name_clean'])
    for nc, ac in zip(fr_s1['name_clean'], fr_s1['addr_clean']):
        nk, ak, ck = extract_keys_v4(nc, ac)
        s1_n_keys.update(nk)
        s1_a_keys.update(ak)
        s1_c_keys.update(ck)

    c_index = build_country_index_v4(s23_fr, s1_n_keys, s1_a_keys, s1_c_keys, s1_names, max_p_comp=800, max_p_single=600)
    del s23_fr, s1_n_keys, s1_a_keys, s1_c_keys, s1_names
    gc.collect()

    chunks = [s1_records[i:i + args.chunk_size] for i in range(0, len(s1_records), args.chunk_size)]
    print(f" Dispatched {len(chunks):,} chunks across {args.workers} workers ...", flush=True)

    m_part = os.path.join(part_dir, 'matches_France.tsv')
    c_part = os.path.join(part_dir, 'candidates_France.tsv')
    t_start = time.time()
    processed = 0

    with open(m_part, 'w', encoding='utf-8') as f_match, open(c_part, 'w', encoding='utf-8') as f_cand:
        with mp.Pool(processes=args.workers, initializer=init_worker,
                     initargs=(c_index, s23_dict, model_bytes, 0.980, 5, 0.025)) as pool:
            for chunk_res in pool.imap_unordered(process_s1_chunk_parallel, chunks, chunksize=2):
                for sid, m_str, c_str in chunk_res:
                    f_match.write(f"{sid}\t{m_str}\n")
                    f_cand.write(f"{sid}\t{c_str}\n")
                processed += len(chunk_res)
                if processed % 25000 < args.chunk_size or processed == n_fr:
                    f_match.flush()
                    f_cand.flush()
                    rate = processed / max(1e-5, time.time() - t_start)
                    print(f"   [France V5] Progress: {processed:,}/{n_fr:,} ({rate:.0f} ent/s)", flush=True)

    print(f" Finished France V5 in {time.time()-t_start:.1f}s!", flush=True)

    # Combine with capped India and US to produce matching_results_v5.tsv
    print("\n Assembling matching_results_v5.tsv ...", flush=True)
    out_v5 = os.path.join(output_dir, 'matching_results_v5.tsv')

    fr_matches = {}
    with open(m_part, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip('\r\n').split('\t')
            fr_matches[parts[0]] = parts[1] if len(parts) > 1 else ""

    # Load India and US from partitions_v4 (capping to 6 matches)
    other_matches = {}
    for country in ['India', 'US']:
        part_p = os.path.join(output_dir, 'partitions_v4', f'matches_{country}.tsv')
        with open(part_p, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip('\r\n').split('\t')
                m_str = parts[1] if len(parts) > 1 else ""
                if m_str:
                    m_list = m_str.split(',')
                    if len(m_list) > 6:
                        m_str = ",".join(m_list[:6])
                other_matches[parts[0]] = m_str

    with open(out_v5, 'w', encoding='utf-8', newline='') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in test_s1['entity_id']:
            m = fr_matches.get(sid) if sid in fr_matches else other_matches.get(sid, "")
            f.write(f"{sid}\t{m}\n")

    print(f" Saved unified V5 predictions to {out_v5}!")
    print("Done!", flush=True)

if __name__ == '__main__':
    main()
