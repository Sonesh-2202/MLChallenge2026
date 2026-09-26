"""
Controlled Experiment: High-Recall Blocking Optimization
Evaluates blocking candidate recall and candidate volume on the exact 10,000 S1 validation split.
Compares:
  - Baseline v2 blocking: Recall = 93.74%, ~102 pairs/entity
  - Candidate v3 blocking with:
      1. Separated name/address stop words (preserving French region names in business names)
      2. 2-letter uppercase/acronym support in conjunctions
      3. 3-letter address tokens (oak, bay, elm, box)
      4. Subword/domain squashing handler
      5. Prefix 3-grams and increased posting limits (max_p = 5000)
"""
import os
import sys
import gc
import time
import heapq
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocess import normalize_address, strip_accents, CLEAN_NONWORD_RE, CLEAN_SPACE_RE
from features_v2 import clean_name_enhanced, extract_clean_numbers
from train_v2 import extract_keys as extract_keys_v2, build_country_index as build_index_v2, retrieve_candidates as retrieve_cands_v2

DATA_DIR = os.path.join(SRC_DIR, '..', '..', '..', 'dataset')

# Pure name stopwords - NO regional names or meaningful business words
NAME_STOP_WORDS = {
    'the', 'and', 'for', 'of', 'in', 'on', 'at', 'to', 'a', 'an', 'is',
    'llc', 'inc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private', 'co', 'company',
    'llp', 'plc', 'sarl', 'gmbh', 'ag', 'dba', 'sas', 'sa', 'eurl', 'srl', 'sl', 'pty',
    'pllc', 'pc', 'group', 'services', 'industries', 'associates', 'federation',
    'null', 'nan', 'com', 'org', 'net'
}

ADDR_STOP_WORDS = {
    'street', 'road', 'st', 'rd', 'ave', 'avenue', 'lane', 'ln', 'dr', 'drive',
    'court', 'ct', 'city', 'state', 'unit', 'apt', 'apartment', 'floor', 'fl',
    'near', 'opp', 'behind', 'beside', 'null', 'nan',
    'france', 'india', 'united', 'states', 'usa', 'us',
    'rue', 'boulevard', 'blvd', 'chemin', 'impasse', 'route', 'place', 'allee',
    'cours', 'passage', 'quai', 'zone', 'zi', 'za', 'cedex', 'batiment', 'residence',
    'nagar', 'colony', 'complex', 'building', 'plot', 'sector', 'phase', 'bazar',
    'mandi', 'chowk', 'market', 'dist', 'taluk', 'village', 'post', 'po',
    'suite', 'ste', 'pobox', 'box', 'highway', 'hwy', 'parkway', 'pkwy', 'circle', 'cir',
    'plaza', 'center', 'ctr'
}

def extract_keys_v3(nc, ac):
    """
    Enhanced V3 candidate key generation:
    1. Extracts pure name tokens (len >= 2 for uppercase/acronyms, len >= 3 for standard words)
    2. Character 3-gram prefixes for name tokens
    3. Handles squashed domain names by generating sub-tokens
    4. 3-letter address tokens
    5. Robust conjunction keys
    """
    n_keys = set()
    a_keys = set()
    c_keys = set()

    # Name tokens
    raw_n_words = nc.split()
    n_tokens = []
    for w in raw_n_words:
        if w not in NAME_STOP_WORDS:
            if len(w) >= 3:
                n_tokens.append(w)
            elif len(w) == 2 and w.isupper():
                n_tokens.append(w)

    nums = extract_clean_numbers(ac)
    
    # Address tokens: allow 3-char tokens (oak, bay, box)
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

    # Name token pair conjunctions
    for i in range(len(n_tokens)):
        for j in range(i + 1, min(i + 4, len(n_tokens))):
            pair = tuple(sorted([n_tokens[i], n_tokens[j]]))
            c_keys.add(f'np:{pair[0]}_{pair[1]}')

    # Name x Number conjunctions
    for t in n_tokens[:3]:
        for num in nums:
            c_keys.add(f'n_num:{t}_{num}')

    # Number x Address token conjunctions
    for num in nums:
        for at in a_tokens[:3]:
            c_keys.add(f'num_a:{num}_{at}')

    # Name x Address token conjunctions (incorporating street at start and city/state at end)
    target_a = list(dict.fromkeys(a_tokens[:2] + a_tokens[-2:])) if len(a_tokens) > 2 else a_tokens
    for t in n_tokens[:3]:
        for at in target_a:
            c_keys.add(f'n_a:{t}_{at}')

    # Squashed / domain single token subword n-grams
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

    return {'exact': exact_name_idx, 'name': n_idx, 'addr': a_idx, 'comp': c_idx}


def retrieve_candidates_v3(nc, ac, idx, comp_k=60, name_k=40, addr_k=40, exact_k=100):
    """Enhanced retrieval with IDF weighting."""
    cands = set()
    exact_idx = idx['exact']
    c_idx = idx['comp']
    n_idx = idx['name']
    a_idx = idx['addr']

    if nc and nc in exact_idx:
        cands.update(exact_idx[nc][:exact_k])

    nk, ak, ck = extract_keys_v3(nc, ac)

    # 1. Compound pass (highest precision)
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

    # 2. Name pass
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

    # 3. Addr pass
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


def run_blocking_experiment(n_val=10000):
    print("=" * 80)
    print(f" CONTROLLED BLOCKING EXPERIMENT: V2 vs V3 (Validation sample: {n_val:,} entities)")
    print("=" * 80)

    # Load Ground Truth
    gt_df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv'), sep='\t', keep_default_na=False)
    gt_map = {}
    for sid, mids in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']):
        gt_map[sid] = set(mids.split(',')) if mids else set()

    # Load S1 validation split
    s1_cols = ['entity_id', 'country', 'business_name', 'business_address']
    s1_all = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source1.tsv'), sep='\t', usecols=s1_cols, keep_default_na=False)

    np.random.seed(42)
    countries = ['US', 'India']
    val_s1_list = []
    for c in countries:
        s1_c = s1_all[s1_all['country'] == c].copy()
        s1_c = s1_c.sample(frac=1.0, random_state=42).reset_index(drop=True)
        c_ratio = len(s1_c) / len(s1_all)
        n_va = int(n_val * c_ratio)
        val_s1_list.append(s1_c.iloc[:n_va])

    val_s1_df = pd.concat(val_s1_list, ignore_index=True)
    del s1_all, val_s1_list
    gc.collect()

    val_s1_df['name_clean'] = val_s1_df['business_name'].apply(clean_name_enhanced)
    val_s1_df['addr_clean'] = val_s1_df['business_address'].apply(normalize_address)

    total_true = sum(len(gt_map.get(sid, set())) for sid in val_s1_df['entity_id'])

    v2_found = 0
    v2_cand_counts = []
    v3_found = 0
    v3_cand_counts = []

    for country in countries:
        c_s1 = val_s1_df[val_s1_df['country'] == country]
        print(f"\nEvaluating Country: {country} ({len(c_s1):,} S1 entities)...")

        s2_df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source2.tsv'), sep='\t',
                            usecols=['entity_id', 'country', 'business_name', 'business_address'], keep_default_na=False)
        s3_df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source3.tsv'), sep='\t',
                            usecols=['entity_id', 'country', 'business_name', 'business_address'], keep_default_na=False)
        s2_c = s2_df[s2_df['country'] == country].copy()
        s3_c = s3_df[s3_df['country'] == country].copy()
        del s2_df, s3_df
        gc.collect()

        s23_c = pd.concat([s2_c, s3_c], ignore_index=True)
        del s2_c, s3_c
        gc.collect()

        s23_c['name_clean'] = s23_c['business_name'].apply(clean_name_enhanced)
        s23_c['addr_clean'] = s23_c['business_address'].apply(normalize_address)

        # 1. Build V2 Index
        print("  Building Baseline V2 Index...")
        s1_n_v2, s1_a_v2, s1_c_v2, s1_names = set(), set(), set(), set()
        for r in c_s1.itertuples(index=False):
            if r.name_clean:
                s1_names.add(r.name_clean)
            nk, ak, ck = extract_keys_v2(r.name_clean, r.addr_clean)
            s1_n_v2.update(nk)
            s1_a_v2.update(ak)
            s1_c_v2.update(ck)
        idx_v2 = build_index_v2(s23_c, s1_n_v2, s1_a_v2, s1_c_v2, s1_names)

        # 2. Build V3 Index
        print("  Building Enhanced V3 Index...")
        s1_n_v3, s1_a_v3, s1_c_v3 = set(), set(), set()
        for r in c_s1.itertuples(index=False):
            nk, ak, ck = extract_keys_v3(r.name_clean, r.addr_clean)
            s1_n_v3.update(nk)
            s1_a_v3.update(ak)
            s1_c_v3.update(ck)
        idx_v3 = build_country_index_v3(s23_c, s1_n_v3, s1_a_v3, s1_c_v3, s1_names)

        # Compare retrieval
        print("  Retrieving candidates under both blocking strategies...")
        for r in c_s1.itertuples(index=False):
            sid = r.entity_id
            true_set = gt_map.get(sid, set())
            nc = r.name_clean
            ac = r.addr_clean

            # V2
            c_v2 = retrieve_cands_v2(nc, ac, idx_v2)
            v2_cand_counts.append(len(c_v2))
            v2_found += len(true_set & c_v2)

            # V3
            c_v3 = retrieve_candidates_v3(nc, ac, idx_v3)
            v3_cand_counts.append(len(c_v3))
            v3_found += len(true_set & c_v3)

        del s23_c, idx_v2, idx_v3
        gc.collect()

    # Results
    v2_recall = v2_found / max(1, total_true)
    v3_recall = v3_found / max(1, total_true)
    v2_avg_cands = np.mean(v2_cand_counts)
    v3_avg_cands = np.mean(v3_cand_counts)

    print("\n" + "=" * 80)
    print(" EXPERIMENT RESULTS: BLOCKING RECALL COMPARISON")
    print("=" * 80)
    print(f"Total True Matches:        {total_true:,}")
    print(f"Baseline V2 Recall:        {v2_recall:.4f} ({v2_recall*100:.2f}%) | True Recalled: {v2_found:,} | Avg Cands/Ent: {v2_avg_cands:.1f}")
    print(f"Enhanced V3 Recall:        {v3_recall:.4f} ({v3_recall*100:.2f}%) | True Recalled: {v3_found:,} | Avg Cands/Ent: {v3_avg_cands:.1f}")
    print(f"Absolute Recall Delta:     {v3_recall - v2_recall:+.4f} ({(v3_recall - v2_recall)*100:+.2f}%)")
    print(f"Missed Matches Eliminated: {v3_found - v2_found:,} ({((v3_found - v2_found) / max(1, total_true - v2_found))*100:.1f}% reduction in blocking misses)")
    print("=" * 80)

if __name__ == '__main__':
    run_blocking_experiment(n_val=10000)
