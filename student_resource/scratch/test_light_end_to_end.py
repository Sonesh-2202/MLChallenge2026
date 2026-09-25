import sys, os, time, pickle, gc
import multiprocessing as mp
import numpy as np
import pandas as pd
import heapq
from rapidfuzz import fuzz

sys.path.append(r'code/business_entity_resolution/src')
from preprocess import normalize_name, normalize_address, extract_numbers
from blocking import extract_name_keys, extract_addr_keys

_G_INDEX = None
_G_S23 = None
_G_MODEL = None
_G_THRESHOLD = 0.70

def init_worker(idx_data, s23_compact, model_bytes, threshold):
    global _G_INDEX, _G_S23, _G_MODEL, _G_THRESHOLD
    _G_INDEX = idx_data
    _G_S23 = s23_compact
    _G_MODEL = pickle.loads(model_bytes)
    _G_THRESHOLD = threshold

def process_chunk(chunk, max_p=1000, name_k=25, addr_k=25):
    idx = _G_INDEX
    s23_dict = _G_S23
    model = _G_MODEL
    threshold = _G_THRESHOLD

    chunk_candidates = []
    total_pairs = 0

    for sid, nc, ac in chunk:
        cands = set()
        if nc and nc in idx['exact_name']:
            cands.update(idx['exact_name'][nc][:name_k])

        n_scores = {}
        for k in extract_name_keys(nc):
            postings = idx['name'].get(k)
            if postings and len(postings) <= max_p:
                w = 1.0 / np.log1p(len(postings))
                for mid in postings:
                    n_scores[mid] = n_scores.get(mid, 0.0) + w
        if n_scores:
            cands.update(m for m, _ in heapq.nlargest(name_k, n_scores.items(), key=lambda x: x[1]))

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

    X = np.empty((total_pairs, 17), dtype=np.float32)
    pair_meta = []
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

            n_ratio = fuzz.ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
            n_tset = fuzz.token_set_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
            n_tsort = fuzz.token_sort_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
            n_part = fuzz.partial_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0

            w2_set = set(n2.split()) if n2 else set()
            n_jacc = len(w1_set & w2_set) / max(1, len(w1_set | w2_set)) if (w1_set or w2_set) else 0.0

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

    probs = model.predict(X)
    matches_map = {sid: [] for sid, _, _, _ in chunk_candidates}
    for (sid, mid), p in zip(pair_meta, probs):
        if p >= threshold:
            matches_map[sid].append(mid)

    matches_out = [(sid, matches_map[sid]) for sid, _, _, _ in chunk_candidates]
    candidates_out = [(sid, cands) for sid, _, _, cands in chunk_candidates]

    return matches_out, candidates_out

if __name__ == '__main__':
    mp.freeze_support()
    print("Testing ultra-light end-to-end parallel inference...")
    t0 = time.time()
    
    with open(r'cache/lgbm_model.pkl', 'rb') as f:
        mdata = pickle.load(f)
    model_bytes = pickle.dumps(mdata['model'])
    threshold = mdata['threshold']

    s1 = pd.read_csv(r'dataset/test/test_source1.tsv', sep='\t', nrows=100000, dtype=str).fillna('')
    s1_fr = s1[s1['country'] == 'France'].head(15000).copy()
    s1_fr['name_clean'] = s1_fr['business_name'].apply(normalize_name)
    s1_fr['addr_clean'] = s1_fr['business_address'].apply(normalize_address)
    s1_records = list(zip(s1_fr['entity_id'], s1_fr['name_clean'], s1_fr['addr_clean']))

    s2_list = [c[c['country']=='France'] for c in pd.read_csv(r'dataset/test/test_source2.tsv', sep='\t', chunksize=500000, dtype=str)]
    s23_fr = pd.concat(s2_list, ignore_index=True).drop_duplicates('entity_id').head(300000)
    s23_fr['name_clean'] = s23_fr['business_name'].apply(normalize_name)
    s23_fr['addr_clean'] = s23_fr['business_address'].apply(normalize_address)

    exact_name_idx, name_idx, addr_idx = {}, {}, {}
    s23_compact = {}
    for r in s23_fr.itertuples(index=False):
        eid, nc, ac = r.entity_id, r.name_clean, r.addr_clean
        if nc:
            exact_name_idx.setdefault(nc, []).append(eid)
            for k in extract_name_keys(nc): name_idx.setdefault(k, []).append(eid)
        if ac:
            for k in extract_addr_keys(ac): addr_idx.setdefault(k, []).append(eid)
        s23_compact[eid] = (nc, ac)

    idx_data = {'exact_name': exact_name_idx, 'name': name_idx, 'addr': addr_idx}

    chunk_size = 2500
    chunks = [s1_records[i:i+chunk_size] for i in range(0, len(s1_records), chunk_size)]
    
    t_start = time.time()
    n_workers = min(12, mp.cpu_count())
    with mp.Pool(processes=n_workers, initializer=init_worker, initargs=(idx_data, s23_compact, model_bytes, threshold)) as pool:
        results = pool.map(process_chunk, chunks)
    
    t_end = time.time()
    all_matches = [m for res in results for m in res[0]]
    total_matches = sum(len(m[1]) for m in all_matches)
    print(f"End-to-End: 15,000 entities in {t_end - t_start:.2f}s ({len(all_matches)/(t_end-t_start):.0f} entities/sec)!")
    print(f"Total Matches: {total_matches}")
