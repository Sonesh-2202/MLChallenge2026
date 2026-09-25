import sys, os, time, pickle, gc
import multiprocessing as mp
import numpy as np
import pandas as pd
import heapq
from rapidfuzz import fuzz

sys.path.append(r'code/business_entity_resolution/src')
from preprocess import normalize_name, normalize_address, extract_numbers, STOP_WORDS
from blocking import extract_name_keys, extract_addr_keys
from features import FEATURE_COLS

# Global worker state
_G_INDEX = None
_G_S23 = None
_G_MODEL = None
_G_THRESHOLD = 0.70

def init_worker(index_data, s23_data, model_bytes, threshold):
    global _G_INDEX, _G_S23, _G_MODEL, _G_THRESHOLD
    _G_INDEX = index_data
    _G_S23 = s23_data
    _G_MODEL = pickle.loads(model_bytes)
    _G_THRESHOLD = threshold

def process_s1_batch(s1_records, max_postings=1000, name_k=25, addr_k=25):
    """
    s1_records: list of tuples (sid, nc, ac)
    """
    idx = _G_INDEX
    s23_dict = _G_S23
    model = _G_MODEL
    threshold = _G_THRESHOLD

    matches_out = []
    candidates_out = []

    for sid, nc, ac in s1_records:
        cands = set()
        if nc and nc in idx['exact_name']:
            cands.update(idx['exact_name'][nc][:name_k])

        n_scores = {}
        for k in extract_name_keys(nc):
            postings = idx['name'].get(k)
            if postings and len(postings) <= max_postings:
                w = 1.0 / np.log1p(len(postings))
                for mid in postings:
                    n_scores[mid] = n_scores.get(mid, 0.0) + w
        if n_scores:
            cands.update(m for m, _ in heapq.nlargest(name_k, n_scores.items(), key=lambda x: x[1]))

        a_scores = {}
        for k in extract_addr_keys(ac):
            postings = idx['addr'].get(k)
            if postings and len(postings) <= max_postings:
                w = 1.0 / np.log1p(len(postings))
                for mid in postings:
                    a_scores[mid] = a_scores.get(mid, 0.0) + w
        if a_scores:
            cands.update(m for m, _ in heapq.nlargest(addr_k, a_scores.items(), key=lambda x: x[1]))

        candidates_out.append((sid, sorted(cands)))

        if not cands:
            matches_out.append((sid, []))
            continue

        # Extract features for candidate pairs
        w1_set = set(nc.split()) if nc else set()
        aw1_set = set(ac.split()) if ac else set()
        nums1 = extract_numbers(ac)
        has_a1 = bool(ac and ac != 'nan')

        n_cands = len(cands)
        X = np.empty((n_cands, 17), dtype=np.float32)
        mid_list = list(cands)

        for row_idx, mid in enumerate(mid_list):
            s23_info = s23_dict.get(mid)
            if s23_info is None:
                X[row_idx, :] = 0.0
                continue
            n2, a2, w2_set, aw2_set, nums2, has_a2 = s23_info

            n_ratio = fuzz.ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
            n_tset = fuzz.token_set_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
            n_tsort = fuzz.token_sort_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
            n_part = fuzz.partial_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
            n_jacc = len(w1_set & w2_set) / max(1, len(w1_set | w2_set)) if (w1_set or w2_set) else 0.0

            both_addr = 1.0 if (has_a1 and has_a2) else 0.0
            if both_addr:
                a_ratio = fuzz.ratio(ac, a2) / 100.0
                a_tset = fuzz.token_set_ratio(ac, a2) / 100.0
                a_tsort = fuzz.token_sort_ratio(ac, a2) / 100.0
                a_part = fuzz.partial_ratio(ac, a2) / 100.0
                a_jacc = len(aw1_set & aw2_set) / max(1, len(aw1_set | aw2_set)) if (aw1_set or aw2_set) else 0.0
            else:
                a_ratio = a_tset = a_tsort = a_part = a_jacc = 0.0

            num_match = 1.0 if (nums1 and nums2 and (nums1 & nums2)) else 0.0
            num_jacc = len(nums1 & nums2) / max(1, len(nums1 | nums2)) if (nums1 or nums2) else 0.0

            eff_a_score = a_tset if both_addr else n_tset
            harm = (2.0 * n_tset * eff_a_score) / (n_tset + eff_a_score + 1e-5)
            exact_n = 1.0 if (nc and nc == n2) else 0.0
            exact_a = 1.0 if (ac and ac == a2) else 0.0

            X[row_idx, 0] = n_ratio
            X[row_idx, 1] = n_tset
            X[row_idx, 2] = n_tsort
            X[row_idx, 3] = n_part
            X[row_idx, 4] = n_jacc
            X[row_idx, 5] = a_ratio
            X[row_idx, 6] = a_tset
            X[row_idx, 7] = a_tsort
            X[row_idx, 8] = a_part
            X[row_idx, 9] = a_jacc
            X[row_idx, 10] = num_match
            X[row_idx, 11] = num_jacc
            X[row_idx, 12] = both_addr
            X[row_idx, 13] = 1.0 - both_addr
            X[row_idx, 14] = harm
            X[row_idx, 15] = exact_n
            X[row_idx, 16] = exact_a

        probs = model.predict(X)
        matched_mids = [mid_list[i] for i in range(n_cands) if probs[i] >= threshold]
        matches_out.append((sid, matched_mids))

    return matches_out, candidates_out

if __name__ == '__main__':
    mp.freeze_support()
    print("Testing parallel worker pipeline on France sample...")
    t0 = time.time()
    
    # Load model
    with open(r'cache/lgbm_model.pkl', 'rb') as f:
        mdata = pickle.load(f)
    model_bytes = pickle.dumps(mdata['model'])
    threshold = mdata['threshold']

    # Load 10,000 S1 France
    s1 = pd.read_csv(r'dataset/test/test_source1.tsv', sep='\t', nrows=50000, dtype=str).fillna('')
    s1_fr = s1[s1['country'] == 'France'].head(10000).copy()
    s1_fr['name_clean'] = s1_fr['business_name'].apply(normalize_name)
    s1_fr['addr_clean'] = s1_fr['business_address'].apply(normalize_address)
    s1_records = list(zip(s1_fr['entity_id'], s1_fr['name_clean'], s1_fr['addr_clean']))
    print(f"Loaded {len(s1_records)} S1 France in {time.time()-t0:.2f}s")

    # Load S2 France
    s2_list = [c[c['country']=='France'] for c in pd.read_csv(r'dataset/test/test_source2.tsv', sep='\t', chunksize=500000, dtype=str)]
    s23_fr = pd.concat(s2_list, ignore_index=True).drop_duplicates('entity_id').head(200000)
    s23_fr['name_clean'] = s23_fr['business_name'].apply(normalize_name)
    s23_fr['addr_clean'] = s23_fr['business_address'].apply(normalize_address)
    
    # Build index & s23_dict
    exact_name_idx = {}
    name_idx = {}
    addr_idx = {}
    s23_dict = {}

    for r in s23_fr.itertuples(index=False):
        eid = r.entity_id
        nc = r.name_clean
        ac = r.addr_clean
        if nc:
            exact_name_idx.setdefault(nc, []).append(eid)
            for k in extract_name_keys(nc):
                name_idx.setdefault(k, []).append(eid)
        if ac:
            for k in extract_addr_keys(ac):
                addr_idx.setdefault(k, []).append(eid)
        s23_dict[eid] = (
            nc, ac,
            set(nc.split()) if nc else set(),
            set(ac.split()) if ac else set(),
            extract_numbers(ac),
            bool(ac and ac != 'nan')
        )

    idx_data = {'exact_name': exact_name_idx, 'name': name_idx, 'addr': addr_idx}
    print(f"Built index in {time.time()-t0:.2f}s")

    # Split S1 into chunks of 1,000
    chunk_size = 1000
    chunks = [s1_records[i:i+chunk_size] for i in range(0, len(s1_records), chunk_size)]
    
    t_start = time.time()
    n_workers = min(8, mp.cpu_count())
    print(f"Spawning {n_workers} workers for {len(chunks)} chunks...")
    with mp.Pool(processes=n_workers, initializer=init_worker, initargs=(idx_data, s23_dict, model_bytes, threshold)) as pool:
        results = pool.map(process_s1_batch, chunks)
    
    t_end = time.time()
    all_matches = [m for res in results for m in res[0]]
    all_cands = [c for res in results for c in res[1]]
    total_matches = sum(len(m[1]) for m in all_matches)
    total_cands = sum(len(c[1]) for c in all_cands)

    print(f"DONE in {t_end - t_start:.2f}s!")
    print(f"Processed: {len(all_matches)} entities ({len(all_matches)/(t_end-t_start):.1f} entities/sec)")
    print(f"Total Matches: {total_matches}, Total Candidates: {total_cands}")
