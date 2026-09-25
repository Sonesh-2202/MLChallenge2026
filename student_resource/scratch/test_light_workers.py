import sys, os, time, gc
import multiprocessing as mp
import numpy as np
import pandas as pd
import heapq

sys.path.append(r'code/business_entity_resolution/src')
from preprocess import normalize_name, normalize_address
from blocking import extract_name_keys, extract_addr_keys

_G_INDEX = None

def init_worker(idx_data):
    global _G_INDEX
    _G_INDEX = idx_data

def retrieve_chunk(chunk, max_postings=1000, name_k=25, addr_k=25):
    idx = _G_INDEX
    res = []
    for sid, nc, ac in chunk:
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

        res.append((sid, sorted(cands)))
    return res

if __name__ == '__main__':
    mp.freeze_support()
    print("Testing lightweight index parallel retrieval...")
    t0 = time.time()
    
    # Load 50,000 France S1
    s1 = pd.read_csv(r'dataset/test/test_source1.tsv', sep='\t', nrows=100000, dtype=str).fillna('')
    s1_fr = s1[s1['country'] == 'France'].head(50000).copy()
    s1_fr['name_clean'] = s1_fr['business_name'].apply(normalize_name)
    s1_fr['addr_clean'] = s1_fr['business_address'].apply(normalize_address)
    s1_records = list(zip(s1_fr['entity_id'], s1_fr['name_clean'], s1_fr['addr_clean']))
    print(f"Loaded {len(s1_records)} S1 France in {time.time()-t0:.2f}s")

    # Load 300,000 France S2
    s2_list = [c[c['country']=='France'] for c in pd.read_csv(r'dataset/test/test_source2.tsv', sep='\t', chunksize=500000, dtype=str)]
    s23_fr = pd.concat(s2_list, ignore_index=True).drop_duplicates('entity_id').head(300000)
    s23_fr['name_clean'] = s23_fr['business_name'].apply(normalize_name)
    s23_fr['addr_clean'] = s23_fr['business_address'].apply(normalize_address)

    exact_name_idx, name_idx, addr_idx = {}, {}, {}
    for r in s23_fr.itertuples(index=False):
        eid, nc, ac = r.entity_id, r.name_clean, r.addr_clean
        if nc:
            exact_name_idx.setdefault(nc, []).append(eid)
            for k in extract_name_keys(nc): name_idx.setdefault(k, []).append(eid)
        if ac:
            for k in extract_addr_keys(ac): addr_idx.setdefault(k, []).append(eid)

    idx_data = {'exact_name': exact_name_idx, 'name': name_idx, 'addr': addr_idx}
    print(f"Index built in {time.time()-t0:.2f}s")

    chunk_size = 2500
    chunks = [s1_records[i:i+chunk_size] for i in range(0, len(s1_records), chunk_size)]
    
    t_start = time.time()
    n_workers = min(12, mp.cpu_count())
    with mp.Pool(processes=n_workers, initializer=init_worker, initargs=(idx_data,)) as pool:
        results = pool.map(retrieve_chunk, chunks)
    
    t_end = time.time()
    all_cands = [item for sub in results for item in sub]
    total_cands = sum(len(c[1]) for c in all_cands)
    print(f"Retrieved {len(all_cands)} entities with {n_workers} workers in {t_end - t_start:.2f}s ({len(all_cands)/(t_end-t_start):.0f} entities/sec)!")
    print(f"Total candidates: {total_cands}")
