import time
import multiprocessing as mp
import numpy as np
import heapq

_WORKER_INDEX = None

def init_worker(idx):
    global _WORKER_INDEX
    _WORKER_INDEX = idx

def process_chunk(chunk):
    results = {}
    name_k = 25
    addr_k = 25
    max_p = 1000
    idx = _WORKER_INDEX
    
    for sid, c, nc, ac in chunk:
        cands = set()
        if nc and nc in idx.get('exact_name_idx', {}):
            cands.update(idx['exact_name_idx'][nc][:name_k])
            
        n_scores = {}
        for k in nc.split():
            postings = idx['name_idx'].get('n:' + k)
            if postings and len(postings) <= max_p:
                w = 1.0 / np.log1p(len(postings))
                for mid in postings:
                    n_scores[mid] = n_scores.get(mid, 0.0) + w
        if n_scores:
            cands.update(m for m, _ in heapq.nlargest(name_k, n_scores.items(), key=lambda x: x[1]))
            
        results[sid] = cands
    return results

if __name__ == '__main__':
    mp.freeze_support()
    print("Testing MP worker...")
    dummy_idx = {
        'exact_name_idx': {'acme': ['e1', 'e2']},
        'name_idx': {'n:acme': ['e1', 'e2', 'e3']}
    }
    sample_data = [(f's_{i}', 'US', 'acme corp', '123 main st') for i in range(50000)]
    chunk_size = 5000
    chunks = [sample_data[i:i+chunk_size] for i in range(0, len(sample_data), chunk_size)]
    
    t0 = time.time()
    n_workers = min(8, mp.cpu_count())
    with mp.Pool(processes=n_workers, initializer=init_worker, initargs=(dummy_idx,)) as pool:
        res_list = pool.map(process_chunk, chunks)
    
    total = sum(len(r) for r in res_list)
    print(f"Success! Processed {total} records with {n_workers} workers in {time.time()-t0:.2f}s")
