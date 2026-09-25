import sys, time
import pandas as pd
import numpy as np
import heapq

sys.path.append(r'code/business_entity_resolution/src')
from preprocess import normalize_name, normalize_address, extract_numbers, STOP_WORDS
from blocking import CountryInvertedIndex, extract_name_keys, extract_addr_keys

print("Loading test S1 France (50,000 records)...")
t0 = time.time()
s1 = pd.read_csv(r'dataset/test/test_source1.tsv', sep='\t', dtype=str).fillna('')
s1_fr = s1[s1['country'] == 'France'].head(50000).copy()
s1_fr['name_clean'] = s1_fr['business_name'].apply(normalize_name)
s1_fr['addr_clean'] = s1_fr['business_address'].apply(normalize_address)
print(f"Loaded {len(s1_fr)} S1 France in {time.time()-t0:.2f}s")

# Load France S2+S3 (sample 200,000 for quick test)
s2_list = [c[c['country']=='France'] for c in pd.read_csv(r'dataset/test/test_source2.tsv', sep='\t', chunksize=500000, dtype=str)]
s23_fr = pd.concat(s2_list, ignore_index=True).drop_duplicates('entity_id').head(300000)
s23_fr['name_clean'] = s23_fr['business_name'].apply(normalize_name)
s23_fr['addr_clean'] = s23_fr['business_address'].apply(normalize_address)
print(f"Loaded {len(s23_fr)} S23 France records in {time.time()-t0:.2f}s")

idx = CountryInvertedIndex('France')
idx.add_records(s23_fr[['entity_id', 'name_clean', 'addr_clean']].to_dict('records'))

# Run retrieval on 50,000 queries with max_postings=1000
t1 = time.time()
max_p = 1000
name_k = 25
addr_k = 25
cands_count = 0

for row in s1_fr.itertuples(index=False):
    nc = row.name_clean
    ac = row.addr_clean
    cands = set()
    if nc in idx.exact_name_idx:
        cands.update(idx.exact_name_idx[nc][:name_k])
    n_scores = {}
    for k in extract_name_keys(nc):
        if k in idx.name_idx:
            postings = idx.name_idx[k]
            if len(postings) > max_p:
                continue
            w = 1.0 / np.log1p(len(postings))
            for mid in postings:
                n_scores[mid] = n_scores.get(mid, 0.0) + w
    if n_scores:
        cands.update(m for m, _ in heapq.nlargest(name_k, n_scores.items(), key=lambda x: x[1]))
    
    a_scores = {}
    for k in extract_addr_keys(ac):
        if k in idx.addr_idx:
            postings = idx.addr_idx[k]
            if len(postings) > max_p:
                continue
            w = 1.0 / np.log1p(len(postings))
            for mid in postings:
                a_scores[mid] = a_scores.get(mid, 0.0) + w
    if a_scores:
        cands.update(m for m, _ in heapq.nlargest(addr_k, a_scores.items(), key=lambda x: x[1]))
    cands_count += len(cands)

elap = time.time() - t1
print(f"Retrieved candidates for {len(s1_fr)} entities in {elap:.2f}s ({elap/len(s1_fr)*1000:.3f} ms/query, {len(s1_fr)/elap:.0f} queries/sec)")
print(f"Total candidates: {cands_count}, avg: {cands_count/len(s1_fr):.1f}/query")
