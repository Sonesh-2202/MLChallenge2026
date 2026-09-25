import pandas as pd
import numpy as np
import time
import unicodedata
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("Testing Name/Number Priority Candidate Retrieval...")

import os
sys.path.insert(0, os.path.dirname(__file__))
from tune_top_k import clean_text, extract_numbers, STOP_WORDS, gt_dict, s1_all, s1_ids, s23_pool, s23_dict, total_trues

def extract_structured_keys(name_clean, addr_clean):
    name_keys = set()
    num_keys = set()
    addr_keys = set()
    
    tokens = [w for w in name_clean.split() if w not in STOP_WORDS]
    for t in tokens:
        if len(t) >= 3:
            name_keys.add('n:' + t)
        elif len(t) == 2 and t.isalnum():
            name_keys.add('n2:' + t)
        if len(t) >= 4:
            name_keys.add('ng:' + t[:4])
            
    nums = extract_numbers(addr_clean)
    for num in nums:
        num_keys.add('num:' + num)
        
    addr_words = [w for w in addr_clean.split() if len(w) >= 4 and w not in STOP_WORDS]
    for w in addr_words:
        addr_keys.add('a:' + w)
        
    return name_keys, num_keys, addr_keys

# Build index separated by type
index_by_country = {}
for eid, row in s23_dict.items():
    c = row['country']
    if c not in index_by_country:
        index_by_country[c] = {'name': {}, 'num': {}, 'addr': {}}
    idx = index_by_country[c]
    nk, numk, ak = extract_structured_keys(row['name_clean'], row['addr_clean'])
    for k in nk:
        idx['name'].setdefault(k, []).append(eid)
    for k in numk:
        idx['num'].setdefault(k, []).append(eid)
    for k in ak:
        idx['addr'].setdefault(k, []).append(eid)

print("Structured index built. Running retrieval...")

for max_k in [25, 35, 50]:
    found = 0
    t0 = time.time()
    for _, row in s1_all.iterrows():
        sid = row['entity_id']
        c = row['country']
        idx = index_by_country.get(c, {'name': {}, 'num': {}, 'addr': {}})
        nk, numk, ak = extract_structured_keys(row['name_clean'], row['addr_clean'])
        
        scores = {}
        # Priority 1: Name keys (strongest signal)
        for k in nk:
            if k in idx['name']:
                postings = idx['name'][k]
                w = 10.0 / np.log1p(len(postings))
                for mid in postings:
                    scores[mid] = scores.get(mid, 0.0) + w
                    
        # Priority 2: Address number keys (strongest location anchor)
        for k in numk:
            if k in idx['num']:
                postings = idx['num'][k]
                w = 8.0 / np.log1p(len(postings))
                for mid in postings:
                    scores[mid] = scores.get(mid, 0.0) + w
                    
        # Priority 3: Address locality keys (secondary signal)
        for k in ak:
            if k in idx['addr']:
                postings = idx['addr'][k]
                if len(postings) < 1000:
                    w = 1.0 / np.log1p(len(postings))
                    for mid in postings:
                        scores[mid] = scores.get(mid, 0.0) + w
                        
        if scores:
            top_k = {mid for mid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_k]}
        else:
            top_k = set()
            
        found += len(gt_dict.get(sid, set()) & top_k)
    elapsed = time.time() - t0
    print(f"Structured Top-{max_k}: Recall = {found:,}/{total_trues:,} ({found/total_trues*100:.2f}%)  [{elapsed:.1f}s]")
