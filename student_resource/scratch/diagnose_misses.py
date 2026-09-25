import pandas as pd
import numpy as np
import sys
import unicodedata
import re

sys.stdout.reconfigure(encoding='utf-8')

import os
sys.path.insert(0, os.path.dirname(__file__))
from test_pipeline_val import clean_text, extract_blocking_keys, STOP_WORDS

print("Loading 10,000 S1 and GT...")
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
gt_dict = {}
for _, r in gt_df.iterrows():
    m = r['matched_entity_ids']
    gt_dict[r['source1_entity_id']] = set(str(m).split(',')) if (pd.notna(m) and str(m).strip()) else set()

s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=10000)
s1_ids = set(s1_df['entity_id'])

needed = set()
for sid in s1_ids:
    needed.update(gt_dict.get(sid, set()))

# Load matches
s2_matches = []
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=500000):
    hits = chunk[chunk['entity_id'].isin(needed)]
    s2_matches.append(hits)
s3_matches = []
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=500000):
    hits = chunk[chunk['entity_id'].isin(needed)]
    s3_matches.append(hits)

s23_df = pd.concat(s2_matches + s3_matches, ignore_index=True)
s1_dict = s1_df.set_index('entity_id').to_dict('index')
s23_dict = s23_df.set_index('entity_id').to_dict('index')

missed_keys = []
has_key_overlap = 0
total_pairs = 0

for sid in s1_ids:
    r1 = s1_dict[sid]
    k1 = extract_blocking_keys(clean_text(r1['business_name']), clean_text(r1['business_address'], True))
    for mid in gt_dict.get(sid, set()):
        if mid in s23_dict:
            total_pairs += 1
            r2 = s23_dict[mid]
            k2 = extract_blocking_keys(clean_text(r2['business_name']), clean_text(r2['business_address'], True))
            overlap = k1 & k2
            if overlap:
                has_key_overlap += 1
            else:
                missed_keys.append((sid, mid, r1, r2, k1, k2))

print(f"Total evaluated true pairs: {total_pairs}")
print(f"Pairs with key overlap: {has_key_overlap} ({has_key_overlap/total_pairs*100:.2f}%)")
print(f"Pairs without key overlap: {len(missed_keys)} ({len(missed_keys)/total_pairs*100:.2f}%)")

print("\nInspecting first 10 misses:")
for sid, mid, r1, r2, k1, k2 in missed_keys[:10]:
    print("="*70)
    print(f"S1:  Name='{r1['business_name']}', Addr='{r1['business_address']}'")
    print(f"S23: Name='{r2['business_name']}', Addr='{r2['business_address']}'")
    print(f"K1: {k1}")
    print(f"K2: {k2}")
