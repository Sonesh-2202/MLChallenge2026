import pandas as pd
import sys

# Read first chunk of ground truth
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=10000)
# Find entities with at least 1 match
gt_matches = gt[gt['matched_entity_ids'].notna() & (gt['matched_entity_ids'] != '')]

# Collect needed IDs
needed_s1 = set(gt_matches['source1_entity_id'].head(20))
needed_s2 = set()
needed_s3 = set()
for _, row in gt_matches.head(20).iterrows():
    for mid in row['matched_entity_ids'].split(','):
        if mid.startswith('S2-'):
            needed_s2.add(mid)
        elif mid.startswith('S3-'):
            needed_s3.add(mid)

print(f"Need {len(needed_s1)} S1, {len(needed_s2)} S2, {len(needed_s3)} S3")

# Read S1
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
s1_subset = s1[s1['entity_id'].isin(needed_s1)].set_index('entity_id').to_dict('index')
del s1

# Read S2 in chunks
s2_subset = {}
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=500000):
    match = chunk[chunk['entity_id'].isin(needed_s2)]
    for _, r in match.iterrows():
        s2_subset[r['entity_id']] = r.to_dict()
    if len(s2_subset) == len(needed_s2):
        break

# Read S3 in chunks
s3_subset = {}
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=500000):
    match = chunk[chunk['entity_id'].isin(needed_s3)]
    for _, r in match.iterrows():
        s3_subset[r['entity_id']] = r.to_dict()
    if len(s3_subset) == len(needed_s3):
        break

print("\n=== DETAILED MATCH INSPECTION ===\n")
for _, row in gt_matches.head(15).iterrows():
    s1_id = row['source1_entity_id']
    if s1_id not in s1_subset:
        continue
    r1 = s1_subset[s1_id]
    print(f"==================================================")
    print(f"S1 ID: {s1_id} | Country: {r1['country']}")
    print(f"   Name:    {r1['business_name']}")
    print(f"   Address: {r1['business_address']}")
    for mid in row['matched_entity_ids'].split(','):
        mrec = s2_subset.get(mid) or s3_subset.get(mid)
        if mrec:
            prefix = mid[:2]
            print(f"  {prefix} ID: {mid} | Country: {mrec['country']}")
            print(f"     Name:    {mrec['business_name']}")
            print(f"     Address: {mrec['business_address']}")
        else:
            print(f"  {mid}: [Not in searched chunks]")
    print()
