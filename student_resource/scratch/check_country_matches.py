import pandas as pd

print("Loading S1 countries...")
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', usecols=['entity_id', 'country'])
s1_c = dict(zip(s1['entity_id'], s1['country']))
del s1

print("Loading ground truth (first 100k non-empty)...")
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=100000)
gt = gt[gt['matched_entity_ids'].notna() & (gt['matched_entity_ids'] != '')]

target_s2 = set()
target_s3 = set()
for _, r in gt.iterrows():
    for mid in r['matched_entity_ids'].split(','):
        if mid.startswith('S2-'):
            target_s2.add(mid)
        elif mid.startswith('S3-'):
            target_s3.add(mid)

print(f"Checking {len(target_s2)} S2 IDs and {len(target_s3)} S3 IDs...")

s2_c = {}
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', usecols=['entity_id', 'country'], chunksize=1000000):
    match = chunk[chunk['entity_id'].isin(target_s2)]
    s2_c.update(dict(zip(match['entity_id'], match['country'])))
    if len(s2_c) == len(target_s2):
        break

s3_c = {}
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', usecols=['entity_id', 'country'], chunksize=1000000):
    match = chunk[chunk['entity_id'].isin(target_s3)]
    s3_c.update(dict(zip(match['entity_id'], match['country'])))
    if len(s3_c) == len(target_s3):
        break

mismatches = 0
total_checked = 0
for _, r in gt.iterrows():
    c1 = s1_c[r['source1_entity_id']]
    for mid in r['matched_entity_ids'].split(','):
        c2 = s2_c.get(mid) or s3_c.get(mid)
        if c2:
            total_checked += 1
            if c1 != c2:
                mismatches += 1

print(f"Total checked: {total_checked}, Country mismatches: {mismatches}")
if total_checked > 0:
    print(f"Country match consistency: {100.0 * (total_checked - mismatches) / total_checked:.4f}%")
