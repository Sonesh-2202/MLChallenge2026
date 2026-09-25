import pandas as pd
import sys

# Ensure UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# Read S1 and GT
print("Loading first 5000 S1 and GT...")
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=5000)
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=5000)

s1_dict = s1.set_index('entity_id').to_dict('index')

sample_matches = []
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    if s1_id in s1_dict and pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip():
        sample_matches.append((s1_id, str(row['matched_entity_ids']).split(',')))
    if len(sample_matches) >= 8:
        break

needed_s2 = {m for _, ms in sample_matches for m in ms if m.startswith('S2-')}
needed_s3 = {m for _, ms in sample_matches for m in ms if m.startswith('S3-')}

print(f"Looking for {len(needed_s2)} S2 IDs and {len(needed_s3)} S3 IDs...")

s2_matches = {}
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=200000):
    hits = chunk[chunk['entity_id'].isin(needed_s2)]
    for _, r in hits.iterrows():
        s2_matches[r['entity_id']] = r.to_dict()
    if len(s2_matches) == len(needed_s2):
        break

s3_matches = {}
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=200000):
    hits = chunk[chunk['entity_id'].isin(needed_s3)]
    for _, r in hits.iterrows():
        s3_matches[r['entity_id']] = r.to_dict()
    if len(s3_matches) == len(needed_s3):
        break

print(f"Found {len(s2_matches)} S2 matches and {len(s3_matches)} S3 matches.")

for s1_id, m_ids in sample_matches:
    s1_rec = s1_dict[s1_id]
    c = s1_rec.get('country', '')
    nm = s1_rec.get('business_name', '')
    addr = s1_rec.get('business_address', '')
    print("=" * 80)
    print(f"S1 [{s1_id}] (Country: {c})")
    print(f"   Name:    {nm}")
    print(f"   Address: {addr}")
    for mid in m_ids:
        rec = s2_matches.get(mid) or s3_matches.get(mid) or {}
        print(f" -> MATCH [{mid}] (Country: {rec.get('country')})")
        print(f"      Name:    {rec.get('business_name')}")
        print(f"      Address: {rec.get('business_address')}")
