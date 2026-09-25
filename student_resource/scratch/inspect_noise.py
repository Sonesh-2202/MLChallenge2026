import pandas as pd
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

print("Loading 10,000 S1 and GT...")
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=10000)
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=10000)

s1_dict = s1.set_index('entity_id').to_dict('index')

# Select 10 diverse examples (including US and India)
matches = []
for _, row in gt.iterrows():
    s1_id = row['source1_entity_id']
    if s1_id in s1_dict and pd.notna(row['matched_entity_ids']) and str(row['matched_entity_ids']).strip():
        m_list = str(row['matched_entity_ids']).split(',')
        matches.append((s1_id, m_list))
    if len(matches) >= 15:
        break

needed_s2 = {m for _, ms in matches for m in ms if m.startswith('S2-')}
needed_s3 = {m for _, ms in matches for m in ms if m.startswith('S3-')}

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

for s1_id, m_ids in matches[:10]:
    s1_rec = s1_dict[s1_id]
    print("=" * 80)
    print(f"S1: {s1_id} | Country: {s1_rec['country']}")
    print(f"    Name:    '{s1_rec['business_name']}'")
    print(f"    Address: '{s1_rec['business_address']}'")
    for mid in m_ids:
        rec = s2_matches.get(mid) or s3_matches.get(mid) or {}
        print(f"  -> {mid} | Country: {rec.get('country')}")
        print(f"     Name:    '{rec.get('business_name')}'")
        print(f"     Address: '{rec.get('business_address')}'")
