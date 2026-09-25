import pandas as pd
import numpy as np
from rapidfuzz import fuzz
import sys
import re

sys.stdout.reconfigure(encoding='utf-8')

print("Reading 20,000 train S1 and GT...")
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=20000)
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=20000)

s1_dict = s1.set_index('entity_id').to_dict('index')

gt_pairs = []
for _, r in gt.iterrows():
    s1_id = r['source1_entity_id']
    if s1_id in s1_dict and pd.notna(r['matched_entity_ids']) and str(r['matched_entity_ids']).strip():
        for mid in str(r['matched_entity_ids']).split(','):
            gt_pairs.append((s1_id, mid))

print(f"Total ground truth pairs in sample: {len(gt_pairs)}")

needed_s2 = {m for _, m in gt_pairs if m.startswith('S2-')}
needed_s3 = {m for _, m in gt_pairs if m.startswith('S3-')}

print(f"Loading needed {len(needed_s2)} S2 and {len(needed_s3)} S3 records...")
s2_map = {}
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=250000):
    hits = chunk[chunk['entity_id'].isin(needed_s2)]
    for _, r in hits.iterrows():
        s2_map[r['entity_id']] = r.to_dict()
    if len(s2_map) == len(needed_s2):
        break

s3_map = {}
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=250000):
    hits = chunk[chunk['entity_id'].isin(needed_s3)]
    for _, r in hits.iterrows():
        s3_map[r['entity_id']] = r.to_dict()
    if len(s3_map) == len(needed_s3):
        break

print(f"Loaded {len(s2_map)} S2 and {len(s3_map)} S3.")

LEGAL_SUFFIXES = {
    'llc', 'inc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private', 'co', 'company',
    'llp', 'plc', 'sarl', 'gmbh', 'ag', 'dba', 'd.b.a.', 'sas', 'sa', 'eurl', 'srl', 'sl', 'pty',
    'pllc', 'pc', 'group', 'services', 'industries', 'associates', 'federation'
}

def clean_toks(s):
    if not s or pd.isna(s):
        return []
    s = re.sub(r'[^\w\s]', ' ', str(s).lower())
    return [w for w in s.split() if w not in LEGAL_SUFFIXES]

def extract_numbers(s):
    if not s or pd.isna(s):
        return set()
    return set(re.findall(r'\b\d+[-\w/]*\b', str(s).lower()))

# Check statistics across gt_pairs
name_exact = 0
name_fuzz_90 = 0
name_fuzz_70 = 0
name_tok_overlap = 0

addr_exact = 0
addr_fuzz_80 = 0
addr_fuzz_60 = 0
addr_num_match = 0

either_name70_or_addr70 = 0
either_name80_or_addr80 = 0
hard_cases = []

valid_pairs = [p for p in gt_pairs if p[1] in s2_map or p[1] in s3_map]
print(f"Evaluating {len(valid_pairs)} complete pairs...")

for s1_id, mid in valid_pairs:
    r1 = s1_dict[s1_id]
    r2 = s2_map.get(mid) or s3_map.get(mid)
    
    n1, n2 = str(r1.get('business_name', '')), str(r2.get('business_name', ''))
    a1, a2 = str(r1.get('business_address', '')), str(r2.get('business_address', ''))
    
    t1 = set(clean_toks(n1))
    t2 = set(clean_toks(n2))
    
    nums1 = extract_numbers(a1)
    nums2 = extract_numbers(a2)
    
    n_ratio = fuzz.ratio(n1.lower(), n2.lower())
    n_token_set = fuzz.token_set_ratio(n1.lower(), n2.lower())
    n_score = max(n_ratio, n_token_set)
    
    a_ratio = fuzz.ratio(a1.lower(), a2.lower())
    a_token_set = fuzz.token_set_ratio(a1.lower(), a2.lower())
    a_score = max(a_ratio, a_token_set)
    
    if n1.lower() == n2.lower():
        name_exact += 1
    if n_score >= 90:
        name_fuzz_90 += 1
    if n_score >= 70:
        name_fuzz_70 += 1
    if t1 and t2 and (t1 & t2):
        name_tok_overlap += 1
        
    if a1.lower() == a2.lower():
        addr_exact += 1
    if a_score >= 80:
        addr_fuzz_80 += 1
    if a_score >= 60:
        addr_fuzz_60 += 1
    if nums1 and nums2 and (nums1 & nums2):
        addr_num_match += 1
        
    if n_score >= 70 or a_score >= 70:
        either_name70_or_addr70 += 1
    if n_score >= 80 or a_score >= 80:
        either_name80_or_addr80 += 1
    else:
        if len(hard_cases) < 10:
            hard_cases.append((s1_id, mid, r1['country'], n1, n2, a1, a2, n_score, a_score))

N = len(valid_pairs)
print("\n" + "="*50)
print(f"Name exact match:                {name_exact/N*100:.2f}% ({name_exact}/{N})")
print(f"Name token overlap:              {name_tok_overlap/N*100:.2f}% ({name_tok_overlap}/{N})")
print(f"Name fuzz (ratio/set) >= 90:     {name_fuzz_90/N*100:.2f}% ({name_fuzz_90}/{N})")
print(f"Name fuzz (ratio/set) >= 70:     {name_fuzz_70/N*100:.2f}% ({name_fuzz_70}/{N})")
print(f"Address exact match:             {addr_exact/N*100:.2f}% ({addr_exact}/{N})")
print(f"Address num match:               {addr_num_match/N*100:.2f}% ({addr_num_match}/{N})")
print(f"Address fuzz >= 80:              {addr_fuzz_80/N*100:.2f}% ({addr_fuzz_80}/{N})")
print(f"Address fuzz >= 60:              {addr_fuzz_60/N*100:.2f}% ({addr_fuzz_60}/{N})")
print(f"EITHER name>=70 OR addr>=70:     {either_name70_or_addr70/N*100:.2f}% ({either_name70_or_addr70}/{N})")
print(f"EITHER name>=80 OR addr>=80:     {either_name80_or_addr80/N*100:.2f}% ({either_name80_or_addr80}/{N})")

print("\nHard cases (neither name>=80 nor addr>=80):")
for s1_id, mid, c, n1, n2, a1, a2, ns, as_ in hard_cases:
    print(f"[{c}] S1: '{n1}' | '{a1}'")
    print(f"     Match: '{n2}' | '{a2}' (NameScore: {ns:.1f}, AddrScore: {as_:.1f})")
