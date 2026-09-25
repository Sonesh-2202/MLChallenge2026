import pandas as pd
import numpy as np
import time
import unicodedata
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

def strip_accents(s):
    if not s or pd.isna(s):
        return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn').lower()

STOP_WORDS = {
    'the', 'and', 'for', 'of', 'in', 'on', 'at', 'to', 'a', 'an', 'is',
    'llc', 'inc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private', 'co', 'company',
    'llp', 'plc', 'sarl', 'gmbh', 'ag', 'dba', 'sas', 'sa', 'eurl', 'srl', 'sl', 'pty',
    'pllc', 'services', 'associates', 'street', 'road', 'st', 'rd', 'ave', 'avenue',
    'lane', 'ln', 'dr', 'drive', 'court', 'ct', 'city', 'state', 'unit', 'apt', 'apartment',
    'floor', 'fl', 'near', 'opp', 'behind', 'beside', 'null', 'nan'
}

def extract_keys(name, addr):
    keys = set()
    
    # 1. Clean name tokens
    name_clean = strip_accents(name)
    name_clean = re.sub(r'[^\w\s]', ' ', name_clean)
    name_tokens = [w for w in name_clean.split() if w not in STOP_WORDS]
    
    for t in name_tokens:
        if len(t) >= 3:
            keys.add('n:' + t)
        elif len(t) == 2 and t.isalnum():
            keys.add('n2:' + t)
            
    # 2. Name 2-gram pairs if multiple tokens
    if len(name_tokens) >= 2:
        for i in range(len(name_tokens)-1):
            keys.add('nb:' + name_tokens[i] + '_' + name_tokens[i+1])
            
    # 3. Address numbers and special codes
    addr_clean = strip_accents(addr)
    nums = re.findall(r'\b[a-z0-9]+(?:[-/][a-z0-9]+)+\b|\b\d{2,}\b', addr_clean)
    for num in nums:
        keys.add('num:' + num)
        
    # 4. Address distinctive words
    addr_words = [w for w in re.sub(r'[^\w\s]', ' ', addr_clean).split() if len(w) >= 4 and w not in STOP_WORDS]
    for w in addr_words:
        keys.add('a:' + w)
        
    # 5. Composite key: addr_num + first significant name word
    if nums and name_tokens:
        keys.add('na:' + name_tokens[0] + '_' + nums[0])
        
    return keys

print("Loading 100,000 S1 and 200,000 S2+S3 for US and India...")
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=50000)
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=50000)
s2 = pd.read_csv('dataset/train/train_source2.tsv', sep='\t', nrows=100000)
s3 = pd.read_csv('dataset/train/train_source3.tsv', sep='\t', nrows=100000)
s23 = pd.concat([s2, s3], ignore_index=True)

# Parse ground truth
gt_dict = {}
for _, r in gt.iterrows():
    s1_id = r['source1_entity_id']
    m = r['matched_entity_ids']
    if pd.notna(m) and str(m).strip():
        gt_dict[s1_id] = set(str(m).split(','))
    else:
        gt_dict[s1_id] = set()

# Filter GT to only entities whose matches could exist in our sample
available_s23_ids = set(s23['entity_id'])
eval_gt = {}
for s1_id, matches in gt_dict.items():
    present_matches = matches & available_s23_ids
    if present_matches:
        eval_gt[s1_id] = present_matches

print(f"Number of S1 with true matches present in sample: {len(eval_gt)}")
total_eval_pairs = sum(len(v) for v in eval_gt.values())
print(f"Total true pairs to find: {total_eval_pairs}")

# Build Inverted Index
t0 = time.time()
print("\nBuilding inverted index on S2+S3...")
index = {} # key -> list of s23_indices
# Group by country
countries = s23['country'].unique()
print(f"Countries: {countries}")

country_indices = {c: {} for c in countries}

s23_records = s23[['entity_id', 'business_name', 'business_address', 'country']].to_dict('records')
for idx, rec in enumerate(s23_records):
    c = rec['country']
    c_index = country_indices.setdefault(c, {})
    keys = extract_keys(rec['business_name'], rec['business_address'])
    for k in keys:
        if k not in c_index:
            c_index[k] = []
        c_index[k].append(idx)

print(f"Index built in {time.time()-t0:.2f}s!")

# Query S1 against Index
t1 = time.time()
print("\nQuerying S1 entities...")
s1_records = s1[['entity_id', 'business_name', 'business_address', 'country']].to_dict('records')

found_pairs = 0
total_candidates = 0

for rec in s1_records:
    s1_id = rec['entity_id']
    c = rec['country']
    c_index = country_indices.get(c, {})
    
    keys = extract_keys(rec['business_name'], rec['business_address'])
    cand_indices = set()
    for k in keys:
        if k in c_index:
            # avoid exploding on too common keys (e.g. if key has > 500 records)
            lst = c_index[k]
            if len(lst) <= 500:
                cand_indices.update(lst)
                
    total_candidates += len(cand_indices)
    
    if s1_id in eval_gt:
        true_set = eval_gt[s1_id]
        cand_ids = {s23_records[i]['entity_id'] for i in cand_indices}
        found_pairs += len(true_set & cand_ids)

query_time = time.time() - t1
print(f"Querying 50,000 S1 done in {query_time:.2f}s! ({len(s1_records)/query_time:.0f} S1/sec)")
print(f"Average candidates per S1: {total_candidates / len(s1_records):.1f}")
print(f"Recall on true matches: {found_pairs}/{total_eval_pairs} ({found_pairs/max(1, total_eval_pairs)*100:.2f}%)")
