import pandas as pd
import numpy as np
import time
import unicodedata
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("Testing candidate retrieval recall at various max_cands...")

STOP_WORDS = {
    'the', 'and', 'for', 'of', 'in', 'on', 'at', 'to', 'a', 'an', 'is',
    'llc', 'inc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private', 'co', 'company',
    'llp', 'plc', 'sarl', 'gmbh', 'ag', 'dba', 'sas', 'sa', 'eurl', 'srl', 'sl', 'pty',
    'pllc', 'pc', 'group', 'services', 'industries', 'associates', 'federation',
    'street', 'road', 'st', 'rd', 'ave', 'avenue', 'lane', 'ln', 'dr', 'drive',
    'court', 'ct', 'city', 'state', 'unit', 'apt', 'apartment', 'floor', 'fl',
    'near', 'opp', 'behind', 'beside', 'null', 'nan', 'com', 'org', 'net'
}

ADDR_MAP = {
    r'\brd\b': 'road', r'\bst\b': 'street', r'\bave\b': 'avenue',
    r'\bdr\b': 'drive', r'\bln\b': 'lane', r'\bct\b': 'court',
    r'\bblvd\b': 'boulevard', r'\bpkwy\b': 'parkway', r'\bhwy\b': 'highway',
    r'\bapt\b': 'apartment', r'\bste\b': 'suite', r'\bfl\b': 'floor',
    r'\bpo box\b': 'pobox',
    # State expansions (US)
    r'\bil\b': 'illinois', r'\bca\b': 'california', r'\btx\b': 'texas',
    r'\bny\b': 'new york', r'\bfl\b': 'florida', r'\boh\b': 'ohio',
    r'\bpa\b': 'pennsylvania', r'\bmi\b': 'michigan', r'\bnc\b': 'north carolina',
    r'\bga\b': 'georgia', r'\bva\b': 'virginia', r'\btn\b': 'tennessee',
    r'\bct\b': 'connecticut', r'\bmo\b': 'missouri', r'\baz\b': 'arizona',
    # State / City variants (India)
    r'\bmh\b': 'maharashtra', r'\bup\b': 'uttar pradesh', r'\bgj\b': 'gujarat',
    r'\brj\b': 'rajasthan', r'\btg\b': 'telangana', r'\btn\b': 'tamil nadu',
    r'\bwb\b': 'west bengal', r'\bka\b': 'karnataka', r'\bdl\b': 'delhi',
    r'\bcalcutta\b': 'kolkata', r'\bbengaluru\b': 'bangalore', r'\bbombay\b': 'mumbai',
    r'\bmadras\b': 'chennai', r'\bodisha\b': 'orissa',
}

def strip_accents(s):
    if not s or pd.isna(s):
        return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn').lower()

def clean_text(s, is_addr=False):
    s = strip_accents(s)
    if is_addr:
        for pat, repl in ADDR_MAP.items():
            s = re.sub(pat, repl, s)
    s = re.sub(r'[^\w\s]', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()

def extract_numbers(s):
    if not s or pd.isna(s):
        return set()
    s = str(s).lower()
    nums = re.findall(r'\b[a-z0-9]+(?:[-/][a-z0-9]+)+\b|\b\d+[a-z]?\b', s)
    return {n for n in nums if len(n) >= 2 or n.isdigit()}

def extract_keys(name_clean, addr_clean):
    keys = {}
    tokens = [w for w in name_clean.split() if w not in STOP_WORDS]
    for t in tokens:
        if len(t) >= 3:
            keys['n:' + t] = 5.0
        elif len(t) == 2 and t.isalnum():
            keys['n2:' + t] = 3.0
    for t in tokens:
        if len(t) >= 4:
            keys['ng:' + t[:4]] = 3.0
    nums = extract_numbers(addr_clean)
    for num in nums:
        keys['num:' + num] = 6.0
    addr_words = [w for w in addr_clean.split() if len(w) >= 4 and w not in STOP_WORDS]
    for w in addr_words:
        keys['a:' + w] = 1.0
    return keys

print("Loading 20,000 S1 and GT...")
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
gt_dict = {}
for _, r in gt_df.iterrows():
    m = r['matched_entity_ids']
    gt_dict[r['source1_entity_id']] = set(str(m).split(',')) if (pd.notna(m) and str(m).strip()) else set()

s1_all = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=20000)
s1_ids = set(s1_all['entity_id'])

needed_s23 = set()
for sid in s1_ids:
    needed_s23.update(gt_dict.get(sid, set()))

# Load S2 and S3 matches + distractors
s2_list, s3_list = [], []
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=500000):
    hits = chunk[chunk['entity_id'].isin(needed_s23)]
    s2_list.append(hits)
    if len(s2_list) == 1: s2_list.append(chunk.iloc[:30000])

for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=500000):
    hits = chunk[chunk['entity_id'].isin(needed_s23)]
    s3_list.append(hits)
    if len(s3_list) == 1: s3_list.append(chunk.iloc[:30000])

s23_pool = pd.concat(s2_list + s3_list, ignore_index=True).drop_duplicates(subset=['entity_id'])
print(f"S23 Pool: {len(s23_pool):,} records")

s1_all['name_clean'] = s1_all['business_name'].apply(lambda s: clean_text(s, False))
s1_all['addr_clean'] = s1_all['business_address'].apply(lambda s: clean_text(s, True))
s23_pool['name_clean'] = s23_pool['business_name'].apply(lambda s: clean_text(s, False))
s23_pool['addr_clean'] = s23_pool['business_address'].apply(lambda s: clean_text(s, True))

s23_dict = s23_pool.set_index('entity_id').to_dict('index')
index_by_country = {}
for eid, row in s23_dict.items():
    c = row['country']
    if c not in index_by_country: index_by_country[c] = {}
    idx = index_by_country[c]
    keys = extract_keys(row['name_clean'], row['addr_clean'])
    for k, w in keys.items():
        if k not in idx: idx[k] = []
        idx[k].append((eid, w))

total_trues = sum(len(gt_dict.get(sid, set())) for sid in s1_ids)

for max_k in [25, 35, 50, 70]:
    found = 0
    t0 = time.time()
    for _, row in s1_all.iterrows():
        sid = row['entity_id']
        c = row['country']
        idx = index_by_country.get(c, {})
        keys = extract_keys(row['name_clean'], row['addr_clean'])
        scores = {}
        for k, base_w in keys.items():
            if k in idx:
                postings = idx[k]
                idf = base_w / np.log1p(len(postings))
                for mid, mid_w in postings:
                    scores[mid] = scores.get(mid, 0.0) + (idf * mid_w)
        if scores:
            top_k = {mid for mid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_k]}
        else:
            top_k = set()
        found += len(gt_dict.get(sid, set()) & top_k)
    elapsed = time.time() - t0
    print(f"Top-{max_k:2d}: Recall = {found:,}/{total_trues:,} ({found/total_trues*100:.2f}%)  [{elapsed:.1f}s]")
