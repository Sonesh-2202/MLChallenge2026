import pandas as pd
import numpy as np
import time
import unicodedata
import re
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("Testing Two-Pass Union Blocking (Name Pass + Address Pass)...")

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
    # US States
    r'\bil\b': 'illinois', r'\bca\b': 'california', r'\btx\b': 'texas',
    r'\bny\b': 'new york', r'\bfl\b': 'florida', r'\boh\b': 'ohio',
    r'\bpa\b': 'pennsylvania', r'\bmi\b': 'michigan', r'\bnc\b': 'north carolina',
    r'\bga\b': 'georgia', r'\bva\b': 'virginia', r'\btn\b': 'tennessee',
    r'\bct\b': 'connecticut', r'\bmo\b': 'missouri', r'\baz\b': 'arizona',
    # India States & Cities
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

def extract_name_keys(name_clean):
    keys = set()
    tokens = [w for w in name_clean.split() if w not in STOP_WORDS]
    for t in tokens:
        if len(t) >= 3:
            keys.add('n:' + t)
        elif len(t) == 2 and t.isalnum():
            keys.add('n2:' + t)
        if len(t) >= 4:
            keys.add('ng:' + t[:4])
    return keys

def extract_addr_keys(addr_clean):
    keys = set()
    nums = extract_numbers(addr_clean)
    for num in nums:
        keys.add('num:' + num)
    addr_words = [w for w in addr_clean.split() if len(w) >= 4 and w not in STOP_WORDS]
    for w in addr_words:
        keys.add('a:' + w)
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

# Build two separate indices by country: name_index and addr_index
name_index = {}
addr_index = {}

for eid, row in s23_dict.items():
    c = row['country']
    if c not in name_index: name_index[c] = {}
    if c not in addr_index: addr_index[c] = {}
    
    n_keys = extract_name_keys(row['name_clean'])
    for k in n_keys:
        name_index[c].setdefault(k, []).append(eid)
        
    a_keys = extract_addr_keys(row['addr_clean'])
    for k in a_keys:
        addr_index[c].setdefault(k, []).append(eid)

print("Separate indices built. Querying...")
total_trues = sum(len(gt_dict.get(sid, set())) for sid in s1_ids)

for name_k, addr_k in [(20, 20), (25, 25), (30, 30)]:
    t0 = time.time()
    found = 0
    total_cands = 0
    for _, row in s1_all.iterrows():
        sid = row['entity_id']
        c = row['country']
        n_idx = name_index.get(c, {})
        a_idx = addr_index.get(c, {})
        
        # Name pass
        n_keys = extract_name_keys(row['name_clean'])
        n_scores = {}
        for k in n_keys:
            if k in n_idx:
                postings = n_idx[k]
                w = 1.0 / np.log1p(len(postings))
                for mid in postings:
                    n_scores[mid] = n_scores.get(mid, 0.0) + w
        top_name = {mid for mid, _ in sorted(n_scores.items(), key=lambda x: x[1], reverse=True)[:name_k]}
        
        # Addr pass
        a_keys = extract_addr_keys(row['addr_clean'])
        a_scores = {}
        for k in a_keys:
            if k in a_idx:
                postings = a_idx[k]
                w = 1.0 / np.log1p(len(postings))
                for mid in postings:
                    a_scores[mid] = a_scores.get(mid, 0.0) + w
        top_addr = {mid for mid, _ in sorted(a_scores.items(), key=lambda x: x[1], reverse=True)[:addr_k]}
        
        # Union
        union_cands = top_name | top_addr
        total_cands += len(union_cands)
        found += len(gt_dict.get(sid, set()) & union_cands)
        
    elapsed = time.time() - t0
    recall = found / total_trues
    avg_cands = total_cands / len(s1_all)
    print(f"Name Top-{name_k} + Addr Top-{addr_k} (Avg {avg_cands:.1f} cands): Recall = {found:,}/{total_trues:,} ({recall*100:.2f}%)  [{elapsed:.1f}s]")
