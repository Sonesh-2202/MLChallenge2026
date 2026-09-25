import pandas as pd
import numpy as np
import time
import unicodedata
import re
import sys
from rapidfuzz import fuzz
import lightgbm as lgb
from sklearn.model_selection import train_test_split

sys.stdout.reconfigure(encoding='utf-8')

print("="*70)
print("TESTING SELF-CONTAINED HIGH-RECALL PIPELINE")
print("="*70)

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

def extract_keys(name_clean, addr_clean):
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

print("1. Loading ground truth...")
gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
gt_dict = {}
for _, r in gt_df.iterrows():
    m = r['matched_entity_ids']
    gt_dict[r['source1_entity_id']] = set(str(m).split(',')) if (pd.notna(m) and str(m).strip()) else set()

print(f"Total GT entries: {len(gt_dict):,}")

# Load 30,000 S1 records
print("\n2. Loading 30,000 S1 records...")
s1_all = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=30000)
s1_ids = set(s1_all['entity_id'])

needed_s23 = set()
for sid in s1_ids:
    needed_s23.update(gt_dict.get(sid, set()))

print(f"30,000 S1 require {len(needed_s23):,} true matching S2/S3 records.")

# Load matching S2 and S3 + distractors
print("\n3. Loading S2 and S3 records...")
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
print(f"Candidate pool S23 size: {len(s23_pool):,} records")

# Normalizing
print("\n4. Normalizing text...")
t0 = time.time()
s1_all['name_clean'] = s1_all['business_name'].apply(lambda s: clean_text(s, False))
s1_all['addr_clean'] = s1_all['business_address'].apply(lambda s: clean_text(s, True))
s23_pool['name_clean'] = s23_pool['business_name'].apply(lambda s: clean_text(s, False))
s23_pool['addr_clean'] = s23_pool['business_address'].apply(lambda s: clean_text(s, True))
print(f"Normalization done in {time.time()-t0:.1f}s")

# Train / Validation Split (80/20)
train_s1, val_s1 = train_test_split(s1_all, test_size=0.2, random_state=42, stratify=s1_all['country'])
print(f"Train S1: {len(train_s1):,}, Val S1: {len(val_s1):,}")

# 5. Build Index on S23 by country
print("\n5. Building index on S23...")
t0 = time.time()
s23_dict = s23_pool.set_index('entity_id').to_dict('index')
index_by_country = {}

for eid, row in s23_dict.items():
    c = row['country']
    if c not in index_by_country:
        index_by_country[c] = {'name': {}, 'num': {}, 'addr': {}}
    idx = index_by_country[c]
    nk, numk, ak = extract_keys(row['name_clean'], row['addr_clean'])
    for k in nk:
        idx['name'].setdefault(k, []).append(eid)
    for k in numk:
        idx['num'].setdefault(k, []).append(eid)
    for k in ak:
        idx['addr'].setdefault(k, []).append(eid)

print(f"Index built in {time.time()-t0:.2f}s")

# 6. Candidate Retrieval
def retrieve_candidates(s1_df, index_dict, max_k=40):
    candidates = {}
    for _, row in s1_df.iterrows():
        sid = row['entity_id']
        c = row['country']
        idx = index_dict.get(c, {'name': {}, 'num': {}, 'addr': {}})
        nk, numk, ak = extract_keys(row['name_clean'], row['addr_clean'])
        
        scores = {}
        for k in nk:
            if k in idx['name']:
                postings = idx['name'][k]
                w = 10.0 / np.log1p(len(postings))
                for mid in postings:
                    scores[mid] = scores.get(mid, 0.0) + w
                    
        for k in numk:
            if k in idx['num']:
                postings = idx['num'][k]
                w = 8.0 / np.log1p(len(postings))
                for mid in postings:
                    scores[mid] = scores.get(mid, 0.0) + w
                    
        for k in ak:
            if k in idx['addr']:
                postings = idx['addr'][k]
                if len(postings) < 1500:
                    w = 1.0 / np.log1p(len(postings))
                    for mid in postings:
                        scores[mid] = scores.get(mid, 0.0) + w
                        
        if scores:
            top_set = {mid for mid, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_k]}
        else:
            top_set = set()
        candidates[sid] = top_set
    return candidates

print("\n6. Running candidate retrieval...")
t0 = time.time()
train_cands = retrieve_candidates(train_s1, index_by_country, max_k=40)
val_cands = retrieve_candidates(val_s1, index_by_country, max_k=40)
print(f"Blocking done in {time.time()-t0:.2f}s")

def eval_blocking(cands, s1_df):
    total_true = 0
    found_true = 0
    for sid in s1_df['entity_id']:
        trues = gt_dict.get(sid, set())
        total_true += len(trues)
        found_true += len(trues & cands.get(sid, set()))
    recall = found_true / max(1, total_true)
    avg_cands = sum(len(v) for v in cands.values()) / max(1, len(cands))
    print(f"  Blocking Recall: {found_true:,} / {total_true:,} ({recall*100:.2f}%)  |  Avg cands/S1: {avg_cands:.1f}")
    return recall

print("Train Blocking Evaluation:")
eval_blocking(train_cands, train_s1)
print("Val Blocking Evaluation:")
eval_blocking(val_cands, val_s1)

# 7. Fast Feature Extraction
def extract_pair_features(s1_records, cands_dict):
    rows = []
    for sid, cand_set in cands_dict.items():
        if not cand_set:
            continue
        r1 = s1_records[sid]
        n1 = r1['name_clean']
        a1 = r1['addr_clean']
        nums1 = extract_numbers(a1)
        w1_set = set(n1.split())
        aw1_set = set(a1.split())
        
        for mid in cand_set:
            r2 = s23_dict[mid]
            n2 = r2['name_clean']
            a2 = r2['addr_clean']
            
            n_ratio = fuzz.ratio(n1, n2) / 100.0
            n_tset = fuzz.token_set_ratio(n1, n2) / 100.0
            n_tsort = fuzz.token_sort_ratio(n1, n2) / 100.0
            n_part = fuzz.partial_ratio(n1, n2) / 100.0
            
            w2_set = set(n2.split())
            n_jacc = len(w1_set & w2_set) / max(1, len(w1_set | w2_set)) if (w1_set or w2_set) else 0.0
            
            has_a1 = bool(a1 and a1 != 'nan')
            has_a2 = bool(a2 and a2 != 'nan')
            both_addr = 1.0 if (has_a1 and has_a2) else 0.0
            either_missing = 1.0 if (not has_a1 or not has_a2) else 0.0
            
            if both_addr:
                a_ratio = fuzz.ratio(a1, a2) / 100.0
                a_tset = fuzz.token_set_ratio(a1, a2) / 100.0
                a_tsort = fuzz.token_sort_ratio(a1, a2) / 100.0
                a_part = fuzz.partial_ratio(a1, a2) / 100.0
                aw2_set = set(a2.split())
                a_jacc = len(aw1_set & aw2_set) / max(1, len(aw1_set | aw2_set)) if (aw1_set or aw2_set) else 0.0
            else:
                a_ratio = a_tset = a_tsort = a_part = a_jacc = 0.0
                
            nums2 = extract_numbers(a2)
            num_match = 1.0 if (nums1 and nums2 and (nums1 & nums2)) else 0.0
            num_jacc = len(nums1 & nums2) / max(1, len(nums1 | nums2)) if (nums1 or nums2) else 0.0
            
            eff_a_score = a_tset if both_addr else n_tset
            harm = (2.0 * n_tset * eff_a_score) / (n_tset + eff_a_score + 1e-5)
            
            is_match = 1 if mid in gt_dict.get(sid, set()) else 0
            
            rows.append({
                's1_id': sid, 's23_id': mid,
                'n_ratio': n_ratio, 'n_tset': n_tset, 'n_tsort': n_tsort, 'n_part': n_part, 'n_jacc': n_jacc,
                'a_ratio': a_ratio, 'a_tset': a_tset, 'a_tsort': a_tsort, 'a_part': a_part, 'a_jacc': a_jacc,
                'num_match': num_match, 'num_jacc': num_jacc,
                'both_addr': both_addr, 'either_missing': either_missing,
                'harm': harm,
                'exact_name': 1.0 if (n1 and n1 == n2) else 0.0,
                'exact_addr': 1.0 if (a1 and a1 == a2) else 0.0,
                'label': is_match
            })
    return pd.DataFrame(rows)

print("\n7. Extracting features...")
t0 = time.time()
s1_dict_all = s1_all.set_index('entity_id').to_dict('index')
train_feat = extract_pair_features(s1_dict_all, train_cands)
val_feat = extract_pair_features(s1_dict_all, val_cands)
print(f"Features built: Train={len(train_feat):,} pairs, Val={len(val_feat):,} pairs in {time.time()-t0:.1f}s")

# 8. Train LightGBM Model
print("\n8. Training LightGBM Model...")
feat_cols = [
    'n_ratio', 'n_tset', 'n_tsort', 'n_part', 'n_jacc',
    'a_ratio', 'a_tset', 'a_tsort', 'a_part', 'a_jacc',
    'num_match', 'num_jacc', 'both_addr', 'either_missing',
    'harm', 'exact_name', 'exact_addr'
]
X_tr = train_feat[feat_cols].values
y_tr = train_feat['label'].values

clf = lgb.LGBMClassifier(
    n_estimators=300,
    learning_rate=0.06,
    num_leaves=127,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
clf.fit(X_tr, y_tr)

# Predict Probabilities on Val
val_probs = clf.predict_proba(val_feat[feat_cols].values)[:, 1]
val_feat['prob'] = val_probs

# 9. Evaluate Macro F0.5 Metric with Singleton Handling
def compute_macro_f05(val_predictions, val_s1_ids):
    scores = []
    for sid in val_s1_ids:
        preds = val_predictions.get(sid, set())
        trues = gt_dict.get(sid, set())
        
        if len(trues) == 0 and len(preds) == 0:
            scores.append(1.0)
        elif len(trues) == 0 and len(preds) > 0:
            scores.append(0.0)
        elif len(trues) > 0 and len(preds) == 0:
            scores.append(0.0)
        else:
            tp = len(preds & trues)
            fp = len(preds - trues)
            fn = len(trues - preds)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            if (0.25 * prec + rec) > 0:
                f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
            else:
                f05 = 0.0
            scores.append(f05)
    return float(np.mean(scores))

print("\n9. Optimizing Threshold for Macro F0.5...")
val_s1_id_list = list(val_s1['entity_id'])

best_th = 0.5
best_score = 0.0

for th in np.arange(0.30, 0.90, 0.05):
    preds = {sid: set() for sid in val_s1_id_list}
    above = val_feat[val_feat['prob'] >= th]
    for sid, mid in zip(above['s1_id'], above['s23_id']):
        preds[sid].add(mid)
        
    score = compute_macro_f05(preds, val_s1_id_list)
    print(f"  Threshold {th:.2f} -> Macro F0.5 = {score:.4f}")
    if score > best_score:
        best_score = score
        best_th = th

print("="*70)
print(f"FINAL RESULT: Best Threshold = {best_th:.2f}, BEST VALIDATION MACRO F0.5 = {best_score:.4f} ({best_score*100:.2f}%)")
print("="*70)
