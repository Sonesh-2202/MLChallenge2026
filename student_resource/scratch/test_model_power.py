import pandas as pd
import numpy as np
import sys
from rapidfuzz import fuzz
import unicodedata
import re

sys.stdout.reconfigure(encoding='utf-8')

print("Loading full GT and 50,000 S1...")
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')
gt_dict = {}
for _, r in gt.iterrows():
    m = r['matched_entity_ids']
    if pd.notna(m) and str(m).strip():
        gt_dict[r['source1_entity_id']] = set(str(m).split(','))
    else:
        gt_dict[r['source1_entity_id']] = set()

s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t', nrows=50000)
s1_sample_ids = set(s1['entity_id'])

# Collect true pairs for these 50k S1
true_pairs = []
for sid in s1_sample_ids:
    for mid in gt_dict.get(sid, set()):
        true_pairs.append((sid, mid))

print(f"Sample S1: {len(s1_sample_ids)}, True pairs: {len(true_pairs)}")

# Sample 5,000 true pairs to evaluate
np.random.seed(42)
sampled_true_indices = np.random.choice(len(true_pairs), size=min(4000, len(true_pairs)), replace=False)
sampled_true_pairs = [true_pairs[i] for i in sampled_true_indices]

needed_s2 = {m for _, m in sampled_true_pairs if m.startswith('S2-')}
needed_s3 = {m for _, m in sampled_true_pairs if m.startswith('S3-')}

print(f"Loading {len(needed_s2)} S2 and {len(needed_s3)} S3...")
s2_map = {}
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', chunksize=400000):
    hits = chunk[chunk['entity_id'].isin(needed_s2)]
    for _, r in hits.iterrows():
        s2_map[r['entity_id']] = r.to_dict()
    if len(s2_map) == len(needed_s2):
        break

s3_map = {}
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', chunksize=400000):
    hits = chunk[chunk['entity_id'].isin(needed_s3)]
    for _, r in hits.iterrows():
        s3_map[r['entity_id']] = r.to_dict()
    if len(s3_map) == len(needed_s3):
        break

s1_map = s1.set_index('entity_id').to_dict('index')

# Build features for positive pairs and random negative pairs
def compute_metrics(r1, r2):
    n1 = str(r1.get('business_name', '')).lower()
    n2 = str(r2.get('business_name', '')).lower()
    a1 = str(r1.get('business_address', '')).lower()
    a2 = str(r2.get('business_address', '')).lower()
    
    # Strip accents
    n1 = ''.join(c for c in unicodedata.normalize('NFD', n1) if unicodedata.category(c) != 'Mn')
    n2 = ''.join(c for c in unicodedata.normalize('NFD', n2) if unicodedata.category(c) != 'Mn')
    a1 = ''.join(c for c in unicodedata.normalize('NFD', a1) if unicodedata.category(c) != 'Mn')
    a2 = ''.join(c for c in unicodedata.normalize('NFD', a2) if unicodedata.category(c) != 'Mn')
    
    # Ratios
    n_ratio = fuzz.ratio(n1, n2) / 100.0
    n_tset = fuzz.token_set_ratio(n1, n2) / 100.0
    n_tsort = fuzz.token_sort_ratio(n1, n2) / 100.0
    n_part = fuzz.partial_ratio(n1, n2) / 100.0
    
    a_ratio = fuzz.ratio(a1, a2) / 100.0
    a_tset = fuzz.token_set_ratio(a1, a2) / 100.0
    a_tsort = fuzz.token_sort_ratio(a1, a2) / 100.0
    a_part = fuzz.partial_ratio(a1, a2) / 100.0
    
    # Number overlap in address
    nums1 = set(re.findall(r'\b\d+[-\w/]*\b', a1))
    nums2 = set(re.findall(r'\b\d+[-\w/]*\b', a2))
    num_jacc = len(nums1 & nums2) / max(1, len(nums1 | nums2)) if (nums1 or nums2) else 0.5
    
    return [n_ratio, n_tset, n_tsort, n_part, a_ratio, a_tset, a_tsort, a_part, num_jacc]

pos_features = []
for s1_id, mid in sampled_true_pairs:
    if mid in s2_map or mid in s3_map:
        r1 = s1_map[s1_id]
        r2 = s2_map.get(mid) or s3_map.get(mid)
        pos_features.append(compute_metrics(r1, r2))

# Negative pairs (random pairing)
neg_features = []
s23_pool = list(s2_map.values()) + list(s3_map.values())
for _ in range(len(pos_features) * 3):
    r1 = s1_map[np.random.choice(list(s1_map.keys()))]
    r2 = s23_pool[np.random.choice(len(s23_pool))]
    neg_features.append(compute_metrics(r1, r2))

print(f"Positives: {len(pos_features)}, Negatives: {len(neg_features)}")

import lightgbm as lgb
from sklearn.model_selection import cross_val_score

X = np.array(pos_features + neg_features)
y = np.array([1]*len(pos_features) + [0]*len(neg_features))

clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, max_depth=6, random_state=42)
scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
print(f"5-Fold CV F1: {scores.mean():.4f} (+/- {scores.std():.4f})")
clf.fit(X, y)
for name, imp in zip(['n_ratio', 'n_tset', 'n_tsort', 'n_part', 'a_ratio', 'a_tset', 'a_tsort', 'a_part', 'num_jacc'], clf.feature_importances_):
    print(f"  {name:12s}: {imp}")
