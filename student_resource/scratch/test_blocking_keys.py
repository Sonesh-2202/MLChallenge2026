import pandas as pd
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

needed_s2 = {m for _, m in gt_pairs if m.startswith('S2-')}
needed_s3 = {m for _, m in gt_pairs if m.startswith('S3-')}

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

valid_pairs = [p for p in gt_pairs if p[1] in s2_map or p[1] in s3_map]
print(f"Testing blocking key coverage on {len(valid_pairs)} pairs...")

LEGAL_SUFFIXES = {
    'llc', 'inc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private', 'co', 'company',
    'llp', 'plc', 'sarl', 'gmbh', 'ag', 'dba', 'd.b.a.', 'sas', 'sa', 'eurl', 'srl', 'sl', 'pty',
    'pllc', 'pc', 'group', 'services', 'industries', 'associates', 'federation', 'and', 'the', 'for', 'of'
}

def get_name_tokens(s):
    if not s or pd.isna(s):
        return set()
    s = re.sub(r'[^\w\s]', ' ', str(s).lower())
    return {w for w in s.split() if len(w) >= 3 and w not in LEGAL_SUFFIXES}

def get_addr_numbers(s):
    if not s or pd.isna(s):
        return set()
    nums = re.findall(r'\b\d+[-\w/]*\b', str(s).lower())
    return {n for n in nums if len(n) >= 2}

def get_addr_tokens(s):
    if not s or pd.isna(s):
        return set()
    s = re.sub(r'[^\w\s]', ' ', str(s).lower())
    common = {'street', 'road', 'st', 'rd', 'ave', 'avenue', 'lane', 'ln', 'dr', 'drive', 'court', 'ct', 'fl', 'floor', 'near', 'opp', 'behind', 'beside', 'sector', 'block', 'plot', 'shop', 'no', 'unit', 'apartment', 'apt', 'city', 'state', 'india', 'us', 'null', 'nan'}
    return {w for w in s.split() if len(w) >= 4 and w not in common}

hits_name_token = 0
hits_addr_num = 0
hits_addr_token = 0
hits_any = 0
misses = []

for s1_id, mid in valid_pairs:
    r1 = s1_dict[s1_id]
    r2 = s2_map.get(mid) or s3_map.get(mid)
    
    nt1 = get_name_tokens(r1.get('business_name'))
    nt2 = get_name_tokens(r2.get('business_name'))
    
    an1 = get_addr_numbers(r1.get('business_address'))
    an2 = get_addr_numbers(r2.get('business_address'))
    
    at1 = get_addr_tokens(r1.get('business_address'))
    at2 = get_addr_tokens(r2.get('business_address'))
    
    h_nt = bool(nt1 & nt2)
    h_an = bool(an1 & an2)
    h_at = bool(at1 & at2)
    
    if h_nt: hits_name_token += 1
    if h_an: hits_addr_num += 1
    if h_at: hits_addr_token += 1
    
    if h_nt or h_an or h_at:
        hits_any += 1
    else:
        misses.append((s1_id, mid, r1['country'], r1.get('business_name'), r2.get('business_name'), r1.get('business_address'), r2.get('business_address')))

N = len(valid_pairs)
print(f"Name token match:    {hits_name_token/N*100:.2f}% ({hits_name_token}/{N})")
print(f"Addr number match:   {hits_addr_num/N*100:.2f}% ({hits_addr_num}/{N})")
print(f"Addr token match:    {hits_addr_token/N*100:.2f}% ({hits_addr_token}/{N})")
print(f"ANY of the 3:        {hits_any/N*100:.2f}% ({hits_any}/{N})")
print(f"Misses count:        {len(misses)}/{N}")
print("\nSample misses:")
for m in misses[:10]:
    print(f"[{m[2]}] S1: '{m[3]}' | '{m[5]}'")
    print(f"     Match: '{m[4]}' | '{m[6]}'")
