import pandas as pd
import numpy as np
import sys

sys.stdout.reconfigure(encoding='utf-8')

import os
sys.path.insert(0, os.path.dirname(__file__))
from test_union_pipeline import val_s1, val_feat, best_th, gt_dict, s1_dict_all, s23_dict, compute_macro_f05

val_s1_id_list = list(val_s1['entity_id'])
preds = {sid: set() for sid in val_s1_id_list}
above = val_feat[val_feat['prob'] >= best_th]
for sid, mid in zip(above['s1_id'], above['s23_id']):
    preds[sid].add(mid)

fps = []
fns = []

for sid in val_s1_id_list:
    trues = gt_dict.get(sid, set())
    pred_set = preds.get(sid, set())
    
    fp_set = pred_set - trues
    fn_set = trues - pred_set
    
    if fp_set:
        fps.append((sid, fp_set, trues))
    if fn_set:
        fns.append((sid, fn_set, pred_set))

print(f"Total S1 validation entities: {len(val_s1_id_list):,}")
print(f"Entities with False Positives: {len(fps):,} ({len(fps)/len(val_s1_id_list)*100:.2f}%)")
print(f"Entities with False Negatives: {len(fns):,} ({len(fns)/len(val_s1_id_list)*100:.2f}%)")

print("\n--- SAMPLE FALSE POSITIVES (Wrong Merges - Heavily Penalized!) ---")
for sid, fp_set, trues in fps[:5]:
    r1 = s1_dict_all[sid]
    print("="*70)
    print(f"S1:  Name='{r1['business_name']}', Addr='{r1['business_address']}'")
    for fpid in list(fp_set)[:2]:
        r2 = s23_dict[fpid]
        p = val_feat[(val_feat['s1_id']==sid) & (val_feat['s23_id']==fpid)]['prob'].values[0]
        print(f"FP:  Name='{r2['business_name']}', Addr='{r2['business_address']}' (prob={p:.4f})")
    print(f"True matches were: {trues}")

print("\n--- SAMPLE FALSE NEGATIVES (Missed Matches) ---")
for sid, fn_set, pred_set in fns[:5]:
    r1 = s1_dict_all[sid]
    print("="*70)
    print(f"S1:  Name='{r1['business_name']}', Addr='{r1['business_address']}'")
    for fnid in list(fn_set)[:2]:
        if fnid in s23_dict:
            r2 = s23_dict[fnid]
            sub = val_feat[(val_feat['s1_id']==sid) & (val_feat['s23_id']==fnid)]
            p = sub['prob'].values[0] if len(sub) > 0 else -1.0
            print(f"FN:  Name='{r2['business_name']}', Addr='{r2['business_address']}' (prob={p:.4f})")
        else:
            print(f"FN:  {fnid} (Not in S23 pool)")
