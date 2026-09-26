"""
Error Analysis Script for V2 Model & Blocking Pipeline
Systematically investigates:
1. Blocking Recall vs Classification False Negatives
2. Country-specific error distributions (US vs India)
3. Singleton vs Non-singleton error profiles
4. Concrete failure cases for both Precision and Recall
"""
import os
import sys
import gc
import time
import json
import heapq
import pickle
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from preprocess import normalize_address, STOP_WORDS
from features_v2 import FEATURE_COLS, clean_name_enhanced, extract_clean_numbers, compute_pair_features, char_ngrams
from evaluate import compute_f05_per_entity
from train_v2 import extract_keys, build_country_index, retrieve_candidates

DATA_DIR = os.path.join(SRC_DIR, '..', '..', '..', 'dataset')
CACHE_DIR = os.path.join(SRC_DIR, '..', '..', '..', 'cache')

def run_error_analysis(n_val=10000):
    print("=" * 80)
    print(f" ERROR ANALYSIS ON V2 PIPELINE (Validation sample: {n_val:,} S1 entities)")
    print("=" * 80)

    # 1. Load model checkpoint
    ckpt_path = os.path.join(CACHE_DIR, 'lgbm_model_v2.pkl')
    print(f"Loading checkpoint: {ckpt_path}")
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)
    model = ckpt['model']
    threshold = ckpt['threshold']
    print(f"Model best iteration: {model.best_iteration}, decision threshold: {threshold}")

    # 2. Load Ground Truth
    print("Loading Ground Truth...")
    gt_df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_ground_truth.tsv'), sep='\t', keep_default_na=False)
    gt_map = {}
    for sid, mids in zip(gt_df['source1_entity_id'], gt_df['matched_entity_ids']):
        gt_map[sid] = set(mids.split(',')) if mids else set()

    # 3. Load S1 data and replicate stratified split
    print("Loading S1 data...")
    s1_cols = ['entity_id', 'country', 'business_name', 'business_address']
    s1_all = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source1.tsv'), sep='\t', usecols=s1_cols, keep_default_na=False)

    np.random.seed(42)
    countries = ['US', 'India']
    val_s1_list = []
    for c in countries:
        s1_c = s1_all[s1_all['country'] == c].copy()
        s1_c = s1_c.sample(frac=1.0, random_state=42).reset_index(drop=True)
        c_ratio = len(s1_c) / len(s1_all)
        n_va = int(n_val * c_ratio)
        val_s1_list.append(s1_c.iloc[:n_va])
        print(f"  Validation entities for {c}: {n_va:,}")

    val_s1_df = pd.concat(val_s1_list, ignore_index=True)
    del s1_all, val_s1_list
    gc.collect()

    print("Normalizing S1 records...")
    val_s1_df['name_clean'] = val_s1_df['business_name'].apply(clean_name_enhanced)
    val_s1_df['addr_clean'] = val_s1_df['business_address'].apply(normalize_address)

    s1_dict = {
        r.entity_id: {
            'name': r.business_name,
            'addr': r.business_address,
            'name_clean': r.name_clean,
            'addr_clean': r.addr_clean,
            'country': r.country
        }
        for r in val_s1_df.itertuples(index=False)
    }

    # Tracking structures
    overall_stats = {
        'total_entities': len(val_s1_df),
        'singletons': 0,
        'non_singletons': 0,
        'total_true_matches': 0,
        'blocking_recalled_matches': 0,
        'blocking_missed_matches': 0,
        'classifier_tp': 0,
        'classifier_fp': 0,
        'classifier_fn': 0,
        'classifier_missed_matches': 0, # in candidates, but score < threshold
    }

    country_stats = {
        c: {
            'entities': 0,
            'singletons': 0,
            'true_matches': 0,
            'blocking_recalled': 0,
            'tp': 0, 'fp': 0, 'fn': 0,
            'f05_scores': []
        }
        for c in countries
    }

    blocking_miss_examples = []
    classifier_miss_examples = []
    false_positive_examples = []
    classifier_miss_scores = []

    # Process by country
    for country in countries:
        c_s1 = val_s1_df[val_s1_df['country'] == country]
        c_s1_ids = set(c_s1['entity_id'])
        print(f"\nProcessing {country} ({len(c_s1):,} S1 entities)...")

        # Load S2 and S3 for this country
        print(f"  Loading S2 and S3 for {country}...")
        s2_df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source2.tsv'), sep='\t',
                            usecols=['entity_id', 'country', 'business_name', 'business_address'], keep_default_na=False)
        s3_df = pd.read_csv(os.path.join(DATA_DIR, 'train', 'train_source3.tsv'), sep='\t',
                            usecols=['entity_id', 'country', 'business_name', 'business_address'], keep_default_na=False)
        s2_c = s2_df[s2_df['country'] == country].copy()
        s3_c = s3_df[s3_df['country'] == country].copy()
        del s2_df, s3_df
        gc.collect()

        s23_c = pd.concat([s2_c, s3_c], ignore_index=True)
        del s2_c, s3_c
        gc.collect()

        print(f"  Normalizing {len(s23_c):,} S2/S3 records...")
        s23_c['name_clean'] = s23_c['business_name'].apply(clean_name_enhanced)
        s23_c['addr_clean'] = s23_c['business_address'].apply(normalize_address)

        # Build S23 lookup dict for quick text retrieval
        s23_dict = {
            r.entity_id: {
                'name': r.business_name,
                'addr': r.business_address,
                'name_clean': r.name_clean,
                'addr_clean': r.addr_clean
            }
            for r in s23_c.itertuples(index=False)
        }

        # Build query key sets from validation S1
        s1_n_keys = set()
        s1_a_keys = set()
        s1_c_keys = set()
        s1_names = set()
        for r in c_s1.itertuples(index=False):
            if r.name_clean:
                s1_names.add(r.name_clean)
            nk, ak, ck = extract_keys(r.name_clean, r.addr_clean)
            s1_n_keys.update(nk)
            s1_a_keys.update(ak)
            s1_c_keys.update(ck)

        print(f"  Building index over {len(s23_c):,} records...")
        idx = build_country_index(s23_c, s1_n_keys, s1_a_keys, s1_c_keys, s1_names)

        # Retrieve candidates and extract features for each validation S1
        print(f"  Querying candidates and extracting features for {len(c_s1):,} S1 entities...", flush=True)
        t_cand_start = time.time()
        
        feature_rows = []
        pair_meta = []
        entity_cands = {}

        for i, r in enumerate(c_s1.itertuples(index=False)):
            sid = r.entity_id
            nc = r.name_clean
            ac = r.addr_clean
            cands = retrieve_candidates(nc, ac, idx)
            entity_cands[sid] = cands

            if not cands:
                continue

            w1_set = set(nc.split()) if nc else set()
            aw1_set = set(ac.split()) if ac else set()
            nums1 = extract_clean_numbers(ac)
            has_a1 = bool(ac and ac != 'nan')
            n1_ngrams = char_ngrams(nc, 3)
            a1_ngrams = char_ngrams(ac, 3) if has_a1 else set()

            for mid in sorted(cands):
                s23_info = s23_dict.get(mid)
                if not s23_info:
                    continue
                n2 = s23_info['name_clean']
                a2 = s23_info['addr_clean']
                feats = compute_pair_features(nc, ac, n2, a2, w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams)
                feature_rows.append(feats)
                pair_meta.append((sid, mid))

            if (i + 1) % 2000 == 0 or (i + 1) == len(c_s1):
                print(f"    Progress: {i+1:,}/{len(c_s1):,} entities | {len(feature_rows):,} pairs ({time.time()-t_cand_start:.1f}s)", flush=True)

        print(f"  Total candidate pairs generated: {len(feature_rows):,}")

        # Compute predictions
        print(f"  Predicting probabilities with LightGBM...", flush=True)
        X_cand = np.array(feature_rows, dtype=np.float32) if feature_rows else np.empty((0, len(FEATURE_COLS)))
        if len(X_cand) > 0:
            probs = model.predict(X_cand, num_threads=16)
        else:
            probs = np.array([])

        cand_prob_map = {}
        for (sid, mid), p in zip(pair_meta, probs):
            cand_prob_map.setdefault(sid, {})[mid] = float(p)

        del feature_rows, pair_meta, X_cand, probs
        gc.collect()

        # Evaluate per entity
        for r in c_s1.itertuples(index=False):
            sid = r.entity_id
            true_set = gt_map.get(sid, set())
            cands = entity_cands.get(sid, set())
            c_probs = cand_prob_map.get(sid, {})

            pred_set = {mid for mid, p in c_probs.items() if p >= threshold}

            score = compute_f05_per_entity(pred_set, true_set)
            country_stats[country]['f05_scores'].append(score)
            country_stats[country]['entities'] += 1

            is_singleton = (len(true_set) == 0)
            if is_singleton:
                overall_stats['singletons'] += 1
                country_stats[country]['singletons'] += 1
            else:
                overall_stats['non_singletons'] += 1

            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)

            country_stats[country]['tp'] += tp
            country_stats[country]['fp'] += fp
            country_stats[country]['fn'] += fn
            country_stats[country]['true_matches'] += len(true_set)

            overall_stats['total_true_matches'] += len(true_set)
            overall_stats['classifier_tp'] += tp
            overall_stats['classifier_fp'] += fp
            overall_stats['classifier_fn'] += fn

            # Analyze False Negatives: Blocking Miss vs Classifier Miss
            for true_m in true_set:
                if true_m in cands:
                    country_stats[country]['blocking_recalled'] += 1
                    overall_stats['blocking_recalled_matches'] += 1
                    p = c_probs.get(true_m, 0.0)
                    if p < threshold:
                        overall_stats['classifier_missed_matches'] += 1
                        classifier_miss_scores.append(p)
                        if len(classifier_miss_examples) < 15:
                            m_info = s23_dict.get(true_m, {})
                            classifier_miss_examples.append({
                                'sid': sid, 'mid': true_m, 'prob': round(p, 4),
                                's1_name': r.business_name, 's1_addr': r.business_address,
                                'cand_name': m_info.get('name', ''), 'cand_addr': m_info.get('addr', '')
                            })
                else:
                    overall_stats['blocking_missed_matches'] += 1
                    if len(blocking_miss_examples) < 15:
                        m_info = s23_dict.get(true_m, {})
                        blocking_miss_examples.append({
                            'sid': sid, 'mid': true_m,
                            's1_name': r.business_name, 's1_addr': r.business_address,
                            'true_name': m_info.get('name', ''), 'true_addr': m_info.get('addr', '')
                        })

            # Collect False Positives
            for fp_mid in (pred_set - true_set):
                if len(false_positive_examples) < 15:
                    m_info = s23_dict.get(fp_mid, {})
                    false_positive_examples.append({
                        'sid': sid, 'mid': fp_mid, 'prob': round(c_probs.get(fp_mid, 0.0), 4),
                        's1_name': r.business_name, 's1_addr': r.business_address,
                        'cand_name': m_info.get('name', ''), 'cand_addr': m_info.get('addr', '')
                    })

        del s23_c, s23_dict, idx, cand_prob_map, entity_cands
        gc.collect()

    # Compute macro F0.5
    all_f05 = country_stats['US']['f05_scores'] + country_stats['India']['f05_scores']
    overall_macro_f05 = float(np.mean(all_f05))
    us_macro_f05 = float(np.mean(country_stats['US']['f05_scores']))
    india_macro_f05 = float(np.mean(country_stats['India']['f05_scores']))

    total_true = overall_stats['total_true_matches']
    rec_matches = overall_stats['blocking_recalled_matches']
    miss_matches = overall_stats['blocking_missed_matches']
    blocking_recall = rec_matches / max(1, total_true)

    tp = overall_stats['classifier_tp']
    fp = overall_stats['classifier_fp']
    fn = overall_stats['classifier_fn']
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)

    print("\n" + "=" * 80)
    print(" ERROR ANALYSIS REPORT")
    print("=" * 80)
    print(f"Overall Macro F0.5:           {overall_macro_f05:.4f} ({overall_macro_f05*100:.2f}%)")
    print(f"  US Macro F0.5:              {us_macro_f05:.4f} ({us_macro_f05*100:.2f}%)")
    print(f"  India Macro F0.5:           {india_macro_f05:.4f} ({india_macro_f05*100:.2f}%)")
    print("-" * 80)
    print(f"Precision:                    {prec:.4f} (TP: {tp:,}, FP: {fp:,})")
    print(f"Recall:                       {rec:.4f} (TP: {tp:,}, FN: {fn:,})")
    print(f"Blocking Recall:              {blocking_recall:.4f} ({blocking_recall*100:.2f}%)")
    print(f"  True matches in candidates: {rec_matches:,} / {total_true:,}")
    print(f"  True matches MISSED at block:{miss_matches:,} / {total_true:,} ({miss_matches/max(1, total_true)*100:.2f}%)")
    print(f"False Negatives Breakdown (Total FN = {fn:,}):")
    print(f"  - Missed at Blocking:       {miss_matches:,} ({miss_matches/max(1, fn)*100:.1f}% of all FN)")
    print(f"  - Missed by Classifier:     {overall_stats['classifier_missed_matches']:,} ({overall_stats['classifier_missed_matches']/max(1, fn)*100:.1f}% of all FN)")
    
    if classifier_miss_scores:
        scores_arr = np.array(classifier_miss_scores)
        print(f"Classifier Miss Probabilities (threshold={threshold}):")
        print(f"  >= 0.80:                   {np.sum(scores_arr >= 0.80):,} (very close to threshold!)")
        print(f"  0.70 - 0.79:               {np.sum((scores_arr >= 0.70) & (scores_arr < 0.80)):,}")
        print(f"  0.50 - 0.69:               {np.sum((scores_arr >= 0.50) & (scores_arr < 0.70)):,}")
        print(f"  < 0.50:                    {np.sum(scores_arr < 0.50):,}")

    report = {
        'overall_macro_f05': overall_macro_f05,
        'us_macro_f05': us_macro_f05,
        'india_macro_f05': india_macro_f05,
        'blocking_recall': blocking_recall,
        'precision': prec,
        'recall': rec,
        'tp': tp, 'fp': fp, 'fn': fn,
        'blocking_missed_count': miss_matches,
        'classifier_missed_count': overall_stats['classifier_missed_matches'],
        'blocking_miss_examples': blocking_miss_examples,
        'classifier_miss_examples': classifier_miss_examples,
        'false_positive_examples': false_positive_examples
    }

    report_path = os.path.join(CACHE_DIR, 'error_analysis_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nDetailed report saved to: {report_path}")

if __name__ == '__main__':
    run_error_analysis(n_val=10000)
