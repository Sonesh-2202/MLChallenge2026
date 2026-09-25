"""
High-Speed Feature Engineering for Candidate Pairs using C-Accelerated RapidFuzz.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
import time
from preprocess import extract_numbers

FEATURE_COLS = [
    'n_ratio', 'n_tset', 'n_tsort', 'n_part', 'n_jacc',
    'a_ratio', 'a_tset', 'a_tsort', 'a_part', 'a_jacc',
    'num_match', 'num_jacc', 'both_addr', 'either_missing',
    'harm', 'exact_name', 'exact_addr'
]


def extract_features_for_candidates(candidates, s1_dict, s23_dict, ground_truth=None, chunk_size=500000):
    """
    Compute features for all candidate pairs in (s1, s23).
    candidates: dict {s1_id: set(candidate_s23_ids)}
    s1_dict: dict {entity_id: {'name_clean': str, 'addr_clean': str, ...}}
    s23_dict: dict {entity_id: {'name_clean': str, 'addr_clean': str, ...}}
    ground_truth: optional dict {s1_id: set(true_match_ids)}
    """
    total_pairs = sum(len(v) for v in candidates.values())
    print(f"  Extracting features for {total_pairs:,} candidate pairs ...", flush=True)
    t0 = time.time()

    rows = []
    has_gt = ground_truth is not None

    for sid, cand_set in candidates.items():
        if not cand_set:
            continue
        r1 = s1_dict.get(sid)
        if r1 is None:
            continue
        n1 = r1.get('name_clean', '')
        a1 = r1.get('addr_clean', '')
        nums1 = extract_numbers(a1)
        w1_set = set(n1.split()) if n1 else set()
        aw1_set = set(a1.split()) if a1 else set()
        has_a1 = bool(a1 and a1 != 'nan')

        true_set = ground_truth.get(sid, set()) if has_gt else None

        for mid in cand_set:
            r2 = s23_dict.get(mid)
            if r2 is None:
                continue
            n2 = r2.get('name_clean', '')
            a2 = r2.get('addr_clean', '')

            # Name metrics
            n_ratio = fuzz.ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0
            n_tset = fuzz.token_set_ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0
            n_tsort = fuzz.token_sort_ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0
            n_part = fuzz.partial_ratio(n1, n2) / 100.0 if (n1 and n2) else 0.0

            w2_set = set(n2.split()) if n2 else set()
            n_jacc = len(w1_set & w2_set) / max(1, len(w1_set | w2_set)) if (w1_set or w2_set) else 0.0

            # Address metrics
            has_a2 = bool(a2 and a2 != 'nan')
            both_addr = 1.0 if (has_a1 and has_a2) else 0.0

            if both_addr:
                a_ratio = fuzz.ratio(a1, a2) / 100.0
                a_tset = fuzz.token_set_ratio(a1, a2) / 100.0
                a_tsort = fuzz.token_sort_ratio(a1, a2) / 100.0
                a_part = fuzz.partial_ratio(a1, a2) / 100.0
                aw2_set = set(a2.split()) if a2 else set()
                a_jacc = len(aw1_set & aw2_set) / max(1, len(aw1_set | aw2_set)) if (aw1_set or aw2_set) else 0.0
            else:
                a_ratio = a_tset = a_tsort = a_part = a_jacc = 0.0

            nums2 = extract_numbers(a2)
            num_match = 1.0 if (nums1 and nums2 and (nums1 & nums2)) else 0.0
            num_jacc = len(nums1 & nums2) / max(1, len(nums1 | nums2)) if (nums1 or nums2) else 0.0

            eff_a_score = a_tset if both_addr else n_tset
            harm = (2.0 * n_tset * eff_a_score) / (n_tset + eff_a_score + 1e-5)

            rec = {
                's1_id': sid, 's23_id': mid,
                'n_ratio': n_ratio, 'n_tset': n_tset, 'n_tsort': n_tsort, 'n_part': n_part, 'n_jacc': n_jacc,
                'a_ratio': a_ratio, 'a_tset': a_tset, 'a_tsort': a_tsort, 'a_part': a_part, 'a_jacc': a_jacc,
                'num_match': num_match, 'num_jacc': num_jacc,
                'both_addr': both_addr, 'either_missing': 1.0 - both_addr,
                'harm': harm,
                'exact_name': 1.0 if (n1 and n1 == n2) else 0.0,
                'exact_addr': 1.0 if (a1 and a1 == a2) else 0.0,
            }
            if has_gt:
                rec['label'] = 1 if mid in true_set else 0

            rows.append(rec)

    df = pd.DataFrame(rows)
    print(f"  Feature matrix built: {df.shape} in {time.time()-t0:.1f}s", flush=True)
    return df
