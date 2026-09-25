"""
Evaluation utilities for Business Entity Resolution.
Implements the challenge's macro-averaged F0.5 metric with proper singleton handling.
"""
import numpy as np
import pandas as pd


def compute_f05_per_entity(pred_matches: set, true_matches: set) -> float:
    """Compute F0.5 for a single Source 1 entity."""
    if len(true_matches) == 0 and len(pred_matches) == 0:
        return 1.0   # correct singleton
    if len(true_matches) == 0 and len(pred_matches) > 0:
        return 0.0   # false merge on singleton
    if len(pred_matches) == 0 and len(true_matches) > 0:
        return 0.0   # missed everything

    tp = len(pred_matches & true_matches)
    fp = len(pred_matches - true_matches)
    fn = len(true_matches - pred_matches)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

    if precision + recall == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)


def compute_macro_f05(predictions: dict, ground_truth: dict, s1_id_list=None) -> float:
    """
    Macro-averaged F0.5 across all S1 entities with proper singleton handling.
    """
    if s1_id_list is None:
        s1_id_list = list(ground_truth.keys())
    scores = []
    for s1_id in s1_id_list:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())
        scores.append(compute_f05_per_entity(pred_set, true_set))
    return float(np.mean(scores))


def parse_ground_truth(gt_df: pd.DataFrame) -> dict:
    """Parse ground truth DataFrame → {s1_id: set(matched_ids)}."""
    result = {}
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        val = row.get('matched_entity_ids', '')
        if pd.isna(val) or str(val).strip() == '' or str(val) == 'nan':
            result[s1_id] = set()
        else:
            result[s1_id] = set(str(val).split(','))
    return result


def blocking_recall(candidates: dict, ground_truth: dict) -> dict:
    """
    Measure blocking recall: what fraction of true matches appear in candidates.
    Returns dict with overall stats and per-entity details.
    """
    total_true = 0
    total_found = 0
    missed_entities = 0
    entity_recalls = []

    for s1_id, true_set in ground_truth.items():
        if len(true_set) == 0:
            entity_recalls.append(1.0)  # singleton: nothing to find
            continue
        cand_set = candidates.get(s1_id, set())
        found = len(true_set & cand_set)
        total_true += len(true_set)
        total_found += found
        recall = found / len(true_set)
        entity_recalls.append(recall)
        if recall < 1.0:
            missed_entities += 1

    return {
        'micro_recall': total_found / total_true if total_true > 0 else 1.0,
        'macro_recall': float(np.mean(entity_recalls)),
        'total_true_pairs': total_true,
        'total_found_in_candidates': total_found,
        'entities_with_missed_matches': missed_entities,
        'total_entities': len(ground_truth),
    }
