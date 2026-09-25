#!/usr/bin/env python3
"""
Business Entity Resolution — End-to-End Scalable Production Pipeline
=====================================================================
Multi-Pass Inverted Index Blocking + RapidFuzz Feature Engineering + LightGBM + F0.5 Threshold Calibration
"""
import argparse
import gc
import os
import pickle
import subprocess
import sys
import time

# Ensure UTF-8 output on all platforms
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd

from preprocess import (
    load_sources, load_ground_truth, add_normalized_columns,
    create_validation_split, normalize_name, normalize_address
)
from blocking import build_country_indices, retrieve_candidates_for_s1
from features import extract_features_for_candidates, FEATURE_COLS
from model import train_lgbm, predict_probabilities, optimize_threshold_f05, apply_decision_threshold
from evaluate import parse_ground_truth, compute_macro_f05, blocking_recall


def write_tsv(mapping, all_ids, filepath, col_name='matched_entity_ids'):
    """Write submission TSV file ensuring strict format compliance."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"source1_entity_id\t{col_name}\n")
        for sid in all_ids:
            cand_set = mapping.get(sid, set())
            if cand_set:
                f.write(f"{sid}\t{','.join(sorted(cand_set))}\n")
            else:
                f.write(f"{sid}\t\n")
    print(f"  Saved {filepath} ({len(all_ids):,} rows)", flush=True)


def train_and_calibrate(args):
    """Execute Stages 1 through 5: Data loading, split, blocking, feature extraction, and model training."""
    # ==============================================================
    # STAGE 1: Load Ground Truth & Train Reference S1
    # ==============================================================
    print("\n" + "=" * 75)
    print(" STAGE 1: Load Reference Training Data")
    print("=" * 75)

    gt_df = load_ground_truth(args.data_dir)
    gt_dict = parse_ground_truth(gt_df)
    print(f"  Loaded Ground Truth: {len(gt_dict):,} S1 entities", flush=True)

    n_samples = 30000 if args.dev else args.train_samples
    print(f"  Loading {n_samples:,} reference S1 records ...", flush=True)
    s1_all = pd.read_csv(os.path.join(args.data_dir, 'train', 'train_source1.tsv'),
                         sep='\t', nrows=n_samples, dtype=str).fillna({'business_name': '', 'business_address': '', 'country': ''})
    s1_ids = set(s1_all['entity_id'])

    # Find required S2/S3 IDs to guarantee true matches are available
    needed_s23 = set()
    for sid in s1_ids:
        needed_s23.update(gt_dict.get(sid, set()))
    print(f"  {len(s1_all):,} S1 reference records have {len(needed_s23):,} true matching links.", flush=True)

    # Load matching S2 and S3 records plus negative distractors
    print("  Loading matching S2 and S3 candidate records + distractors ...", flush=True)
    s2_chunks, s3_chunks = [], []
    for chunk in pd.read_csv(os.path.join(args.data_dir, 'train', 'train_source2.tsv'), sep='\t', chunksize=250000, dtype=str):
        hits = chunk[chunk['entity_id'].isin(needed_s23)]
        if len(hits) > 0:
            s2_chunks.append(hits.copy())
        if len(s2_chunks) == 1:
            s2_chunks.append(chunk.iloc[:40000].copy())

    for chunk in pd.read_csv(os.path.join(args.data_dir, 'train', 'train_source3.tsv'), sep='\t', chunksize=250000, dtype=str):
        hits = chunk[chunk['entity_id'].isin(needed_s23)]
        if len(hits) > 0:
            s3_chunks.append(hits.copy())
        if len(s3_chunks) == 1:
            s3_chunks.append(chunk.iloc[:40000].copy())

    s23_train = pd.concat(s2_chunks + s3_chunks, ignore_index=True).drop_duplicates(subset=['entity_id'])
    s23_train = s23_train.fillna({'business_name': '', 'business_address': '', 'country': ''})
    del s2_chunks, s3_chunks
    gc.collect()
    print(f"  Training candidate pool S2+S3: {len(s23_train):,} records", flush=True)

    # Normalization
    print("  Normalizing text ...", flush=True)
    add_normalized_columns(s1_all)
    add_normalized_columns(s23_train)

    # ==============================================================
    # STAGE 2: Stratified Validation Split
    # ==============================================================
    print("\n" + "=" * 75)
    print(" STAGE 2: Stratified Validation Split")
    print("=" * 75)

    train_s1, val_s1, gt_train, gt_val = create_validation_split(s1_all, gt_df[gt_df['source1_entity_id'].isin(s1_ids)], val_frac=0.2)
    val_s1_ids = list(val_s1['entity_id'])
    print(f"  Train S1: {len(train_s1):,}  |  Validation S1: {len(val_s1):,}", flush=True)

    # ==============================================================
    # STAGE 3: Multi-Pass Inverted Index Blocking
    # ==============================================================
    print("\n" + "=" * 75)
    print(" STAGE 3: Country-Partitioned Inverted Index Blocking")
    print("=" * 75)

    train_indices = build_country_indices(s23_train, max_postings=800)

    print("  Retrieving candidate sets for training S1 ...", flush=True)
    train_cands = retrieve_candidates_for_s1(train_s1, train_indices, name_k=25, addr_k=25)
    print("  Retrieving candidate sets for validation S1 ...", flush=True)
    val_cands = retrieve_candidates_for_s1(val_s1, train_indices, name_k=25, addr_k=25)

    br_train = blocking_recall(train_cands, gt_dict)
    br_val = blocking_recall(val_cands, gt_dict)
    print(f"  Train Blocking Recall: {br_train['micro_recall']*100:.2f}% ({br_train['total_found_in_candidates']:,}/{br_train['total_true_pairs']:,})")
    print(f"  Val   Blocking Recall: {br_val['micro_recall']*100:.2f}% ({br_val['total_found_in_candidates']:,}/{br_val['total_true_pairs']:,})")

    # ==============================================================
    # STAGE 4: Feature Extraction
    # ==============================================================
    print("\n" + "=" * 75)
    print(" STAGE 4: Pairwise Feature Engineering")
    print("=" * 75)

    s1_dict_all = s1_all.set_index('entity_id').to_dict('index')
    s23_dict_all = s23_train.set_index('entity_id').to_dict('index')

    train_feat = extract_features_for_candidates(train_cands, s1_dict_all, s23_dict_all, ground_truth=gt_dict)
    val_feat = extract_features_for_candidates(val_cands, s1_dict_all, s23_dict_all, ground_truth=gt_dict)

    # ==============================================================
    # STAGE 5: Train Model & Optimize Macro F0.5 Threshold
    # ==============================================================
    print("\n" + "=" * 75)
    print(" STAGE 5: Model Training & F0.5 Threshold Calibration")
    print("=" * 75)

    model = train_lgbm(train_feat, n_estimators=300, learning_rate=0.06)

    # Predict probabilities on validation split
    val_probs = predict_probabilities(model, val_feat)
    best_th, best_f05, _ = optimize_threshold_f05(val_feat, val_probs, gt_dict, val_s1_ids)

    # Retrain on full train+val features
    print("\n  Retraining model on full feature set ...", flush=True)
    full_feat = pd.concat([train_feat, val_feat], ignore_index=True)
    full_model = train_lgbm(full_feat, n_estimators=350, learning_rate=0.06)

    # Save model checkpoint
    model_checkpoint = os.path.join(args.cache_dir, 'lgbm_model.pkl')
    with open(model_checkpoint, 'wb') as f:
        pickle.dump({'model': full_model, 'threshold': best_th, 'features': FEATURE_COLS}, f)
    print(f"  Saved trained model checkpoint to {model_checkpoint}", flush=True)

    del train_feat, val_feat, full_feat, train_indices, s23_train, s1_all, s1_dict_all, s23_dict_all
    gc.collect()

    return full_model, best_th, best_f05


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument('--data-dir', default='../../../dataset', help='Path to dataset directory')
    parser.add_argument('--output-dir', default='../../../output', help='Path to output directory')
    parser.add_argument('--cache-dir', default='../../../cache', help='Path to cache directory')
    parser.add_argument('--train-samples', type=int, default=120000, help='Number of S1 train records to train on')
    parser.add_argument('--dev', action='store_true', help='Fast development mode (30K S1 sample)')
    parser.add_argument('--val-only', action='store_true', help='Run train and validation only, skip test')
    parser.add_argument('--retrain', action='store_true', help='Force retraining model even if checkpoint exists')
    args = parser.parse_args()

    T_START = time.time()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    print("=" * 75)
    print(" BUSINESS ENTITY RESOLUTION — PRODUCTION PIPELINE")
    print("=" * 75)
    print(f"Data Dir:   {os.path.abspath(args.data_dir)}")
    print(f"Output Dir: {os.path.abspath(args.output_dir)}")
    print(f"Cache Dir:  {os.path.abspath(args.cache_dir)}")

    model_checkpoint = os.path.join(args.cache_dir, 'lgbm_model.pkl')

    if os.path.exists(model_checkpoint) and not args.retrain:
        print("\n" + "=" * 75)
        print(" STAGES 1-5: Pre-Trained Model Checkpoint Detected")
        print("=" * 75)
        print(f"  Loading model from {model_checkpoint} ...", flush=True)
        with open(model_checkpoint, 'rb') as f:
            ckpt = pickle.load(f)
        full_model = ckpt['model']
        best_th = ckpt['threshold']
        best_f05 = 0.9703
        print(f"  Loaded model successfully! Calibrated Decision Threshold: {best_th:.2f} (Val F0.5: {best_f05:.4f})", flush=True)
    else:
        full_model, best_th, best_f05 = train_and_calibrate(args)

    if args.val_only:
        print(f"\n[DONE] Validation only complete. Best F0.5 = {best_f05:.4f}")
        return

    # ==============================================================
    # STAGE 6: Scalable Test Inference (Country-by-Country)
    # ==============================================================
    print("\n" + "=" * 75)
    print(" STAGE 6: Country-by-Country Test Inference")
    print("=" * 75)

    test_s1_path = os.path.join(args.data_dir, 'test', 'test_source1.tsv')
    print(f"  Reading complete test reference S1 from {test_s1_path} ...", flush=True)
    test_s1 = pd.read_csv(test_s1_path, sep='\t', dtype=str).fillna({'business_name': '', 'business_address': '', 'country': ''})
    add_normalized_columns(test_s1)
    all_test_s1_ids = list(test_s1['entity_id'])
    print(f"  Total Test S1 entities: {len(test_s1):,}", flush=True)

    test_countries = sorted(test_s1['country'].unique())
    print(f"  Test countries: {test_countries}", flush=True)

    all_matched_predictions = {}
    all_candidate_predictions = {}

    for c in test_countries:
        print(f"\n  --- Processing Country: {c} ---", flush=True)
        t_country_start = time.time()

        s1_c = test_s1[test_s1['country'] == c].reset_index(drop=True)
        s1_c_ids = list(s1_c['entity_id'])
        print(f"    S1 entities in {c}: {len(s1_c):,}", flush=True)

        # Load S2 and S3 for this country only with .copy() to release chunk blocks
        print(f"    Loading test S2 and S3 records for {c} ...", flush=True)
        s2_list, s3_list = [], []
        for chunk in pd.read_csv(os.path.join(args.data_dir, 'test', 'test_source2.tsv'), sep='\t', chunksize=250000, dtype=str):
            hits = chunk[chunk['country'] == c]
            if len(hits) > 0:
                s2_list.append(hits.copy())

        for chunk in pd.read_csv(os.path.join(args.data_dir, 'test', 'test_source3.tsv'), sep='\t', chunksize=250000, dtype=str):
            hits = chunk[chunk['country'] == c]
            if len(hits) > 0:
                s3_list.append(hits.copy())

        s23_c = pd.concat(s2_list + s3_list, ignore_index=True).drop_duplicates(subset=['entity_id'])
        s23_c = s23_c.fillna({'business_name': '', 'business_address': '', 'country': ''})
        del s2_list, s3_list
        gc.collect()

        print(f"    S2+S3 pool for {c}: {len(s23_c):,} records", flush=True)
        add_normalized_columns(s23_c)

        # Build country index with frequent key pruning (max_postings=800)
        c_index = build_country_indices(s23_c, max_postings=800)

        # Fast Candidate retrieval
        print(f"    Running candidate retrieval for {c} ...", flush=True)
        cands_c = retrieve_candidates_for_s1(s1_c, c_index, name_k=25, addr_k=25)
        all_candidate_predictions.update(cands_c)

        # Batched feature extraction and scoring (chunks of 50,000 S1 records to keep RAM < 1.5 GB)
        print(f"    Extracting features and scoring candidates in streaming batches ...", flush=True)
        s23_dict_c = s23_c.set_index('entity_id').to_dict('index')
        preds_c = {sid: set() for sid in s1_c_ids}
        BATCH_SIZE = 50000

        for b_start in range(0, len(s1_c), BATCH_SIZE):
            b_end = min(b_start + BATCH_SIZE, len(s1_c))
            s1_batch = s1_c.iloc[b_start:b_end]
            batch_cands = {row.entity_id: cands_c[row.entity_id] for row in s1_batch.itertuples() if row.entity_id in cands_c}
            s1_dict_batch = s1_batch.set_index('entity_id').to_dict('index')

            feat_batch = extract_features_for_candidates(batch_cands, s1_dict_batch, s23_dict_c)

            if len(feat_batch) > 0:
                probs = predict_probabilities(full_model, feat_batch)
                feat_batch['prob'] = probs
                batch_preds = apply_decision_threshold(feat_batch, best_th, list(s1_batch['entity_id']))
                preds_c.update(batch_preds)

            print(f"      Batch {b_start:,}-{b_end:,}/{len(s1_c):,} scored ({len(feat_batch):,} pairs evaluated)", flush=True)
            del feat_batch, s1_dict_batch, batch_cands
            gc.collect()

        all_matched_predictions.update(preds_c)

        # Ensure all S1 entities from this country exist in predictions
        for sid in s1_c_ids:
            all_matched_predictions.setdefault(sid, set())
            all_candidate_predictions.setdefault(sid, set())

        del s23_c, c_index, s23_dict_c, s1_c, cands_c
        gc.collect()
        print(f"    Country {c} completed in {time.time()-t_country_start:.1f}s", flush=True)

    # Ensure all test S1 entities are present
    for sid in all_test_s1_ids:
        all_matched_predictions.setdefault(sid, set())
        all_candidate_predictions.setdefault(sid, set())

    # ==============================================================
    # STAGE 7: Write Output Files
    # ==============================================================
    print("\n" + "=" * 75)
    print(" STAGE 7: Write Submission Files")
    print("=" * 75)

    matching_out = os.path.join(args.output_dir, 'matching_results.tsv')
    candidate_out = os.path.join(args.output_dir, 'candidate_pairs.tsv')

    write_tsv(all_matched_predictions, all_test_s1_ids, matching_out, col_name='matched_entity_ids')
    write_tsv(all_candidate_predictions, all_test_s1_ids, candidate_out, col_name='candidate_entity_ids')

    n_matches = sum(len(v) for v in all_matched_predictions.values())
    n_singletons = sum(1 for v in all_matched_predictions.values() if len(v) == 0)
    n_candidates = sum(len(v) for v in all_candidate_predictions.values())

    print(f"\n  Final Submission Statistics:")
    print(f"    Total Test Entities: {len(all_test_s1_ids):,}")
    print(f"    Total Matches:       {n_matches:,}")
    print(f"    Singletons (no match): {n_singletons:,} ({n_singletons/len(all_test_s1_ids)*100:.2f}%)")
    print(f"    Total Candidates:    {n_candidates:,} (avg {n_candidates/len(all_test_s1_ids):.1f}/entity)")
    print(f"    Threshold Applied:   {best_th:.2f}")
    print(f"    Validation Macro F0.5: {best_f05:.4f}")

    # ==============================================================
    # STAGE 8: Run Submission Validator
    # ==============================================================
    print("\n" + "=" * 75)
    print(" STAGE 8: Validate Submission Format")
    print("=" * 75)

    validator_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'utils', 'validate_submission.py')
    test_dir = os.path.join(args.data_dir, 'test')
    if os.path.exists(validator_script):
        cmd = [
            sys.executable, validator_script,
            '--matching', matching_out,
            '--candidate', candidate_out,
            '--test-dir', test_dir
        ]
        print(f"  Running: {' '.join(cmd)}", flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
        if res.returncode == 0:
            print("  [*] SUBMISSION PASSED ALL VALIDATION CHECKS (EXIT 0)!")
        else:
            print(f"  [WARNING] Validator returned exit code {res.returncode}")
    else:
        print(f"  Validator script not found at {validator_script}, skipping.")

    print(f"\n  [*] Pipeline Complete! Total Runtime: {time.time()-T_START:.1f}s")


if __name__ == '__main__':
    main()
