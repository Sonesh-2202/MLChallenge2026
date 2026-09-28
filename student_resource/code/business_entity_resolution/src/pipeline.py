#!/usr/bin/env python3
"""
ML Challenge 2026: Business Entity Resolution
Production Entrypoint: pipeline.py (V13 Dual-Model Ensemble)

Executes the top-scoring V13 pipeline combining:
1. 67-dimensional feature engineering suite
2. Dual LightGBM (55%) + XGBoost Histogram (45%) model ensemble
3. Country-partitioned multi-pass candidate blocking
4. Global 1-to-1 bipartite conflict resolution
5. High-precision singleton and multi-match cluster deficit recovery

Outputs verified files:
  - output/matching_results.tsv (V13 Grandmaster submission)
  - output/candidate_pairs.tsv (Full candidate blocking output)
"""

import os
import sys
import argparse

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from run_v13_ensemble_pipeline import main as run_v13

def main():
    parser = argparse.ArgumentParser(
        description="Run ML Challenge 2026 Business Entity Resolution Pipeline (V13 SOTA)"
    )
    parser.add_argument("--data-dir", default=None, help="Path to dataset directory")
    parser.add_argument("--output-dir", default=None, help="Path to output directory")
    parser.add_argument("--cache-dir", default=None, help="Path to cache directory")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker threads")
    args = parser.parse_args()

    print("=" * 70)
    print(" ASTRACLOUD V13 DUAL-MODEL ENSEMBLE PRODUCTION PIPELINE")
    print("=" * 70)
    run_v13()

if __name__ == '__main__':
    main()
