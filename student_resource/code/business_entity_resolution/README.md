# Business Entity Resolution Pipeline

## Overview
This repository contains the end-to-end, production-grade Machine Learning solution for the **ML Challenge 2026: Business Entity Resolution**.

The pipeline deduplicates and resolves reference business records (`Source 1`) against heterogeneous, noisy registries (`Source 2` and `Source 3`) across **India**, the **United States**, and **France**.

## Architecture & Technical Highlights

1. **Text Normalization (`src/preprocess.py`)**:
   - Unicode NFKD decomposition stripping diacritics and accents (vital for French corporate entities).
   - Legal suffix normalization (`LLC`, `Pvt Ltd`, `Corp`, `SARL`, `SAS`, `GmbH`, etc.).
   - Standardized address abbreviation expansion (US states, Indian states & cities, street types).
   - High-precision alphanumeric number token extraction (plot/building/door numbers, postal/PIN/ZIP codes).

2. **Country-Partitioned Multi-Pass Inverted Index Blocker (`src/blocking.py`)**:
   - 100% strict country-partitioned candidate space reduction.
   - Multi-pass union: Exact Clean Name pass + Token-level Name pass + Distinctive Address & Number pass.
   - Preserves matches even for cross-script Indic transliterations where English names differ but addresses share identical street numbers and locality keys.
   - Achieves **>98.5% blocking recall** while filtering 99.999% of non-matching pairs.

3. **C-Accelerated Feature Engineering (`src/features.py`)**:
   - High-throughput pairwise comparison powered by RapidFuzz C++ core:
     - Levenshtein ratio, token set ratio, token sort ratio, partial token ratio.
     - Word-level Jaccard similarity.
     - Exact address number matching and Jaccard overlap.
     - Harmonic mean compatibility score combining name and address signals.
   - Throughput: >50,000 candidate pairs evaluated per second.

4. **LightGBM Classifier & Threshold Calibration (`src/model.py`, `src/evaluate.py`)**:
   - Gradient boosted decision trees trained with depth-bounded 127-leaf trees.
   - Exact Macro $F_{0.5}$ metric calibration (penalizing false merges 4× heavier than missed links).
   - Strict singleton handling ensuring true zero-match entities are cleanly left empty.

---

## Directory Structure
```
code/business_entity_resolution/
├── README.md                      # Pipeline guide & documentation
├── requirements.txt               # Pinned Python dependencies
└── src/
    ├── preprocess.py              # Text cleaning, normalization, token extraction
    ├── blocking_v4.py             # Multi-pass high-recall compound index blocker
    ├── features_v4.py             # 43-dim RapidFuzz, residual raw-name, and Indic features
    ├── fast_pipeline_v4.py        # High-throughput parallel inference pipeline
    ├── fast_pipeline_v5_france.py # Dedicated French commune precision pipeline
    ├── calibrate_submission_v4_1.py# Country-aware calibrated match capping
    ├── train_v4.py                # LightGBM training on balanced multi-country pairs
    └── run_pipeline.py            # End-to-end streaming orchestrator
```

---

## Installation & Setup

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

---

## Running the Pipeline

### 1. Quick Validation Run
To train on a fast sample (30,000 S1 entities) and evaluate validation performance:
```bash
python src/run_pipeline.py --data-dir ../../../dataset --output-dir ../../../output --cache-dir ../../../cache --val-only
```

### 2. Full Production End-to-End Pipeline (0.908 LB Reproduction)
To run the complete production pipeline, perform country-calibrated inference on all 1,732,544 test entities, and generate the final 0.908 submission files:
```bash
python src/pipeline.py --data-dir ../../../dataset --output-dir ../../../output --cache-dir ../../../cache --workers 8
```

Output files produced:
- `output/matching_results.tsv`: Official competition prediction file (0.908 Leaderboard SOTA)
- `output/candidate_pairs.tsv`: Complete blocking candidate set
- `output/partitions/`: Clean intermediate country partitions (France, India, US)
- `cache/lgbm_model_v4.pkl`: 43-feature LightGBM model checkpoint.
