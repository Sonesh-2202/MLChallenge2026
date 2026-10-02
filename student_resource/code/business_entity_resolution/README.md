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
├── README.md                          # End-to-end reproduction instructions
├── requirements.txt                   # Pinned dependency environment
└── src/
    ├── preprocess.py                  # Text normalization, NFKD decomposition, legal suffix & address parsing
    ├── blocking_v4.py                 # Multi-pass high-recall compound inverted index blocker
    ├── features_v4.py                 # 43-dim RapidFuzz, residual raw-name, and Indic script features
    ├── features_v9.py                 # 16-dim token IDF distinctiveness, Roman numerals, and legal form features
    ├── features_v10.py                # 67-dim feature suite (IDF cosines, US state & FR dept guardrails, 4-grams)
    ├── model.py                       # LightGBM training & prediction wrapper utilities
    ├── train_v10.py                   # High-precision LightGBM training script (67 features)
    ├── train_v13_xgboost.py           # Complementary XGBoost histogram booster training
    ├── evaluate.py                    # Competition Macro F0.5 evaluation engine
    ├── run_strategy1_singleton_recovery.py # V11 production singleton rescue engine (0 duplicates)
    ├── run_v12_multimatch_recovery.py # Multi-match cluster expansion pass
    ├── run_v13_ensemble_pipeline.py   # Dual-model ensemble pipeline orchestrator
    └── pipeline.py                    # Unified CLI production entrypoint
```

---

## Installation & Setup

Ensure Python 3.9+ is installed, then install all pinned dependencies:

```bash
cd code/business_entity_resolution
pip install -r requirements.txt
```

---

## End-to-End Pipeline Reproduction

### 1. Training the Model (Optional / From Scratch)
To fit the token IDF dictionary and train the 67-feature LightGBM booster on the training dataset:
```bash
python src/train_v10.py --data-dir ../../../dataset --cache-dir ../../../cache --out-model ../../../cache/lgbm_model_v10.pkl
```

### 2. Full Test Inference & Singleton Rescue (Best Submission Generation)
To run the production pipeline that generates the verified submission outputs:
```bash
python src/pipeline.py --data-dir ../../../dataset --output-dir ../../../output --cache-dir ../../../cache --workers 8
```

### 3. Submission Format & Constraint Verification
Verify that generated files satisfy all competition requirements (singletons, no duplicates, valid IDs):
```bash
python ../../utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../dataset/test
```

### Output Files Generated:
- `output/matching_results.tsv`: Final matches identical to the leaderboard submission (SHA-256: `E50732AA548A3DDCD561830D18733B2BB6EAF088D62F8828F8B56FE0642E9CC0`).
- `output/candidate_pairs.tsv`: Complete candidate pairs generated during the blocking stage (1,732,544 rows).

