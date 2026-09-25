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
    ├── blocking.py                # Multi-pass inverted index candidate retrieval
    ├── features.py                # Pairwise RapidFuzz feature extraction
    ├── model.py                   # LightGBM training & prediction
    ├── evaluate.py                # Macro F0.5 evaluation metric with singleton logic
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

### 2. Full Production End-to-End Pipeline
To run the complete pipeline, perform country-by-country inference on all 1,732,544 test entities, and generate the final submission files:
```bash
python src/run_pipeline.py --data-dir ../../../dataset --output-dir ../../../output --cache-dir ../../../cache
```

Output files produced:
- `output/matching_results.tsv`: Final predictions (`source1_entity_id\tmatched_entity_ids`)
- `output/candidate_pairs.tsv`: Blocking candidate set (`source1_entity_id\tcandidate_entity_ids`)
- `cache/lgbm_model.pkl`: Trained model artifact and calibrated decision threshold.
