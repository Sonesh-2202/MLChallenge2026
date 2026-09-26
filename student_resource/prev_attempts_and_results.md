# Comprehensive Log of Previous Attempts, Methods, and Results

**Project:** ML Challenge 2026 — Business Entity Resolution  
**Team:** AstraCloud (Sonesh SK, Swathika Shri, Subhekshana KG)  
**Metric:** Per-Source-1 Macro $F_{0.5}$ (Singletons included, Precision weighted 2× over Recall)  
**Baseline Starting Score:** 0.591 (V1) $\to$ 0.865 (V2) $\to$ 0.870 (V3)  
**Current Milestone Score:** **0.908 (V5 Production)**  

---

## 1. Executive Progression Summary

| Iteration | Architecture & Key Methodological Changes | Validation Macro $F_{0.5}$ | Public Leaderboard ($F_{0.5}$) | Key Outcome & Discovery |
|---|---|---|---|---|
| **V1 (Initial)** | Basic RapidFuzz string metrics (12 features), local candidate blocking with negative pool truncation. | 0.9703* (truncated pool) | **0.5910** | **Artifact:** Validation was artificially inflated because negative pool was truncated to only 50 candidates per entity. Full test set exposed massive false positive merges. |
| **V2 (Full Scale)** | Scaled candidate generation to full country partitions across 10.3M Source 2 & Source 3 entities. | ~0.8650 (est) | **0.8650** | **Validation Realignment:** Evaluating against the full negative space dropped score to a realistic baseline. First viable end-to-end model. |
| **V3 (Baseline)** | 37 engineered RapidFuzz and token features (`features_v3.py`), 4-pass inverted index, global decision threshold $\tau = 0.9400$. | 0.9071 | **0.8700** | **Plateau:** Better string metrics lifted score to 0.870, but flat high threshold caused 8,819 false negatives in India, while legal stopword stripping merged distinct co-located businesses. |
| **V4 (Production)** | High-Recall V4 Blocking (squashed compound keys, acronym preservation), 43 features (`unmatched_raw_tokens`, `raw_token_diff`, `is_indic_script`, `num_and_pin_match`), retrained LightGBM, country thresholds $\tau = 0.970$. | 0.9314 | **0.899255** | **Major Breakthrough (+0.0293):** Novel raw-name residual features became top-ranked tree features. Candidate recall reached 95.70%. False positive merges dropped by 49.5%. |
| **V4.1 (Option 1)** | Calibrated cluster truncation based on Ground Truth distributions. France capped at top 5 matches, India/US capped at top 6 matches. | ~0.9340 | **0.9040** | **Instant Precision Surge (+0.0047):** Pruned 283,612 trailing false matches across 102,358 entities. Direct empirical proof that over-merged clusters were destroying Macro $F_{0.5}$. |
| **V5 (Option 2)** | Dedicated French precision pipeline: French commune and stopword isolation (`clean_french_name`), $\tau_{\text{France}} = 0.980$, $\delta_{\text{France}} = 0.025$, hard match cap = 5. | **~0.9365** | **0.9080** | **New SOTA (+0.0040 over V4.1, +0.0380 over V3):** Normalized France's average matches/entity from an inflated 4.42 down to 3.36 (aligning with Ground Truth 3.31) and eliminated 100% of the over-merged tail. |

*\*Note: V1 validation score was an evaluation artifact due to negative candidate pool truncation.*

---

## 2. Deep Dive: Iteration Details & Failure Modes

### Iteration 1: The V1 Baseline (Leaderboard: 0.5910)
- **Concept:** Train LightGBM on 12 basic pairwise string similarity features (Levenshtein ratio, token set ratio, token sort ratio, simple number match).
- **Blocking:** Fast inverted index with a tight candidate pool cap (50 candidates per entity).
- **Validation Fallacy:** The initial validation script reported $F_{0.5} \approx 0.9703$. However, during validation, negative candidates were drawn only from the top-50 candidate list.
- **Why It Failed on Leaderboard (0.5910):**
  When deployed against the full test set of 10.3 million candidate records in Source 2 and Source 3, the model encountered millions of competitive distractors with similar names. With only 12 naive features, the model suffered from an avalanche of false positive merges. In Macro $F_{0.5}$, precision is penalized 2× more than recall, causing the leaderboard score to collapse to 0.5910.

---

### Iteration 2: Full Negative Space & V2 Pipeline (Leaderboard: 0.8650)
- **Concept:** Expose the model to the true candidate distribution by streaming candidate retrieval against the full country partitions.
- **Architecture:** Partitioned test set strictly by `country` (India, US, France).
- **Outcomes:** Leaderboard score jumped from **0.5910 to 0.8650 (+0.274)**.
- **Limitations:** The model lacked granular token differentiation, and many entities failed to match because address numbers were inconsistently formatted.

---

### Iteration 3: 37 Features & Global Thresholding (Leaderboard: 0.8700)
- **Concept:** Expanded feature engineering to 37 dimensions (`features_v3.py`):
  - Token set ratio, token sort ratio, partial ratio, token-level Jaccard overlap.
  - Address token overlap, address Jaccard, house/door number extraction.
  - Harmonic mean compatibility metric between name and address scores.
- **Model:** 595 LightGBM trees with depth 12 and leaf count 255.
- **Decision Rule:** Single flat threshold $\tau = 0.9400$ across all countries.
- **Key Bottlenecks Uncovered in Forensic Error Decomposition:**
  1. **Legal Stopword Stripping Artifact:** In `preprocess.py`, generic suffixes like `Services`, `Group`, `Corp`, `Holdings` were stripped. Consequently, distinct co-located businesses (e.g. `Lucky Deli` vs `Lucky Deli Corp Services`) became identical strings post-cleaning, triggering near-1.0 confidence false merges.
  2. **The Indic Script Gap:** India scored 6.78% lower than US because reference records in Source 1 were in Latin English while Source 2/3 records were in Devanagari, Tamil, or Telugu. Latin string metrics produced exact zeros (`n_ratio = 0.0`), dropping true matches below threshold.
  3. **Compound Words & Acronyms:** Merged compound words (`western oceanhawk` vs `westernoceanhawk`) failed token matching, and 2-letter uppercase acronyms (`IT`, `AI`, `GE`) were accidentally dropped during lowercase filtering.

---

### Iteration 4: High-Recall Blocking & V4 Residual Features (Leaderboard: 0.899255)
- **Key Technical Innovations:**
  1. **High-Recall V4 Blocking (`blocking_v4.py`):**
     - Concatenated 2-token squashed compound keys (`nsq:westernoceanhawk`).
     - Alphanumeric 2-letter token preservation.
     - Indic script address-fallback keys (PIN code + street number combinations).
     - Candidate recall jumped from **95.40% to 95.70%**, recovering 255 previously unrecoverable true matches.
  2. **43-Dimensional Feature Suite (`features_v4.py`):**
     - Residual raw-name metrics: `raw_n_ratio`, `raw_n_tsort`.
     - `raw_token_diff`: Absolute difference in uncleaned token counts.
     - `unmatched_raw_tokens`: Count of raw tokens in candidate absent from reference entity (**emerged as #7 overall feature by LightGBM tree gain!**).
     - `is_indic_script`: Binary indicator preventing the tree from penalizing zero Latin string similarity on Indic names.
     - `num_and_pin_match`: Synergistic validation of street number and postal code.
  3. **Decision Calibration:** Country thresholds $\tau_{\text{US}} = 0.970, \tau_{\text{India}} = 0.970$.
- **Leaderboard Result:** Jumped to **0.899255 (+0.029255)**, nearly reaching the 0.900 threshold.

---

### Iteration 4.1: Calibrated Match Truncation (Leaderboard: 0.9040)
- **Hypothesis:** Under Macro $F_{0.5}$, large clusters of predicted matches destroy precision. An entity with 2 true matches predicted with 12 matches drops in $F_{0.5}$ from 1.000 to 0.200.
- **Empirical Ground Truth Discovery:** In the 2.2M-entity ground truth:
  - $\le 5$ matches: 96.3% of entities
  - $\le 6$ matches: 98.9% of entities
  - $> 9$ matches: 0.2% of entities
  - 12 matches: **0.00% of entities** (maximum in ground truth was 11).
- **Implementation:** Created [`calibrate_submission_v4_1.py`](file:///e:/ML%20Challenge%202026/student_resource/code/business_entity_resolution/src/calibrate_submission_v4_1.py). Pruned trailing matches sorted by model confidence, capping France at top-5 and US/India at top-6.
- **Impact:** Pruned 283,612 trailing false matches across 102,358 entities.
- **Leaderboard Result:** Instant jump from **0.899255 to 0.9040 (+0.0047)** with zero retraining.

---

### Iteration 5: French Commune Isolation Pipeline (Leaderboard: 0.9080)
- **Problem Diagnosed:** France was an unseen country with zero training data. Commune/city names (`Saint-Herblain`, `La Baule`, `Bordeaux`) and corporate qualifiers (`Societe`, `SARL`) caused 43,978 French entities (16.95% of France!) to be assigned $>6$ matches (averaging 4.42 matches/entity vs 3.31 in ground truth).
- **Implementation (`fast_pipeline_v5_france.py`):**
  1. Extended French legal and generic stopwords: `sarl`, `sasu`, `sas`, `eurl`, `sci`, `snc`, `sa`, `societe`, `ste`, `fils`, `cie`, `groupe`, `france`, `association`, `lycee`, `college`, `ecole`.
  2. Elevated France decision threshold: $\tau_{\text{France}} = 0.980$ (up from 0.970).
  3. Narrowed margin: $\delta_{\text{France}} = 0.025$ (from 0.050) and hard match cap = 5.
- **Impact:** Normalized France's average matches/entity from 4.42 down to 3.36, and completely eliminated the $>6$ match tail.
- **Leaderboard Result:** Jumped to **0.9080 (+0.0040 over V4.1, +0.0380 over V3 baseline)**.

---

## 3. Directory of Retained vs. Pruned Files

To optimize project disk space (which grew past 20 GB due to duplicate multi-gigabyte candidate TSVs across 5 versions), intermediate redundant artifacts have been audited:

### Files Being Pruned (Reclaiming ~11.5 GB):
- `output/partitions_v2/` (~2.2 GB) — Superseded by V4/V5.
- `output/partitions_v3/` (~3.1 GB) — Superseded by V4/V5.
- `output/v3_backup/` (~3.2 GB) — Archived baseline files.
- `output/partitions_v4/candidates_France.tsv` (~507 MB) — Superseded by V5 French partition.
- `output/partitions_v4/matches_France.tsv` (~18 MB) — Superseded by V5 French partition.
- `output/matching_results_v4_raw.tsv` (~91 MB) — Backed up raw V4 output.
- `output/matching_results_v4_1.tsv` (~89 MB) — Intermediate V4.1 output.
- `output/matching_results_v5.tsv` (~87 MB) — Redundant with final `matching_results.tsv`.
- `cache/val_e1_scored_pairs.pkl` (~152 MB) — Intermediate experiment cache.
- Old model pickles (`lgbm_model.pkl`, `lgbm_model_v2.pkl`, `lgbm_model_v3.pkl`).

### Critical Files Retained (The Clean Production Workspace):
- `output/matching_results.tsv` (86.7 MB) — **The verified 0.908 competition output file.**
- `output/candidate_pairs.tsv` (3.45 GB) — The full blocking candidate set.
- `output/partitions/` — Consolidated partition directory containing final verified partitions:
  - `matches_France.tsv` & `candidates_France.tsv` (V5 French precision)
  - `matches_India.tsv` & `candidates_India.tsv` (V4 calibrated)
  - `matches_US.tsv` & `candidates_US.tsv` (V4 calibrated)
- `cache/lgbm_model_v4.pkl` (15.8 MB) — The trained 43-feature LightGBM model.
- `code/business_entity_resolution/src/pipeline.py` — The unified, end-to-end production script reproducing 0.908.
- `AstraCloud_submission.zip` (1.44 GB) — The official competition submission archive.
