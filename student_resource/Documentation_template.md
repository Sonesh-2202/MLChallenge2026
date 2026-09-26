# ML Challenge 2026: Business Entity Resolution Solution Document

**Team Name:** AstraCloud  
**Team Members:** Sonesh SK, Swathika Shri, Subhekshana KG  
**Submission Date:** September 26, 2026  
**Pipeline Revision:** Production V4 Architecture  

---

## 1. Executive Summary

We developed an end-to-end, high-precision, country-partitioned entity resolution architecture designed to resolve 1.73 million reference business entities (`Source 1`) against over 10 million noisy, heterogeneous commercial records (`Source 2` and `Source 3`) spanning **India**, the **United States**, and **France**.

Our core technical innovations in the **V4 Production Pipeline** are:
1. **High-Recall 4-Pass Inverted Index Blocker (V4)**: Integrates squashed compound name keys, 2-letter uppercase/acronym preservation, and Indic transliteration address-fallback keys, lifting candidate blocking recall to **95.70%** (US: **97.57%**, India: **92.90%**) and eliminating 255 previously unrecoverable blocking misses while maintaining a >99.997% candidate reduction ratio (~154 candidates per entity).
2. **43-Feature RapidFuzz & Geo-Engineered Feature Suite**: Augments traditional string distance with novel residual raw-name features (`raw_n_ratio`, `raw_token_diff`, `unmatched_raw_tokens`), Indic script detection (`is_indic_script`), and postal code / street number synergies.
3. **Precision-Calibrated LightGBM Decision Engine**: Tuned specifically for the asymmetric **Macro $F_{0.5}$ metric** ($\beta=0.5$), which weights precision 2x over recall and strictly penalizes false merges on singletons.

On our fixed, leakage-safe validation benchmark of 24,999 Source-1 entities evaluated against the full 10.3M candidate pool, the V4 pipeline achieves:
- **Macro $F_{0.5}$ Score:** **0.9314** (an absolute increase of **+2.43%** over the V3 baseline of 0.9071)
- **Validation Precision:** **97.57%** (up from 95.22%)
- **Validation Recall:** **86.67%** (up from 85.19%)
- **False Positive Elimination:** Reduced false positive merges from 3,696 to 1,867 (a **49.5% reduction** in false merges).

---

## 2. Methodology & Problem Analysis

### 2.1 Problem Analysis & Discovered Bottlenecks
Through rigorous empirical error decomposition on 24,999 validation entities, we identified four primary error drivers in previous pipeline versions:

1. **Over-Aggressive Legal Stopword Stripping**:
   - In earlier versions, words like `Services`, `Group`, `Industries`, and `Holdings` were stripped as legal suffixes. This caused distinct businesses situated at the same commercial building (e.g. `Lucky Deli` vs `Lucky Deli Corp Services`) to become identical after normalization, generating false positive merges with near-1.0 confidence.
   - *V4 Solution*: We engineered raw uncleaned name features (`raw_n_ratio`, `unmatched_raw_tokens`, `raw_token_diff`) that provide direct evidence of dropped company qualifiers.
2. **Cross-Script & Indic Transliteration Disconnect**:
   - In India, `Source 1` reference names appear in Latin English, while `Source 2` and `Source 3` frequently contain Indic script transliterations (Devanagari, Tamil, Telugu, Gujarati). String metrics produced exact zeros (`n_ratio = 0.0`), driving true matches to borderline scores ($p \approx 0.9395$) that fell just below high thresholds.
   - *V4 Solution*: Introduced `is_indic_script` feature and PIN-code / address-heavy fallback blocking so the tree learns not to penalize low Latin name similarity when address and number features strongly match.
3. **Squashed Compound Words**:
   - Words like `western oceanhawk` $\leftrightarrow$ `westernoceanhawk` or `sweet tavern` $\leftrightarrow$ `sweettavern` failed multi-word token matching.
   - *V4 Solution*: Constructed concatenated 2-token squashed compound keys (`nsq:westernoceanhawk`) in the blocking index.
4. **Acronym Dropping**:
   - Lowercase normalization before length filtering inadvertently dropped 2-letter acronyms (`IT`, `AI`, `GE`, `HP`, `SA`).
   - *V4 Solution*: Preserved 2-letter alphanumeric tokens in `blocking_v4.py`.

### 2.2 Solution Strategy
- **Approach**: Multi-Pass Country-Partitioned Inverted Index Blocking + 43 C++ RapidFuzz Features + Asymmetric LightGBM GBDT Classifier + Country-Calibrated Margin Decision Rules.
- **Hardware Architecture**: 32-core parallel CPU execution with OpenMP thread pools and streaming chunk processing, guaranteeing that peak memory remains under 8 GB RAM even when processing 6M+ records.

---

## 3. Candidate Generation (Blocking)

To reduce the $1.73 \times 10^6 \times 10.3 \times 10^6 \approx 1.78 \times 10^{13}$ all-pairs Cartesian product space into a tractable candidate set:

1. **Exact Country Partitioning**: Exact matching on `country` reduces the comparison space by $3\times$ with zero recall loss.
2. **4-Pass Inverted Indexing**:
   - *Pass 1 (Exact Clean Name)*: Instant lookup of clean name matches post NFKD accent removal.
   - *Pass 2 (Name Inverted Index)*: Word tokens ($\ge 3$ characters and significant 2-character tokens), 4-gram prefixes, and squashed 2-word concatenations, weighted by inverse document frequency:
     $$w(t) = \frac{1}{\log(1 + |P(t)|)}$$
   - *Pass 3 (Address & Number Inverted Index)*: Alphanumeric building/plot numbers, 5-digit ZIPs, 6-digit PIN codes, and distinctive locality tokens.
   - *Pass 4 (Compound Cross-Field Index)*: Token pairs (`np:`), name+number (`n_num:`), and postal code+address (`pin_a:`).
3. **Retrieval**: Top 70 compound candidates, top 45 name candidates, top 45 address candidates, and top 100 exact matches.
4. **Validation Blocking Recall**:
   - **US**: **97.57%** (missed only 1,259 pairs out of 51,778)
   - **India**: **92.90%** (missed 2,454 pairs out of 34,544)
   - **Overall**: **95.70%** (found 82,609 / 86,322 true matches)
   - **Average Candidates per Entity**: **154.0** (Reduction Ratio: >99.997%).

---

## 4. Matching Model & Feature Engineering

### 4.1 Features Extracted (43 Dimensions)
All string similarity metrics are executed via C++ implementations in `RapidFuzz` for maximum throughput (>50,000 candidate pairs/second):

- **Clean Name Similarity (1–16)**:
  - `n_ratio`, `n_tset`, `n_tsort`, `n_part`, `n_jacc`, `n_nospace_ratio`, `n_len_ratio`, `n_token_overlap`, `n_token_count_ratio`, `n_containment`, `n_prefix3`, `n_char_trigram_jacc`, `exact_name`, `exact_name_missing_addr`, `n_first_token_match`, `n_jw` (Jaro-Winkler).
- **Address Similarity (17–27)**:
  - `a_ratio`, `a_tset`, `a_tsort`, `a_part`, `a_jacc`, `a_len_ratio`, `a_token_overlap`, `a_containment`, `a_char_trigram_jacc`, `exact_addr`, `a_jw`.
- **Numbers & Cross-Modal Signals (28–37)**:
  - `num_match`, `num_jacc`, `num_overlap_count`, `num_mismatch`, `both_addr`, `either_missing`, `harm` (Harmonic Mean), `num_exact_mismatch`, `zip_match`, `zip_mismatch`.
- **V4 Novel Features (38–43)**:
  - `raw_n_ratio`: RapidFuzz ratio on raw uncleaned business names.
  - `raw_n_tsort`: Token sort ratio on raw uncleaned business names.
  - `raw_token_diff`: Absolute difference in raw token counts.
  - `unmatched_raw_tokens`: Count of raw tokens in candidate absent from reference entity.
  - `is_indic_script`: Binary indicator for Indic Unicode codepoints.
  - `num_and_pin_match`: Synergistic match of both street number and postal code.

### 4.2 Model Type & Hyperparameters
- **Model**: LightGBM Classifier (`LGBMClassifier`), 565 trees with early stopping.
- **Hyperparameters**:
  - `objective`: `binary` (binary logloss)
  - `num_leaves`: 255, `max_depth`: 12
  - `learning_rate`: 0.05
  - `feature_fraction`: 0.80, `bagging_fraction`: 0.80, `bagging_freq`: 5
  - `min_child_samples`: 50
  - `lambda_l1`: 0.10, `lambda_l2`: 1.00
  - `n_jobs`: 24 (multi-threaded CPU acceleration)

### 4.3 Feature Importance (Top Gain in V4)
The empirical tree gains directly confirmed the power of the novel V4 features:
1. `a_tset`: 15,713,815.0 Gain
2. `a_jacc`: 1,087,761.4 Gain
3. `n_part`: 1,074,126.5 Gain
4. `a_containment`: 955,975.3 Gain
5. `num_jacc`: 553,574.8 Gain
6. `harm`: 518,004.2 Gain
7. **`unmatched_raw_tokens`**: **501,118.7 Gain (Rank 7 overall!)**
8. `a_ratio`: 389,255.9 Gain
9. `a_len_ratio`: 363,766.1 Gain
10. `n_jw`: 323,142.2 Gain
...
15. **`raw_token_diff`**: **167,066.5 Gain**
18. **`is_indic_script`**: **140,583.0 Gain**
20. **`raw_n_ratio`**: **111,954.7 Gain**

---

## 5. Decision Threshold Calibration & Evaluation

Under the competition's **Macro $F_{0.5}$** metric:
$$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

A false merge on a singleton yields $F_{0.5} = 0.0$. We conducted a granular threshold sweep on the full 25K validation set:

| Decision Threshold $\tau$ | Macro $F_{0.5}$ | Precision | Recall | True Positives | False Positives | False Negatives |
|---|---|---|---|---|---|---|
| 0.85 | 0.9031 | 91.31% | 91.35% | 78,855 | 7,507 | 7,467 |
| 0.90 | 0.9185 | 93.84% | 90.21% | 77,867 | 5,111 | 8,455 |
| 0.93 | 0.9257 | 95.41% | 89.16% | 76,961 | 3,701 | 9,361 |
| 0.95 | 0.9295 | 96.47% | 88.17% | 76,114 | 2,787 | 10,208 |
| **0.97 (Optimal)** | **0.9314** | **97.57%** | **86.67%** | **74,814** | **1,867** | **11,508** |

The optimal decision boundary $\tau^* = 0.970$ strikes the mathematical balance required by the metric, maintaining **97.57% precision** while recovering 74,814 true matches.

---

## 6. Submission Verification & Validation

The final output is verified by `utils/validate_submission.py` against all challenge constraints:
1. **One row per test entity**: Exactly 1,732,544 rows in `matching_results.tsv` and `candidate_pairs.tsv`.
2. **Valid Target IDs**: Only Source-2 and Source-3 IDs existing in the test set appear in prediction lists.
3. **Singleton Compliance**: True singletons are represented by an empty string (no false links).
4. **Candidate Hierarchy**: Every predicted match in `matching_results.tsv` is strictly contained in that entity's candidate list in `candidate_pairs.tsv`.
5. **No Duplicates**: All ID lists contain unique, sorted identifiers with zero duplicates.

---

---

## 7. Version 5 Architecture (French Commune Isolation & Cluster Capping)

Following empirical submission analysis on the public leaderboard (where V4 achieved **0.899255**), error decomposition revealed that the unseen country **France** (15.0% of the test set, 259,452 entities) suffered from cluster inflation (averaging 4.42 matches/entity with 16.95% of entities assigned >6 matches, compared to <0.85% in training ground truth). This was caused by French commune names (`Saint-Herblain`, `La Baule`, `Bordeaux`) and corporate qualifiers (`Societe`, `SARL`) triggering spurious multi-entity matches.

The V5 architecture introduced:
1. **French Stopword & Generic Token Isolation**: Dedicated filtering of French legal forms (`sarl`, `sasu`, `sas`, `eurl`, `sci`, `snc`, `sa`, `societe`, `ste`, `fils`, `cie`, `groupe`, `france`) and educational/communal prefixes (`association`, `lycee`, `college`, `ecole`).
2. **Elevated French Decision Threshold**: $\tau_{\text{France}} = 0.980$ with narrowed margin $\delta = 0.025$.
3. **Calibrated Cluster Capping**: Hard match cap of 5 for France and 6 for US/India, aligning the prediction distribution with ground truth (bringing France's average matches/entity from 4.42 down to 3.36, and eliminating 100% of the over-merged tail).

---

## 8. Conclusion

By systematically eliminating error drivers across iterations—progressing from baseline V1 (0.591 LB) to V2 (0.865 LB), V3 (0.870 LB), V4 (0.899255 LB), and now V5—our pipeline combines high-recall multi-pass blocking (95.70% recall), 43-dimensional RapidFuzz feature extraction, and country-calibrated LightGBM decision boundaries. The entire system is computationally efficient, fully reproducible, and rigorously verified against all competition constraints.
