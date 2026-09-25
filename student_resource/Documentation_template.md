# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:**  AstraCloud
**Team Members:** Sonesh SK , Swathika Shri , Subhekshana KG 
**Submission Date:** September 25, 2026  

---

## 1. Executive Summary

We developed an end-to-end, high-precision, country-partitioned entity resolution architecture designed to resolve 1.73 million reference business entities (`Source 1`) against over 10 million noisy, heterogeneous commercial records (`Source 2` and `Source 3`) spanning **India**, the **United States**, and **France**. Our core technical innovation is a **Two-Pass Country-Partitioned Inverted Index Blocker** combined with a **C-accelerated RapidFuzz Feature Engineering Pipeline** and a **Precision-Calibrated LightGBM Classifier**. By tuning our decision boundary specifically for the competition's asymmetric **Macro $F_{0.5}$ metric** ($\beta=0.5$), which penalizes false merges four times more severely than missed matches and heavily penalizes false merges on singletons, our solution achieves an out-of-fold validation Macro $F_{0.5}$ score of **97.03% - 98.4%** while preserving strict singleton isolation.

---

## 2. Methodology

### 2.1 Problem Analysis
Key empirical insights derived during exploratory data analysis (EDA):
1. **Cross-Lingual & Multi-Script Heterogeneity**:
   - `Source 1` references are in Latin English, while `Source 2` and `Source 3` contain regional Indic script transliterations (Devanagari, Telugu, Tamil, Gujarati, Bengali, Odia) where business names appear completely distinct or empty under standard ASCII string metrics. However, their postal addresses consistently feature matching Arabic numerals (plot numbers, building identifiers, pin codes) and Latin street tokens.
2. **Unseen Country Zero-Shot Generalization (France)**:
   - While the training set comprises India and the United States, the test set introduces **France** (259,452 reference entities). French corporate entries feature heavy accentuation (`é`, `è`, `ê`, `ç`, `ô`) and distinct corporate legal suffixes (`SARL`, `SAS`, `EURL`, `SA`). Without NFKD Unicode normalization and suffix harmonization, matching accuracy drops precipitously.
3. **Address Noise & Synonyms**:
   - Heavy abbreviation variations exist across geographic domains: state abbreviations (`CA` $\leftrightarrow$ `California`, `MH` $\leftrightarrow$ `Maharashtra`), street classifications (`Rd` $\leftrightarrow$ `Road`, `Ave` $\leftrightarrow$ `Avenue`, `Blvd` $\leftrightarrow$ `Boulevard`), and suite/floor descriptors.
4. **Strict Singleton Mechanics**:
   - In the training reference set, **5.58%** of `Source 1` entities have zero true matches in S2/S3 (singletons). Under the official Macro $F_{0.5}$ formulation:
     $$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$
     Predicting any false candidate for a true singleton yields $F_{0.5} = 0.0$, whereas correctly predicting an empty string yields $1.0$. Because Precision is weighted with $\frac{1}{\beta^2} = 4\times$ the importance of Recall, conservative high-confidence thresholding is strictly required.

### 2.2 Solution Strategy
- **Approach Type**: Country-Partitioned Multi-Pass Inverted Index Blocking + C++ Accelerated Pairwise Feature Engineering + Asymmetric Precision-Calibrated Gradient Boosted Trees (LightGBM).
- **Core Innovation**:
  1. *Two-Pass Union Indexing*: Separate inverted index postings for tokenized clean names and distinctive address tokens (door/plot numbers, zip codes, locality keywords). This ensures candidate generation never suffers from name displacement in multi-script Indic records.
  2. *Harmonic Mean Compatibility Feature*: An interaction feature $H = \frac{2 \cdot N \cdot A}{N + A + \epsilon}$ combining normalized name similarity ($N$) and address similarity ($A$), which penalizes pairs that match strongly on a generic name but diverge in address. This feature single-handedly provided the highest split gain in the gradient boosted tree (>3.4M Gain).
  3. *Country-by-Country Memory Streaming*: Processes large test sets (France: 1.43M S2/S3 records, US: 4.8M records, India: 4.1M records) sequentially, guaranteeing peak memory usage remains under 8 GB RAM on consumer hardware.

---

## 3. Candidate Generation (Blocking)
To reduce the $1.73 \times 10^6 \times 10.3 \times 10^6 \approx 1.78 \times 10^{13}$ all-pairs Cartesian product space into a tractable set without sacrificing recall:

- **Country Partitioning**: Exact country matching reduces comparison space by $3\times$ with 0% recall loss (verified across all 2.2M ground truth training links).
- **Blocking Keys Used**:
  1. *Pass 1: Exact Clean Name Index*: Directly captures exact matches post accent stripping and legal suffix normalization.
  2. *Pass 2: Name Inverted Index*: Indexed by significant word tokens ($\ge 3$ characters), 2-character alphanumeric tokens, and 4-gram character prefixes. Candidate retrieval uses TF-IDF posting weight scoring:
     $$w(t) = \frac{1}{\log(1 + |P(t)|)}$$
     where $|P(t)|$ is posting list length. The top 25 candidates are retrieved.
  3. *Pass 3: Address & Number Inverted Index*: Indexed by alphanumeric address numbers (`19/1276`, `46D`, `6-2-101/5/C`, 5-digit ZIPs, 6-digit PIN codes) and distinctive locality keywords. The top 25 candidates are retrieved.
- **Candidate Pairs Generated**: Average of 25–40 candidates per S1 entity (~60M total candidate pairs across the test set).
- **How True Matches Were Preserved**: The union of exact name, TF-IDF name pass, and address number pass ensures that even if one modality is corrupted by noise, transliteration, or missing text, the other modality recovers the true candidate pair. Validation blocking recall consistently reached **98.39% - 98.60%**.

---

## 4. Matching Model

### 4.1 Features Extracted (18 Dimensions)
All string similarity metrics are executed via C++ implementations in `RapidFuzz` for maximum throughput (>50,000 candidate pairs/second):

- **Name Similarity**:
  - `n_ratio`: Normalized Levenshtein edit distance.
  - `n_tset`: RapidFuzz `token_set_ratio` (robust to token permutations and duplications).
  - `n_tsort`: RapidFuzz `token_sort_ratio` (alphabetical token alignment).
  - `n_part`: RapidFuzz `partial_ratio` (substring matching for truncated corporate titles).
  - `n_jacc`: Word-level Jaccard similarity.
  - `exact_name`: Binary indicator of exact clean name equality.
- **Address Similarity**:
  - `a_ratio`, `a_tset`, `a_tsort`, `a_part`: Edit and token set ratios on normalized addresses.
  - `a_jacc`: Word-level Jaccard similarity of address tokens.
  - `exact_addr`: Binary indicator of exact clean address equality.
  - `num_match`: Binary flag indicating identical building/plot/PIN number matches.
  - `num_jacc`: Jaccard overlap ratio between extracted address numbers.
- **Interaction & Cross-Modal Signals**:
  - `both_addr`: Indicator that both records have non-empty address fields.
  - `either_missing`: Complement of `both_addr`.
  - `harm`: Harmonic mean compatibility: $\frac{2 \cdot \text{n\_tset} \cdot \text{eff\_a\_score}}{\text{n\_tset} + \text{eff\_a\_score} + 10^{-5}}$.

### 4.2 Model Type & Hyperparameters
- **Model**: LightGBM Classifier (`LGBMClassifier`) with gradient boosting decision trees.
- **Hyperparameters**:
  - `num_leaves`: 127
  - `max_depth`: 8
  - `learning_rate`: 0.06
  - `n_estimators`: 350
  - `subsample`: 0.85
  - `colsample_bytree`: 0.85
  - `n_jobs`: -1 (multi-threaded CPU acceleration)
- **Feature Importance (Top Gain)**:
  1. `harm`: 3,413,752.4
  2. `a_jacc`: 760,908.3
  3. `a_tset`: 631,651.6
  4. `n_part`: 92,346.8
  5. `n_tsort`: 78,830.0
  6. `num_jacc`: 74,357.3
  7. `n_ratio`: 58,129.5

### 4.3 Threshold Selection Method
Because the competition evaluates on **Macro $F_{0.5}$** across all test entities, symmetric 0.5 probability thresholding is suboptimal. We performed a granular grid search $\tau \in [0.35, 0.85]$ on the stratified validation set:
- $\tau = 0.50 \implies \text{Macro } F_{0.5} = 0.9666$
- $\tau = 0.60 \implies \text{Macro } F_{0.5} = 0.9687$
- $\tau = 0.70 \implies \text{Macro } F_{0.5} = \mathbf{0.9703}$ (Optimal)
- $\tau = 0.75 \implies \text{Macro } F_{0.5} = 0.9700$
- $\tau = 0.85 \implies \text{Macro } F_{0.5} = 0.9661$

The optimal threshold $\tau^* = 0.70$ strikes the mathematical balance between high precision and singleton non-contamination.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score**: **0.9703 (97.03%)** on out-of-fold stratified validation, with micro-precision reaching **98.2%** and blocking recall at **98.6%**.
- **Common False Positives (Wrong Merges)**:
  - *Franchise & Campus False Positives*: Distinct legal entities situated within the same tech park or commercial plaza (e.g., "Tata Consultancy Services Ltd" and "Tata Sons Ltd" at the same postal PIN code). Resolved by demanding high name token sort ratio when address similarity is near 1.0.
- **Common False Negatives (Missed Matches)**:
  - *Extreme Cross-Script Corruption*: Indic transliterations where the English name has 0 shared tokens and the address has no street number, relying entirely on phonetic transliteration of a neighborhood name.
- **France Generalization**:
  - The model trained on US & India generalized effectively to France without domain shift degradation due to NFKD accent normalization and language-agnostic token ratio metrics.

---

## 6. Conclusion
By engineering a dedicated two-pass country-partitioned inverted index and pairing it with C++ accelerated string metrics and an asymmetric $F_{0.5}$-calibrated LightGBM model, we constructed a fast, highly accurate, and resource-efficient entity resolution system. The entire inference pipeline runs within strict memory bounds (<8 GB RAM) and completes full inference on all 1.73 million test entities in under two hours.

---

## Appendix

### A. Code Artefacts
The complete runnable source code is organized under `code/business_entity_resolution/`:
- `src/preprocess.py`: Unicode NFKD normalization, legal suffix elimination, address expansion.
- `src/blocking.py`: Country-partitioned multi-pass inverted index blocking engine.
- `src/features.py`: C-accelerated RapidFuzz pairwise feature extraction.
- `src/model.py`: LightGBM model training, threshold calibration, and probability scoring.
- `src/evaluate.py`: Macro $F_{0.5}$ implementation adhering to competition singleton criteria.
- `src/run_pipeline.py`: Production orchestrator supporting streaming inference and TSV emission.
- `requirements.txt`: Pinned Python dependencies.
- `README.md`: Execution manual and reproducibility guide.

**Primary Entry Points**:
```bash
# Full test set inference and submission generation
python code/business_entity_resolution/src/run_pipeline.py --data-dir dataset --output-dir output --cache-dir cache

# Verify submission validity
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

### B. Additional Results & Validation Checks
- Output format verified: 100% compliant with competition specifications (all 1,732,544 test S1 IDs accounted for, candidate sets strictly contain matching predictions, singletons properly empty, tab-separated).
