# Business Entity Resolution: Engineering Journey to 0.908

**Challenge:** ML Challenge 2026 — Business Entity Resolution  
**Team Name:** AstraCloud  
**Team Members:** Sonesh SK, Swathika Shri, Subhekshana KG  
**Current Milestone Score:** **0.9080 Public Leaderboard** (Top Tier Performance)  

---

## 1. Executive Summary & Score Progression

Over multiple rigorous engineering iterations, our team systematically advanced our entity resolution system from an initial **0.5910** baseline to **0.9080** on the official competition leaderboard:

```
[V1 Baseline: 0.5910] ───────────────> Initial RapidFuzz with Negative Truncation Fallacy
       │
       ▼ (+0.2740)
[V2 Pipeline: 0.8650] ───────────────> Full Country Candidate Space (10.3M Source 2/3 Records)
       │
       ▼ (+0.0050)
[V3 Baseline: 0.8700] ───────────────> 37 RapidFuzz Features + Global Threshold (tau = 0.940)
       │
       ▼ (+0.0293)
[V4 Production: 0.8993] ─────────────> High-Recall Compound Blocking + 43 Raw-Name Residual Features
       │
       ▼ (+0.0047)
[V4.1 Calibration: 0.9040] ──────────> Calibrated Match Capping (Pruned 283,612 Trailing False Merges)
       │
       ▼ (+0.0040)
[V5 SOTA Milestone: 0.9080] ─────────> French Commune Isolation + Dedicated tau=0.980 Pipeline
```

### Complete Scorecard Across Iterations

| Metric | Version 1 (V1) | Version 2 (V2) | Version 3 (V3) | Version 4 (V4) | Version 4.1 (V4.1) | Version 5 (V5 Production) |
|---|---|---|---|---|---|---|
| **Public Leaderboard $F_{0.5}$** | **0.5910** | **0.8650** | **0.8700** | **0.899255** | **0.9040** | **0.9080** |
| **Validation Macro $F_{0.5}$** | 0.9703* (truncated) | ~0.8650 | 0.9071 | 0.9314 | 0.9340 | **~0.9365** |
| **Candidate Blocking Recall** | ~91.5% | 94.80% | 95.40% | **95.70%** | **95.70%** | **95.70%** |
| **Features Extracted** | 12 | 17 | 37 | **43** | **43** | **43** |
| **False Positive Merges** | ~12,000 | ~6,500 | 3,696 | 1,867 | ~1,400 | **~1,200** |
| **France Avg Matches / Entity** | ~4.5 | ~4.5 | 4.38 | 4.42 (inflated) | 3.82 | **3.36 (aligned with GT 3.31)** |
| **Entities with > 6 Matches** | ~60,000 | ~45,000 | 48,000 | 43,978 (France) | 0 (pruned) | **0 (eliminated)** |
| **Test Entities Evaluated** | 1,732,544 | 1,732,544 | 1,732,544 | 1,732,544 | 1,732,544 | **1,732,544 (100% verified)** |

*\*Note: V1 validation score was an artifact caused by evaluating against a truncated pool of 50 candidates per entity. All subsequent versions evaluate against the full 10.3M candidate space.*

---

## 2. Understanding the Competition Metric: Why Macro $F_{0.5}$ is Unique

The evaluation metric in this hackathon is **Per-Source-1 Macro $F_{0.5}$**:

$$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}} = \frac{5 \cdot \text{TP}}{5 \cdot \text{TP} + 4 \cdot \text{FP} + \text{FN}}$$

### Why is this crucial for our team to understand?
1. **Precision is penalized 2× more than Recall**: In business registries, falsely merging two different real-world businesses is twice as damaging as failing to link a business.
2. **Macro-Averaging Across 1.73 Million Entities**: Every Source-1 entity contributes equally ($\frac{1}{N}$) to the final score, regardless of how many matches it has.
3. **The Cluster Explosion Disaster**:
   - Suppose an entity truly has **2 matches**.
   - If our model predicts the true 2 matches: $\text{Precision} = 2/2 = 1.0, \text{Recall} = 2/2 = 1.0 \implies \mathbf{F_{0.5} = 1.000}$.
   - If our model is greedy and predicts **12 matches** (the 2 true matches + 10 false merges):  
     $\text{Precision} = 2/12 = 0.1667, \text{Recall} = 1.0 \implies \mathbf{F_{0.5} = 0.2000}$!
   - **A single greedy prediction collapses that entity's score by 80%!**
4. **Singletons (Entities with No Matches)**:
   - If an entity truly has no matches in Source 2/3, predicting an empty list scores **1.000**.
   - Predicting even a single false match scores **0.000**.

---

## 3. The 5 Core Technical Breakthroughs

### Breakthrough 1: High-Recall Compound Inverted Index Blocking (V4)
- **The Problem:** The Cartesian product space is $1.73 \times 10^6 \times 10.3 \times 10^6 \approx 1.78 \times 10^{13}$ pairs. Baseline blocking missed 4.60% of true matches because of compound words (`western oceanhawk` $\leftrightarrow$ `westernoceanhawk`) and dropped 2-letter uppercase acronyms (`IT`, `AI`, `GE`, `HP`).
- **The Solution (`blocking_v4.py`):**
  - **Squashed compound keys:** Concatenated adjacent word tokens (`nsq:westernoceanhawk`).
  - **Acronym preservation:** Fixed lowercase length checks to retain 2-letter tokens.
  - **Indic address fallbacks:** PIN code and multi-number conjunctions to catch Indic cross-script matches.
- **Result:** Candidate recall reached **95.70%** (US: **97.57%**, India: **92.90%**), recovering **255 previously unrecoverable true matches** while reducing the candidate space by **>99.997%** (~154 candidates/entity).

---

### Breakthrough 2: 43 RapidFuzz & Residual Raw-Name Features (V4)
- **The Problem:** Previous versions stripped legal suffixes (`Services`, `Group`, `Corp`, `Holdings`). Distinct businesses situated at the same commercial building (e.g. `Lucky Deli` vs `Lucky Deli Corp Services`) collapsed to identical names, creating false merges with near-1.0 confidence.
- **The Solution (`features_v4.py`):**
  - Engineered residual raw-name metrics comparing the uncleaned strings: `raw_n_ratio`, `raw_n_tsort`.
  - `raw_token_diff`: Absolute difference in raw token counts.
  - `unmatched_raw_tokens`: Count of raw tokens in the candidate absent from reference entity.
  - `num_and_pin_match`: Synergistic match of both building number and postal code.
- **Empirical Validation:** When we retrained LightGBM, `unmatched_raw_tokens` emerged as the **#7 most important feature across the entire tree ensemble** (over 501,000 gain), cutting validation false positive merges by **49.5%** (from 3,696 down to 1,867)!

---

### Breakthrough 3: Cross-Script Indic Transliteration Bridge
- **The Problem:** In India, reference records in Source 1 were in Latin English, while Source 2 and Source 3 frequently contained Indic script transliterations (Devanagari, Tamil, Telugu, Gujarati). String metrics produced exact zeros (`n_ratio = 0.0`), dropping India's validation score 6.78% below the US.
- **The Solution:** Added `is_indic_script` Unicode detector and PIN-code / address-heavy fallback blocking. This allowed the LightGBM decision trees to learn that when address and number features strongly match, low Latin name similarity on Indic entities should not be penalized.

---

### Breakthrough 4: The French Commune Discovery & Solution (V5)
- **The Problem:** France was an unseen country with zero training data (15.0% of test, 259,452 entities). While India and US averaged ~3.0 matches/entity (matching ground truth 3.31), France exploded to **4.42 matches/entity**, with **43,978 entities (16.95% of France) assigned > 6 matches**!
- **Root Cause Forensic:** French business names frequently feature the town/commune name (`Saint-Herblain Societe SARL`, `La Baule-Escoublac Societe SA`). The model (trained only on US & India) treated the town name and `Societe` as distinctive company names, matching 10–12 completely different businesses in that town.
- **The Solution (`fast_pipeline_v5_france.py`):**
  - Filtered French generic tokens: `sarl`, `sasu`, `sas`, `eurl`, `sci`, `snc`, `sa`, `societe`, `ste`, `fils`, `cie`, `groupe`, `france`, `association`, `lycee`, `college`, `ecole`.
  - Tightened France threshold to **$\tau = 0.980$** and margin to **$\delta = 0.025$**.
  - Hard capped matches at **5**.
- **Result:** Normalized France's average matches to **3.36** and lifted leaderboard score to **0.9080**!

---

### Breakthrough 5: Calibrated Match Capping (V4.1 & V5)
- **The Discovery:** In the 2.2M ground-truth dataset:
  - 96.3% of entities have $\le 5$ matches.
  - 98.9% have $\le 6$ matches.
  - Exactly 0.00% have 12 matches (maximum in ground truth was 11, and $>8$ is under 1%).
- **Action:** Capped France at top-5 and US/India at top-6 matches (ordered strictly by model probability), pruning **283,612 trailing false matches** across 102,358 entities.
- **Impact:** Drove consecutive leaderboard jumps from **0.8992 $\to$ 0.9040 $\to$ 0.9080**.

---

## 4. How to Reproduce the Pipeline (One-Click)

The entire pipeline has been unified into a single, optimized production script: [`student_resource/code/business_entity_resolution/src/pipeline.py`](file:///e:/ML%20Challenge%202026/student_resource/code/business_entity_resolution/src/pipeline.py).

### System Requirements
- Python 3.9+ with `lightgbm`, `rapidfuzz`, `pandas`, `numpy`.
- 8 CPU worker processes, <6 GB RAM usage (zero disk swap thrashing).

### Execution Command
```bash
cd "student_resource/code/business_entity_resolution"
python src/pipeline.py --workers 8 --chunk-size 2000
```

### What the Script Does Automatically:
1. Loads the trained 43-feature LightGBM model (`cache/lgbm_model_v4.pkl`).
2. Checks intermediate partitions in `output/partitions/` and automatically skips already-computed countries.
3. Applies country-calibrated decision thresholds:
   - **France**: French stopwords, $\tau = 0.980$, $\delta = 0.025$, cap = 5.
   - **India**: $\tau = 0.970$, $\delta = 0.030$, cap = 6.
   - **US**: $\tau = 0.970$, $\delta = 0.030$, cap = 6.
4. Merges all partitions into `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
5. Automatically runs `utils/validate_submission.py` to confirm 100% compliance.

---

## 5. Active Workspace Structure (Clean & Pruned)

```
ML Challenge 2026/
├── AstraCloud_submission.zip          # Official 1.44 GB competition submission archive
├── prev_attempts_and_results.md       # Comprehensive historical ledger of all versions
├── team_presentation_summary.md       # This teammate presentation guide
└── student_resource/
    ├── Documentation_template.md      # Official challenge methodology writeup
    ├── output/
    │   ├── matching_results.tsv       # 0.908 SOTA prediction file (upload to portal)
    │   ├── candidate_pairs.tsv        # 3.45 GB blocking candidate set
    │   └── partitions/                # Consolidated active partitions (France, India, US)
    ├── cache/
    │   └── lgbm_model_v4.pkl          # Trained 43-feature LightGBM ensemble
    └── code/business_entity_resolution/
        ├── README.md                  # Code reproduction instructions
        ├── requirements.txt           # Pinned dependencies
        └── src/
            ├── pipeline.py            # Unified production pipeline
            ├── blocking_v4.py         # Multi-pass compound index blocker
            ├── features_v4.py         # 43-dim RapidFuzz feature extraction
            ├── preprocess.py          # Address & text normalization
            └── train_v4.py            # Model training script
```

---

## 6. Recommended Next Steps to Push for 0.93 – 0.95+

1. **Top-1 Fallback for Borderline Singletons:**
   - 112,062 entities (6.47%) currently have 0 matches predicted because their top candidate scored below 0.970.
   - Since ground truth has only ~5.58% singletons, approximately ~15,000 entities are false negatives.
   - If an entity's top candidate scored in $[0.92, 0.97)$ and has an exact street address match, assigning **only that single top-1 match** turns an automatic 0.000 into 1.000!
2. **Dense Semantic Embeddings (MiniLM / BGE):**
   - For candidates surviving the inverted index, compute a cosine similarity using an 8-bit quantized transformer embedding to catch synonym-heavy names (e.g., `Apex Tech Solutions` $\leftrightarrow$ `Apex Technical Services`).
3. **Cross-Encoder Reranker on Top-3 Candidates:**
   - Use a lightweight DeBERTa / MiniLM cross-encoder to rerank the top 3 candidates for entities where the LightGBM margin is tight ($\le 0.015$).
