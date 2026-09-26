"""
High-Speed 59-Feature Engineering for Candidate Pairs (V9).
Designed for ultra-high precision Business Entity Resolution under Macro F0.5.

Inherits all 43 features from V4 (Features 1-43) and introduces 16 novel
high-discriminative features targeting residual false positives:
  - Roman numeral & project phase consistency (44-45)
  - Acronym & professional credential agreement (46-47)
  - Legal form & entity type compatibility (48-49)
  - IDF-weighted distinctive token penalties (50-55, fit on training fold only)
  - Address sub-unit & street type agreement (56-57)
  - Relative length discrepancies (58-59)
"""
import os
import re
import math
import pickle
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from preprocess import normalize_address
from features_v4 import (
    FEATURE_COLS as FEATURE_COLS_V4,
    NAME_STOP_WORDS, ADDR_STOP_WORDS, INDIC_RE,
    extract_clean_numbers, extract_postal_code, clean_name_v3,
    char_ngrams, jaccard_sets
)

FEATURE_COLS_V9 = FEATURE_COLS_V4 + [
    # Roman numeral & sequence (44-45)
    'roman_numeral_match', 'roman_numeral_mismatch',
    # Acronym & credential (46-47)
    'acronym_credential_match', 'acronym_credential_mismatch',
    # Legal form & entity type (48-49)
    'legal_form_match', 'legal_form_mismatch',
    # IDF-weighted token metrics (50-55)
    'unmatched_idf_sum_s1', 'unmatched_idf_max_s1',
    'unmatched_idf_sum_cand', 'unmatched_idf_max_cand',
    'matched_idf_sum', 'idf_weighted_jaccard',
    # Address sub-components (56-57)
    'unit_suite_match', 'street_type_match',
    # Relative length discrepancies (58-59)
    'token_len_diff_ratio', 'char_len_diff_ratio'
]

FEATURE_COLS = FEATURE_COLS_V9

ROMAN_RE = re.compile(r'\b(i{1,3}|iv|v|vi{1,3}|ix|x|xi{1,3}|xii)\b', re.IGNORECASE)
ACRONYM_RE = re.compile(r'\b[A-Z]{2,4}\b')
UNIT_RE = re.compile(r'\b(?:ste|suite|unit|apt|apartment|fl|floor|bldg|building|room|rm)\s*#?\s*([a-z0-9]+)\b', re.IGNORECASE)

LEGAL_FORMS = {
    'llc': 'llc', 'inc': 'inc', 'corp': 'corp', 'corporation': 'corp', 'incorporated': 'inc',
    'ltd': 'ltd', 'limited': 'ltd', 'pvt': 'pvt', 'private': 'pvt', 'co': 'co', 'company': 'co',
    'sarl': 'sarl', 'sas': 'sas', 'eurl': 'eurl', 'gmbh': 'gmbh', 'sa': 'sa',
    'hospital': 'hosp', 'clinic': 'clinic', 'school': 'school', 'college': 'college',
    'foundation': 'fndn', 'trust': 'trust', 'association': 'assoc', 'services': 'serv'
}

STREET_TYPES = {
    'road': 'rd', 'rd': 'rd', 'street': 'st', 'st': 'st', 'avenue': 'ave', 'ave': 'ave',
    'boulevard': 'blvd', 'blvd': 'blvd', 'lane': 'ln', 'ln': 'ln', 'court': 'ct', 'ct': 'ct',
    'drive': 'dr', 'dr': 'dr', 'way': 'way', 'highway': 'hwy', 'hwy': 'hwy', 'parkway': 'pkwy', 'pkwy': 'pkwy',
    'circle': 'cir', 'cir': 'cir', 'plaza': 'plz', 'square': 'sq', 'sq': 'sq'
}

# Cache for IDF dictionary
_G_IDF_DICT = None
DEFAULT_IDF = 10.0

def get_idf_dict():
    global _G_IDF_DICT
    if _G_IDF_DICT is None:
        src_dir = os.path.dirname(os.path.abspath(__file__))
        cache_path = os.path.abspath(os.path.join(src_dir, '..', '..', '..', 'cache', 'train_token_idf.pkl'))
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                _G_IDF_DICT = pickle.load(f)
        else:
            _G_IDF_DICT = {}
    return _G_IDF_DICT

def compute_pair_features_v9(nc, ac, raw_n1, raw_w1_set,
                             n2, a2, raw_n2,
                             w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
                             first_token1, zip1, has_indic1,
                             idf_dict=None):
    """
    Compute exactly 59 features for a single candidate pair.
    Features 1-43 match V4 exactly.
    Features 44-59 provide rich discrepancy and fine-grained signal.
    """
    if idf_dict is None:
        idf_dict = get_idf_dict()

    # 1. Clean Name metrics (1-16)
    n_ratio = fuzz.ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
    n_tset = fuzz.token_set_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
    n_tsort = fuzz.token_sort_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
    n_part = fuzz.partial_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0

    w2_tokens = [w for w in n2.split() if w not in NAME_STOP_WORDS] if n2 else []
    w2_set = set(w2_tokens)
    n_overlap = len(w1_set & w2_set)
    n_union = len(w1_set | w2_set)
    n_jacc = n_overlap / max(1, n_union) if n_union > 0 else 0.0

    n1_ns = nc.replace(' ', '')
    n2_ns = n2.replace(' ', '')
    n_nospace_ratio = fuzz.ratio(n1_ns, n2_ns) / 100.0 if (n1_ns and n2_ns) else 0.0

    len1, len2 = len(nc), len(n2)
    n_len_ratio = min(len1, len2) / max(1, max(len1, len2)) if (len1 or len2) else 0.0

    nw1, nw2 = len(w1_set), len(w2_set)
    n_token_count_ratio = min(nw1, nw2) / max(1, max(nw1, nw2)) if (nw1 or nw2) else 0.0
    n_containment = n_overlap / max(1, min(nw1, nw2)) if (nw1 and nw2) else 0.0

    n_prefix3 = 1.0 if (len1 >= 3 and len2 >= 3 and nc[:3] == n2[:3]) else 0.0
    n_trigram_jacc = jaccard_sets(n1_ngrams, char_ngrams(n2, 3))
    exact_n = 1.0 if (nc and nc == n2) else 0.0

    first_token2 = w2_tokens[0] if w2_tokens else ''
    n_first_token_match = 1.0 if (first_token1 and first_token1 == first_token2) else 0.0
    n_jw = float(JaroWinkler.similarity(nc, n2)) if (nc and n2) else 0.0

    # 2. Address metrics (17-27)
    has_a2 = bool(a2 and a2 != 'nan')
    both_addr = 1.0 if (has_a1 and has_a2) else 0.0
    either_missing = 1.0 if not (has_a1 and has_a2) else 0.0
    exact_name_missing_addr = 1.0 if (exact_n == 1.0 and either_missing == 1.0) else 0.0

    if both_addr:
        a_ratio = fuzz.ratio(ac, a2) / 100.0
        a_tset = fuzz.token_set_ratio(ac, a2) / 100.0
        a_tsort = fuzz.token_sort_ratio(ac, a2) / 100.0
        a_part = fuzz.partial_ratio(ac, a2) / 100.0

        aw2_set = set(a2.split()) if a2 else set()
        a_overlap = len(aw1_set & aw2_set)
        a_union = len(aw1_set | aw2_set)
        a_jacc = a_overlap / max(1, a_union) if a_union > 0 else 0.0

        alen1, alen2 = len(ac), len(a2)
        a_len_ratio = min(alen1, alen2) / max(1, max(alen1, alen2)) if (alen1 or alen2) else 0.0
        anw1, anw2 = len(aw1_set), len(aw2_set)
        a_containment = a_overlap / max(1, min(anw1, anw2)) if (anw1 and anw2) else 0.0

        a_trigram_jacc = jaccard_sets(a1_ngrams, char_ngrams(a2, 3))
        exact_a = 1.0 if (ac and ac == a2) else 0.0
        a_jw = float(JaroWinkler.similarity(ac, a2))
    else:
        a_ratio = a_tset = a_tsort = a_part = a_jacc = 0.0
        a_len_ratio = a_overlap = a_containment = a_trigram_jacc = exact_a = a_jw = 0.0

    # 3. Number & Postal Code metrics (28-37)
    nums2 = extract_clean_numbers(a2) if has_a2 else set()
    num_common = len(nums1 & nums2)
    has_num1 = len(nums1) > 0
    has_num2 = len(nums2) > 0

    if has_num1 and has_num2:
        num_m = 1.0 if num_common > 0 else 0.0
        num_j = num_common / len(nums1 | nums2)
        num_mismatch = 1.0 if num_common == 0 else 0.0
        num_exact_mismatch = 1.0 if num_common == 0 else 0.0
    elif not has_num1 and not has_num2:
        num_m = 0.5
        num_j = 0.5
        num_mismatch = 0.0
        num_exact_mismatch = 0.0
    else:
        num_m = 0.0
        num_j = 0.0
        num_mismatch = 0.5
        num_exact_mismatch = 0.0

    zip2 = extract_postal_code(a2) if has_a2 else None
    if zip1 and zip2:
        zip_match = 1.0 if zip1 == zip2 else 0.0
        zip_mismatch = 1.0 if zip1 != zip2 else 0.0
    else:
        zip_match = 0.0
        zip_mismatch = 0.0

    if n_ratio > 0 and a_ratio > 0:
        harm = 2.0 * n_ratio * a_ratio / (n_ratio + a_ratio)
    else:
        harm = 0.0

    # 4. V4 Features (38-43)
    raw_n_ratio = fuzz.ratio(raw_n1, raw_n2) / 100.0 if (raw_n1 and raw_n2) else 0.0
    raw_n_tsort = fuzz.token_sort_ratio(raw_n1, raw_n2) / 100.0 if (raw_n1 and raw_n2) else 0.0

    raw_w2_tokens = raw_n2.lower().split() if raw_n2 else []
    raw_w2_set = set(raw_w2_tokens)
    raw_token_diff = float(abs(len(raw_w1_set) - len(raw_w2_set)))
    unmatched_raw = float(len(raw_w2_set - raw_w1_set))

    has_indic2 = bool(INDIC_RE.search(raw_n2)) if raw_n2 else False
    is_indic = 1.0 if (has_indic1 or has_indic2) else 0.0
    num_pin_match = 1.0 if (num_common > 0 and zip_match == 1.0) else 0.0

    # 5. V9 Novel Features (44-59)
    # 44-45: Roman numerals & series
    r1 = set(m.lower() for m in ROMAN_RE.findall(raw_n1)) if raw_n1 else set()
    r2 = set(m.lower() for m in ROMAN_RE.findall(raw_n2)) if raw_n2 else set()
    if r1 and r2:
        roman_match = 1.0 if (r1 == r2) else 0.0
        roman_mismatch = 1.0 if (r1 != r2) else 0.0
    elif not r1 and not r2:
        roman_match = 0.5
        roman_mismatch = 0.0
    else:
        roman_match = 0.0
        roman_mismatch = 0.5

    # 46-47: Acronym & credential match
    acr1 = set(ACRONYM_RE.findall(raw_n1)) if raw_n1 else set()
    acr2 = set(ACRONYM_RE.findall(raw_n2)) if raw_n2 else set()
    if acr1 and acr2:
        acr_match = 1.0 if bool(acr1 & acr2) else 0.0
        acr_mismatch = 1.0 if not bool(acr1 & acr2) else 0.0
    elif not acr1 and not acr2:
        acr_match = 0.5
        acr_mismatch = 0.0
    else:
        acr_match = 0.0
        acr_mismatch = 0.5

    # 48-49: Legal form agreement
    l1 = set(LEGAL_FORMS[t] for t in raw_w1_set if t in LEGAL_FORMS)
    l2 = set(LEGAL_FORMS[t] for t in raw_w2_set if t in LEGAL_FORMS)
    if l1 and l2:
        legal_match = 1.0 if bool(l1 & l2) else 0.0
        legal_mismatch = 1.0 if not bool(l1 & l2) else 0.0
    elif not l1 and not l2:
        legal_match = 0.5
        legal_mismatch = 0.0
    else:
        legal_match = 0.0
        legal_mismatch = 0.5

    # 50-55: IDF-weighted distinctive token metrics
    unmatched_s1 = raw_w1_set - raw_w2_set
    unmatched_cand = raw_w2_set - raw_w1_set
    matched_toks = raw_w1_set & raw_w2_set
    union_toks = raw_w1_set | raw_w2_set

    idf_unmatched_s1_vals = [idf_dict.get(t, DEFAULT_IDF) for t in unmatched_s1]
    idf_unmatched_cand_vals = [idf_dict.get(t, DEFAULT_IDF) for t in unmatched_cand]
    idf_matched_vals = [idf_dict.get(t, DEFAULT_IDF) for t in matched_toks]
    idf_union_vals = [idf_dict.get(t, DEFAULT_IDF) for t in union_toks]

    unmatched_idf_sum_s1 = float(sum(idf_unmatched_s1_vals))
    unmatched_idf_max_s1 = float(max(idf_unmatched_s1_vals, default=0.0))
    unmatched_idf_sum_cand = float(sum(idf_unmatched_cand_vals))
    unmatched_idf_max_cand = float(max(idf_unmatched_cand_vals, default=0.0))
    matched_idf_sum = float(sum(idf_matched_vals))

    sum_union_idf = sum(idf_union_vals)
    idf_weighted_jaccard = (matched_idf_sum / sum_union_idf) if sum_union_idf > 0 else 0.0

    # 56-57: Address unit & street type
    u1_m = UNIT_RE.search(ac) if ac else None
    u2_m = UNIT_RE.search(a2) if a2 else None
    if u1_m and u2_m:
        unit_match = 1.0 if u1_m.group(1).lower() == u2_m.group(1).lower() else 0.0
    elif not u1_m and not u2_m:
        unit_match = 0.5
    else:
        unit_match = 0.0

    st1 = set(STREET_TYPES[t] for t in aw1_set if t in STREET_TYPES) if aw1_set else set()
    st2 = set(STREET_TYPES[t] for t in aw2_set if t in STREET_TYPES) if both_addr else set()
    if st1 and st2:
        street_type_match = 1.0 if bool(st1 & st2) else 0.0
    elif not st1 and not st2:
        street_type_match = 0.5
    else:
        street_type_match = 0.0

    # 58-59: Relative length discrepancies
    max_w = max(len(raw_w1_set), len(raw_w2_set), 1)
    token_len_diff_ratio = raw_token_diff / max_w

    raw_len1, raw_len2 = len(raw_n1) if raw_n1 else 0, len(raw_n2) if raw_n2 else 0
    max_char = max(raw_len1, raw_len2, 1)
    char_len_diff_ratio = abs(raw_len1 - raw_len2) / max_char

    return (
        # V4 Features 1-43
        n_ratio, n_tset, n_tsort, n_part, n_jacc,
        n_nospace_ratio, n_len_ratio, float(n_overlap),
        n_token_count_ratio, n_containment, n_prefix3,
        n_trigram_jacc, exact_n,
        exact_name_missing_addr, n_first_token_match, n_jw,
        a_ratio, a_tset, a_tsort, a_part, a_jacc,
        a_len_ratio, float(a_overlap), a_containment,
        a_trigram_jacc, exact_a, a_jw,
        num_m, num_j, float(num_common), num_mismatch,
        both_addr, either_missing, harm,
        num_exact_mismatch, zip_match, zip_mismatch,
        raw_n_ratio, raw_n_tsort, raw_token_diff,
        unmatched_raw, is_indic, num_pin_match,
        # V9 Novel Features 44-59
        roman_match, roman_mismatch,
        acr_match, acr_mismatch,
        legal_match, legal_mismatch,
        unmatched_idf_sum_s1, unmatched_idf_max_s1,
        unmatched_idf_sum_cand, unmatched_idf_max_cand,
        matched_idf_sum, idf_weighted_jaccard,
        unit_match, street_type_match,
        token_len_diff_ratio, char_len_diff_ratio
    )
