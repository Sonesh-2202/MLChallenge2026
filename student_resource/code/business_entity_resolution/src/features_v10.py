"""
=============================================================================
 ULTRA-FAST 67-FEATURE ENGINEERING SUITE (V10 ARCHITECTURE)
=============================================================================
Designed to push Business Entity Resolution performance towards the 0.99 stretch goal.

Inherits all 59 features from V9 and introduces 8 novel deep signals:
  60. name_idf_cosine: Cosine similarity of token IDF weight vectors
  61. addr_idf_cosine: Cosine similarity of address IDF weight vectors
  62. leading_addr_num_match: Exact leading street/building number match vs mismatch
  63. us_state_match: Exact 2-letter US state code match
  64. us_state_mismatch: Explicit 2-letter US state code mismatch
  65. fr_dept_match: Exact 2-digit French Department match from postal code
  66. fr_dept_mismatch: French postal codes from different departments
  67. char_4gram_jacc: Character 4-gram Jaccard similarity for squashed & compound tokens
"""

import os
import re
import math
import pickle
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from preprocess import normalize_address
from features_v4 import (
    FEATURE_COLS as FEATURE_COLS_V4,
    NAME_STOP_WORDS, ADDR_STOP_WORDS, INDIC_RE,
    extract_clean_numbers, extract_postal_code, clean_name_v3,
    char_ngrams, jaccard_sets
)
from features_v9 import (
    FEATURE_COLS_V9,
    ROMAN_RE, ACRONYM_RE, UNIT_RE, LEGAL_FORMS, STREET_TYPES,
    get_idf_dict
)

FEATURE_COLS_V10 = FEATURE_COLS_V9 + [
    'name_idf_cosine',
    'addr_idf_cosine',
    'leading_addr_num_match',
    'us_state_match',
    'us_state_mismatch',
    'fr_dept_match',
    'fr_dept_mismatch',
    'char_4gram_jacc'
]

FEATURE_COLS = FEATURE_COLS_V10

# Pre-compiled regex patterns for V10
LEADING_NUM_RE = re.compile(r'^\s*(\d{1,6})\b')
US_STATE_RE = re.compile(r'\b(AL|AK|AZ|AR|CA|CO|CT|DE|FL|GA|HI|ID|IL|IN|IA|KS|KY|LA|ME|MD|MA|MI|MN|MS|MO|MT|NE|NV|NH|NJ|NM|NY|NC|ND|OH|OK|OR|PA|RI|SC|SD|TN|TX|UT|VT|VA|WA|WV|WI|WY)\b', re.IGNORECASE)
FR_ZIP_RE = re.compile(r'\b(0[1-9]|[1-8][0-9]|9[0-5]|97|98)\d{3}\b')

def extract_leading_number(addr):
    if not addr or addr == 'nan':
        return None
    m = LEADING_NUM_RE.search(addr)
    return m.group(1) if m else None

def extract_us_state(addr):
    if not addr or addr == 'nan':
        return None
    m = US_STATE_RE.search(addr)
    return m.group(1).upper() if m else None

def extract_fr_dept(addr):
    if not addr or addr == 'nan':
        return None
    m = FR_ZIP_RE.search(addr)
    return m.group(1) if m else None

FR_STREET_STOPWORDS = {
    'rue', 'avenue', 'av', 'boulevard', 'blvd', 'chemin', 'route', 'impasse',
    'place', 'allee', 'cours', 'passage', 'quai', 'voie', 'square', 'rond-point',
    'residence', 'batiment', 'france', 'cedex'
}

def extract_street_tokens(addr):
    if not addr or addr == 'nan':
        return None
    tokens = [w for w in re.findall(r'[a-zA-Z\u00C0-\u017F]+', addr.lower()) if len(w) >= 3 and w not in FR_STREET_STOPWORDS]
    return set(tokens) if tokens else None

def street_names_compatible(st1, st2):
    if not st1 or not st2:
        return True
    if st1 & st2:
        return True
    s1_str = ' '.join(sorted(st1))
    s2_str = ' '.join(sorted(st2))
    return JaroWinkler.similarity(s1_str, s2_str) >= 0.85

def compute_idf_cosine(tokens1, tokens2, idf_dict, default_idf=4.0):
    if not tokens1 or not tokens2:
        return 0.0
    shared = tokens1 & tokens2
    if not shared:
        return 0.0
    dot_prod = sum((idf_dict.get(t, default_idf) ** 2) for t in shared)
    norm1 = math.sqrt(sum((idf_dict.get(t, default_idf) ** 2) for t in tokens1))
    norm2 = math.sqrt(sum((idf_dict.get(t, default_idf) ** 2) for t in tokens2))
    if norm1 > 0 and norm2 > 0:
        return float(dot_prod / (norm1 * norm2))
    return 0.0

def compute_pair_features_v10(nc, ac, raw_n1, raw_w1_set,
                              n2, a2, raw_n2,
                              w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
                              first_token1, zip1, has_indic1,
                              lead_num1=None, us_state1=None, fr_dept1=None, n1_4grams=None,
                              idf_dict=None):
    """
    Compute exactly 67 features for a single candidate pair.
    Features 1-59 match V9 exactly.
    Features 60-67 add cutting-edge geographic, numeric, and distributional signals.
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
        aw2_set = set()

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

    # 5. V9 Features (44-59)
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

    unmatched_s1 = raw_w1_set - raw_w2_set
    unmatched_cand = raw_w2_set - raw_w1_set
    matched_toks = raw_w1_set & raw_w2_set
    union_toks = raw_w1_set | raw_w2_set

    def_idf = 4.0
    unmatched_idf_sum_s1 = sum(idf_dict.get(t, def_idf) for t in unmatched_s1)
    unmatched_idf_max_s1 = max((idf_dict.get(t, def_idf) for t in unmatched_s1), default=0.0)
    unmatched_idf_sum_cand = sum(idf_dict.get(t, def_idf) for t in unmatched_cand)
    unmatched_idf_max_cand = max((idf_dict.get(t, def_idf) for t in unmatched_cand), default=0.0)
    matched_idf_sum = sum(idf_dict.get(t, def_idf) for t in matched_toks)
    union_idf_sum = sum(idf_dict.get(t, def_idf) for t in union_toks)
    idf_weighted_jaccard = matched_idf_sum / max(1.0, union_idf_sum)

    u1 = UNIT_RE.findall(ac) if ac else []
    u2 = UNIT_RE.findall(a2) if a2 else []
    if u1 and u2:
        unit_match = 1.0 if (set(u1) == set(u2)) else 0.0
    elif not u1 and not u2:
        unit_match = 0.5
    else:
        unit_match = 0.0

    st1 = set(STREET_TYPES[w] for w in aw1_set if w in STREET_TYPES)
    st2 = set(STREET_TYPES[w] for w in aw2_set if w in STREET_TYPES)
    if st1 and st2:
        street_type_match = 1.0 if bool(st1 & st2) else 0.0
    elif not st1 and not st2:
        street_type_match = 0.5
    else:
        street_type_match = 0.0

    t_len1, t_len2 = len(raw_w1_tokens if 'raw_w1_tokens' in locals() else raw_w1_set), len(raw_w2_set)
    token_len_diff_ratio = abs(t_len1 - t_len2) / max(1, max(t_len1, t_len2))
    c_len1, c_len2 = len(raw_n1), len(raw_n2)
    char_len_diff_ratio = abs(c_len1 - c_len2) / max(1, max(c_len1, c_len2))

    # 6. Novel V10 Features (60-67)
    # 60: Name IDF Cosine
    name_idf_cos = compute_idf_cosine(raw_w1_set, raw_w2_set, idf_dict)

    # 61: Address IDF Cosine
    addr_idf_cos = compute_idf_cosine(aw1_set, aw2_set, idf_dict) if both_addr else 0.0

    # 62: Leading Address Number Match
    if lead_num1 is None:
        lead_num1 = extract_leading_number(ac)
    lead_num2 = extract_leading_number(a2) if has_a2 else None
    if lead_num1 and lead_num2:
        lead_num_match = 1.0 if lead_num1 == lead_num2 else -1.0
    elif not lead_num1 and not lead_num2:
        lead_num_match = 0.5
    else:
        lead_num_match = 0.0

    # 63-64: US State Match & Mismatch
    if us_state1 is None:
        us_state1 = extract_us_state(ac)
    us_state2 = extract_us_state(a2) if has_a2 else None
    if us_state1 and us_state2:
        us_state_match = 1.0 if us_state1 == us_state2 else 0.0
        us_state_mismatch = 1.0 if us_state1 != us_state2 else 0.0
    elif not us_state1 and not us_state2:
        us_state_match = 0.5
        us_state_mismatch = 0.0
    else:
        us_state_match = 0.0
        us_state_mismatch = 0.0

    # 65-66: French Department Match & Mismatch
    if fr_dept1 is None:
        fr_dept1 = extract_fr_dept(ac)
    fr_dept2 = extract_fr_dept(a2) if has_a2 else None
    if fr_dept1 and fr_dept2:
        fr_dept_match = 1.0 if fr_dept1 == fr_dept2 else 0.0
        fr_dept_mismatch = 1.0 if fr_dept1 != fr_dept2 else 0.0
    elif not fr_dept1 and not fr_dept2:
        fr_dept_match = 0.5
        fr_dept_mismatch = 0.0
    else:
        fr_dept_match = 0.0
        fr_dept_mismatch = 0.0

    # 67: Character 4-gram Jaccard
    if n1_4grams is None:
        n1_4grams = char_ngrams(nc, 4)
    char_4gram_jacc = jaccard_sets(n1_4grams, char_ngrams(n2, 4)) if (nc and n2) else 0.0

    return (
        # 1-16: Clean Name
        n_ratio, n_tset, n_tsort, n_part, n_overlap, n_union, n_jacc,
        n_nospace_ratio, n_len_ratio, n_token_count_ratio, n_containment,
        n_prefix3, n_trigram_jacc, exact_n, n_first_token_match, n_jw,
        # 17-27: Address
        both_addr, either_missing, exact_name_missing_addr,
        a_ratio, a_tset, a_tsort, a_part, a_jacc, a_len_ratio, a_containment,
        a_trigram_jacc,
        # 28-37: Numbers & ZIP
        num_common, num_m, num_j, num_mismatch, num_exact_mismatch,
        has_num1, has_num2, zip_match, zip_mismatch, harm,
        # 38-43: V4 Features
        raw_n_ratio, raw_n_tsort, raw_token_diff, unmatched_raw,
        is_indic, num_pin_match,
        # 44-59: V9 Features
        roman_match, roman_mismatch,
        acr_match, acr_mismatch,
        legal_match, legal_mismatch,
        unmatched_idf_sum_s1, unmatched_idf_max_s1,
        unmatched_idf_sum_cand, unmatched_idf_max_cand,
        matched_idf_sum, idf_weighted_jaccard,
        unit_match, street_type_match,
        token_len_diff_ratio, char_len_diff_ratio,
        # 60-67: V10 Features
        name_idf_cos, addr_idf_cos,
        lead_num_match,
        us_state_match, us_state_mismatch,
        fr_dept_match, fr_dept_mismatch,
        char_4gram_jacc
    )
