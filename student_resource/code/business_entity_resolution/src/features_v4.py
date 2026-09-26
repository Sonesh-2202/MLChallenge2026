"""
High-Speed 43-Feature Engineering for Candidate Pairs (V4).
Designed for ultra-high precision Business Entity Resolution under Macro F0.5.

Additions over V3:
1. raw_n_ratio: Ratio on raw uncleaned business names (catches over-stripped stopwords like 'Corp Services')
2. raw_n_tsort: Token sort ratio on raw uncleaned business names
3. raw_token_diff: Absolute difference in raw token counts
4. unmatched_raw_tokens: Count of raw tokens in cand not present in ref
5. is_indic_script: Detects Indic transliterated scripts so tree doesn't penalize zero Latin similarity
6. num_and_pin_match: Synergistic match of both street number and postal code
"""
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from preprocess import normalize_address

FEATURE_COLS = [
    # Name metrics (1-16)
    'n_ratio', 'n_tset', 'n_tsort', 'n_part', 'n_jacc',
    'n_nospace_ratio', 'n_len_ratio', 'n_token_overlap',
    'n_token_count_ratio', 'n_containment', 'n_prefix3',
    'n_char_trigram_jacc', 'exact_name',
    'exact_name_missing_addr', 'n_first_token_match', 'n_jw',
    # Address metrics (17-27)
    'a_ratio', 'a_tset', 'a_tsort', 'a_part', 'a_jacc',
    'a_len_ratio', 'a_token_overlap', 'a_containment',
    'a_char_trigram_jacc', 'exact_addr', 'a_jw',
    # Number & Cross-field metrics (28-37)
    'num_match', 'num_jacc', 'num_overlap_count', 'num_mismatch',
    'both_addr', 'either_missing', 'harm',
    'num_exact_mismatch', 'zip_match', 'zip_mismatch',
    # V4 Novel Features (38-43)
    'raw_n_ratio', 'raw_n_tsort', 'raw_token_diff',
    'unmatched_raw_tokens', 'is_indic_script', 'num_and_pin_match'
]

# Pure name stopwords
NAME_STOP_WORDS = {
    'the', 'and', 'for', 'of', 'in', 'on', 'at', 'to', 'a', 'an', 'is',
    'llc', 'inc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private', 'co', 'company',
    'llp', 'plc', 'sarl', 'gmbh', 'ag', 'dba', 'sas', 'sa', 'eurl', 'srl', 'sl', 'pty',
    'pllc', 'pc', 'group', 'services', 'industries', 'associates', 'federation',
    'null', 'nan', 'com', 'org', 'net'
}

ADDR_STOP_WORDS = {
    'street', 'road', 'st', 'rd', 'ave', 'avenue', 'lane', 'ln', 'dr', 'drive',
    'court', 'ct', 'city', 'state', 'unit', 'apt', 'apartment', 'floor', 'fl',
    'near', 'opp', 'behind', 'beside', 'null', 'nan',
    'france', 'india', 'united', 'states', 'usa', 'us',
    'rue', 'boulevard', 'blvd', 'chemin', 'impasse', 'route', 'place', 'allee',
    'cours', 'passage', 'quai', 'zone', 'zi', 'za', 'cedex', 'batiment', 'residence',
    'nagar', 'colony', 'complex', 'building', 'plot', 'sector', 'phase', 'bazar',
    'mandi', 'chowk', 'market', 'dist', 'taluk', 'village', 'post', 'po',
    'suite', 'ste', 'pobox', 'box', 'highway', 'hwy', 'parkway', 'pkwy', 'circle', 'cir',
    'plaza', 'center', 'ctr'
}

ZIP_RE = re.compile(r'\b\d{5,6}\b')
NUM_RE = re.compile(r'\b[a-z0-9]+(?:[-/][a-z0-9]+)+\b|\b\d+[a-z]?\b')
INDIC_RE = re.compile(r'[\u0900-\u0D7F]')


def extract_clean_numbers(s):
    if not s or pd.isna(s):
        return set()
    s = str(s).lower()
    raw_nums = NUM_RE.findall(s)
    res = set()
    for n in raw_nums:
        res.add(n)
        stripped = n.lstrip('0')
        if stripped:
            res.add(stripped)
    return res


def extract_postal_code(s):
    if not s or pd.isna(s):
        return None
    matches = ZIP_RE.findall(str(s))
    return matches[-1] if matches else None


def clean_name_v3(name):
    if not name or pd.isna(name):
        return ""
    s = str(name).lower()
    for ext in ['.com', '.org', '.net', '.co.in', '.co', '.in', '.gov', '.edu', '.fr', '.io']:
        if s.endswith(ext):
            s = s[:-len(ext)]
        elif ext + ' ' in s:
            s = s.replace(ext + ' ', ' ')
    from preprocess import normalize_name
    return normalize_name(s)


def char_ngrams(s, n=3):
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def jaccard_sets(s1, s2):
    if not s1 or not s2:
        return 0.0
    u = len(s1 | s2)
    return len(s1 & s2) / u if u > 0 else 0.0


def compute_pair_features_v4(nc, ac, raw_n1, raw_w1_set,
                             n2, a2, raw_n2,
                             w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams,
                             first_token1, zip1, has_indic1):
    """
    Compute exactly 43 features for a single candidate pair.
    Takes both clean names and raw names to measure residual information.
    """
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

    # 4. Novel V4 Features (38-43)
    # Raw name metrics
    raw_n_ratio = fuzz.ratio(raw_n1, raw_n2) / 100.0 if (raw_n1 and raw_n2) else 0.0
    raw_n_tsort = fuzz.token_sort_ratio(raw_n1, raw_n2) / 100.0 if (raw_n1 and raw_n2) else 0.0

    raw_w2_tokens = raw_n2.lower().split() if raw_n2 else []
    raw_w2_set = set(raw_w2_tokens)
    raw_token_diff = float(abs(len(raw_w1_set) - len(raw_w2_set)))
    unmatched_raw = float(len(raw_w2_set - raw_w1_set))

    # Indic script detector
    has_indic2 = bool(INDIC_RE.search(raw_n2)) if raw_n2 else False
    is_indic = 1.0 if (has_indic1 or has_indic2) else 0.0

    # Synergistic number + PIN match
    num_pin_match = 1.0 if (num_common > 0 and zip_match == 1.0) else 0.0

    return (
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
        unmatched_raw, is_indic, num_pin_match
    )
