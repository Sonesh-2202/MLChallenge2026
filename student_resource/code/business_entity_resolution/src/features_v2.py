"""
High-Speed 30-Feature Engineering for Candidate Pairs using C-Accelerated RapidFuzz.
Designed for ultra-high precision Business Entity Resolution under Macro F0.5.
"""
import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from preprocess import normalize_address, STOP_WORDS

FEATURE_COLS = [
    # Name metrics (1-13)
    'n_ratio', 'n_tset', 'n_tsort', 'n_part', 'n_jacc',
    'n_nospace_ratio', 'n_len_ratio', 'n_token_overlap',
    'n_token_count_ratio', 'n_containment', 'n_prefix3',
    'n_char_trigram_jacc', 'exact_name',
    # Address metrics (14-23)
    'a_ratio', 'a_tset', 'a_tsort', 'a_part', 'a_jacc',
    'a_len_ratio', 'a_token_overlap', 'a_containment',
    'a_char_trigram_jacc', 'exact_addr',
    # Number & Cross-field metrics (24-30)
    'num_match', 'num_jacc', 'num_overlap_count', 'num_mismatch',
    'both_addr', 'either_missing', 'harm'
]


def extract_clean_numbers(s):
    """Extract street numbers, plot numbers, handling leading zeros."""
    if not s or pd.isna(s):
        return set()
    s = str(s).lower()
    raw_nums = re.findall(r'\b[a-z0-9]+(?:[-/][a-z0-9]+)+\b|\b\d+[a-z]?\b', s)
    res = set()
    for n in raw_nums:
        res.add(n)
        stripped = n.lstrip('0')
        if stripped:
            res.add(stripped)
    return res


def clean_name_enhanced(name):
    """Enhanced normalization with domain suffix stripping."""
    if not name or pd.isna(name):
        return ""
    s = str(name).lower()
    for ext in ['.com', '.org', '.net', '.co.in', '.co', '.in', '.gov', '.edu', '.fr']:
        if s.endswith(ext):
            s = s[:-len(ext)]
        elif ext + ' ' in s:
            s = s.replace(ext + ' ', ' ')
    from preprocess import normalize_name
    return normalize_name(s)


def char_ngrams(s, n=3):
    """Compute character n-grams of a string."""
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def jaccard_sets(s1, s2):
    """Compute Jaccard similarity of two sets."""
    if not s1 or not s2:
        return 0.0
    u = len(s1 | s2)
    return len(s1 & s2) / u if u > 0 else 0.0


def compute_pair_features(nc, ac, n2, a2, w1_set, aw1_set, nums1, has_a1, n1_ngrams, a1_ngrams):
    """
    Compute exactly 30 features for a single candidate pair.
    Optimized for high-throughput in worker loops.
    """
    # 1. Name metrics
    n_ratio = fuzz.ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
    n_tset = fuzz.token_set_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
    n_tsort = fuzz.token_sort_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0
    n_part = fuzz.partial_ratio(nc, n2) / 100.0 if (nc and n2) else 0.0

    w2_set = set(n2.split()) if n2 else set()
    n_overlap = len(w1_set & w2_set)
    n_union = len(w1_set | w2_set)
    n_jacc = n_overlap / max(1, n_union) if n_union > 0 else 0.0

    # Nospace ratio (handles domain names and squashed words)
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

    # 2. Address metrics
    has_a2 = bool(a2 and a2 != 'nan')
    both_addr = 1.0 if (has_a1 and has_a2) else 0.0

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
        a_len_ratio = min(alen1, alen2) / max(1, max(alen1, alen2))
        anw1, anw2 = len(aw1_set), len(aw2_set)
        a_containment = a_overlap / max(1, min(anw1, anw2)) if (anw1 and anw2) else 0.0
        a_trigram_jacc = jaccard_sets(a1_ngrams, char_ngrams(a2, 3))
        exact_a = 1.0 if ac == a2 else 0.0
    else:
        a_ratio = a_tset = a_tsort = a_part = a_jacc = 0.0
        a_len_ratio = a_overlap = a_containment = a_trigram_jacc = exact_a = 0.0

    # 3. Number & Cross-field metrics
    nums2 = extract_clean_numbers(a2)
    num_common = len(nums1 & nums2) if (nums1 and nums2) else 0
    num_match = 1.0 if num_common > 0 else 0.0
    num_jacc = num_common / max(1, len(nums1 | nums2)) if (nums1 or nums2) else 0.0
    num_mismatch = 1.0 if (nums1 and nums2 and num_common == 0) else 0.0

    eff_a_score = a_tset if both_addr else n_tset
    harm = (2.0 * n_tset * eff_a_score) / (n_tset + eff_a_score + 1e-5)

    return (
        n_ratio, n_tset, n_tsort, n_part, n_jacc,
        n_nospace_ratio, n_len_ratio, float(n_overlap),
        n_token_count_ratio, n_containment, n_prefix3,
        n_trigram_jacc, exact_n,
        a_ratio, a_tset, a_tsort, a_part, a_jacc,
        a_len_ratio, float(a_overlap), a_containment,
        a_trigram_jacc, exact_a,
        num_match, num_jacc, float(num_common), num_mismatch,
        both_addr, 1.0 - both_addr, harm
    )
