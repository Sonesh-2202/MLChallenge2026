"""
Data loading and preprocessing utilities for Business Entity Resolution.
"""
import os
import re
import unicodedata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

LEGAL_SUFFIXES = {
    'llc', 'l.l.c.', 'inc', 'inc.', 'incorporated',
    'corp', 'corp.', 'corporation', 'ltd', 'ltd.', 'limited',
    'pvt', 'pvt.', 'private', 'co', 'co.', 'company',
    'llp', 'l.l.p.', 'plc', 'p.l.c.', 'sarl', 's.a.r.l.',
    'gmbh', 'ag', 'dba', 'd.b.a.', 'sas', 'sa', 'eurl',
    'srl', 'sl', 'pty', 'ngo', 'assn', 'association',
    'pllc', 'pc', 'group', 'services', 'industries', 'associates', 'federation'
}

STOP_WORDS = {
    'the', 'and', 'for', 'of', 'in', 'on', 'at', 'to', 'a', 'an', 'is',
    'llc', 'inc', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private', 'co', 'company',
    'llp', 'plc', 'sarl', 'gmbh', 'ag', 'dba', 'sas', 'sa', 'eurl', 'srl', 'sl', 'pty',
    'pllc', 'pc', 'group', 'services', 'industries', 'associates', 'federation',
    'street', 'road', 'st', 'rd', 'ave', 'avenue', 'lane', 'ln', 'dr', 'drive',
    'court', 'ct', 'city', 'state', 'unit', 'apt', 'apartment', 'floor', 'fl',
    'near', 'opp', 'behind', 'beside', 'null', 'nan', 'com', 'org', 'net',
    # Country names
    'france', 'india', 'united', 'states', 'usa', 'us',
    # French common address words
    'rue', 'boulevard', 'blvd', 'chemin', 'impasse', 'route', 'place', 'allee',
    'cours', 'passage', 'quai', 'zone', 'zi', 'za', 'cedex', 'batiment', 'residence',
    'nord', 'sud', 'est', 'ouest', 'hauts', 'pays', 'nouvelle', 'aquitaine', 'ile',
    'grand', 'loire', 'bretagne', 'normandie', 'occitanie', 'rhone', 'alpes', 'provence',
    # Indian common locality tokens
    'nagar', 'colony', 'complex', 'building', 'plot', 'sector', 'phase', 'bazar',
    'mandi', 'chowk', 'market', 'dist', 'taluk', 'village', 'post', 'po',
    # US common words
    'suite', 'ste', 'pobox', 'box', 'highway', 'hwy', 'parkway', 'pkwy', 'circle', 'cir',
    'plaza', 'center', 'ctr'
}

ADDR_MAP = {
    r'\brd\b': 'road', r'\bst\b': 'street', r'\bave\b': 'avenue',
    r'\bdr\b': 'drive', r'\bln\b': 'lane', r'\bct\b': 'court',
    r'\bblvd\b': 'boulevard', r'\bpkwy\b': 'parkway', r'\bhwy\b': 'highway',
    r'\bapt\b': 'apartment', r'\bste\b': 'suite', r'\bfl\b': 'floor',
    r'\bpo box\b': 'pobox',
    # US States
    r'\bil\b': 'illinois', r'\bca\b': 'california', r'\btx\b': 'texas',
    r'\bny\b': 'new york', r'\bfl\b': 'florida', r'\boh\b': 'ohio',
    r'\bpa\b': 'pennsylvania', r'\bmi\b': 'michigan', r'\bnc\b': 'north carolina',
    r'\bga\b': 'georgia', r'\bva\b': 'virginia', r'\btn\b': 'tennessee',
    r'\bct\b': 'connecticut', r'\bmo\b': 'missouri', r'\baz\b': 'arizona',
    # India States & Cities
    r'\bmh\b': 'maharashtra', r'\bup\b': 'uttar pradesh', r'\bgj\b': 'gujarat',
    r'\brj\b': 'rajasthan', r'\btg\b': 'telangana', r'\btn\b': 'tamil nadu',
    r'\bwb\b': 'west bengal', r'\bka\b': 'karnataka', r'\bdl\b': 'delhi',
    r'\bcalcutta\b': 'kolkata', r'\bbengaluru\b': 'bangalore', r'\bbombay\b': 'mumbai',
    r'\bmadras\b': 'chennai', r'\bodisha\b': 'orissa',
}


# Precompiled patterns for ultra-high-throughput normalization
CLEAN_NONWORD_RE = re.compile(r'[^\w\s]')
CLEAN_SPACE_RE = re.compile(r'\s+')
# Sort keys by length descending to match multi-word phrases first (e.g. 'po box')
ADDR_KEYS_SORTED = sorted(ADDR_MAP.keys(), key=lambda k: len(k.replace(r'\b', '')), reverse=True)
ADDR_RE = re.compile('|'.join(ADDR_KEYS_SORTED))
ADDR_LOOKUP = {k.replace(r'\b', ''): v for k, v in ADDR_MAP.items()}


def strip_accents(s):
    """Strip accents using Unicode NFKD decomposition."""
    if not s or pd.isna(s):
        return ""
    return ''.join(c for c in unicodedata.normalize('NFD', str(s)) if unicodedata.category(c) != 'Mn').lower()


def normalize_name(name):
    """Lowercase, strip accents, punctuation, and common legal suffixes."""
    if not name or pd.isna(name):
        return ""
    s = strip_accents(name)
    s = CLEAN_NONWORD_RE.sub(' ', s)
    tokens = [t for t in s.split() if t not in LEGAL_SUFFIXES]
    return ' '.join(tokens)


def normalize_address(addr):
    """Lowercase, strip accents, expand abbreviations, clean spacing."""
    if not addr or pd.isna(addr) or str(addr).strip().lower() in {'nan', 'null'}:
        return ""
    s = strip_accents(addr)
    s = ADDR_RE.sub(lambda m: ADDR_LOOKUP.get(m.group(0), m.group(0)), s)
    s = CLEAN_NONWORD_RE.sub(' ', s)
    return CLEAN_SPACE_RE.sub(' ', s).strip()


def extract_numbers(s):
    """Extract street/building/plot numbers and alphanumeric codes (e.g. 19/1276, 46D, 6-2-101/5/C)."""
    if not s or pd.isna(s):
        return set()
    s = str(s).lower()
    nums = re.findall(r'\b[a-z0-9]+(?:[-/][a-z0-9]+)+\b|\b\d+[a-z]?\b', s)
    return {n for n in nums if len(n) >= 2 or n.isdigit()}


def add_normalized_columns(df):
    """Add name_clean and addr_clean columns to dataframe."""
    df['name_clean'] = df['business_name'].apply(normalize_name)
    df['addr_clean'] = df['business_address'].apply(normalize_address)
    return df


def load_sources(data_dir, split='train'):
    """Load source 1/2/3 TSV files for train or test."""
    prefix = split
    s1 = pd.read_csv(os.path.join(data_dir, split, f'{prefix}_source1.tsv'),
                     sep='\t', dtype=str).fillna({'business_name': '', 'business_address': '', 'country': ''})
    s2 = pd.read_csv(os.path.join(data_dir, split, f'{prefix}_source2.tsv'),
                     sep='\t', dtype=str).fillna({'business_name': '', 'business_address': '', 'country': ''})
    s3 = pd.read_csv(os.path.join(data_dir, split, f'{prefix}_source3.tsv'),
                     sep='\t', dtype=str).fillna({'business_name': '', 'business_address': '', 'country': ''})
    return s1, s2, s3


def load_ground_truth(data_dir):
    """Load training ground truth TSV."""
    return pd.read_csv(os.path.join(data_dir, 'train', 'train_ground_truth.tsv'), sep='\t')


def create_validation_split(s1, gt, val_frac=0.2, seed=42):
    """
    Split S1 entities into train/val stratified by country.
    Returns (s1_train, s1_val, gt_train, gt_val).
    """
    train_idx, val_idx = train_test_split(
        np.arange(len(s1)), test_size=val_frac,
        stratify=s1['country'], random_state=seed
    )
    s1_train = s1.iloc[train_idx].reset_index(drop=True)
    s1_val = s1.iloc[val_idx].reset_index(drop=True)

    train_ids = set(s1_train['entity_id'])
    val_ids = set(s1_val['entity_id'])

    gt_train = gt[gt['source1_entity_id'].isin(train_ids)].reset_index(drop=True)
    gt_val = gt[gt['source1_entity_id'].isin(val_ids)].reset_index(drop=True)

    return s1_train, s1_val, gt_train, gt_val
