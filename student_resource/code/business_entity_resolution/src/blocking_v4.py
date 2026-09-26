"""
High-Recall Inverted Index Blocking Pipeline (V4 Production).
Improves candidate recall by targeting squashed compound words,
2-letter acronym preservation, and Indic transliterated records.
"""
import heapq
import numpy as np
import pandas as pd
from features_v4 import (
    NAME_STOP_WORDS, ADDR_STOP_WORDS,
    extract_clean_numbers, extract_postal_code
)

STOP_2LETTER = {'to', 'in', 'on', 'at', 'of', 'by', 'an', 'as', 'is', 'it', 'or', 'so', 'we', 'he', 'my', 'up', 'no'}


def extract_keys_v4(nc, ac):
    n_keys, a_keys, c_keys = set(), set(), set()
    raw_n_words = nc.split()
    n_tokens = []

    for w in raw_n_words:
        if w not in NAME_STOP_WORDS:
            if len(w) >= 3:
                n_tokens.append(w)
            elif len(w) == 2 and w not in STOP_2LETTER and w.isalnum():
                n_tokens.append(w)

    nums = extract_clean_numbers(ac)
    a_tokens = [w for w in ac.split() if w not in ADDR_STOP_WORDS and len(w) >= 3]
    pin = extract_postal_code(ac)

    # 1. Name keys
    for t in n_tokens:
        if len(t) >= 4:
            n_keys.add('n:' + t)
            n_keys.add('ng:' + t[:4])
            n_keys.add('pfx:' + t[:3])
        elif len(t) == 3:
            n_keys.add('n3:' + t)
        elif len(t) == 2:
            n_keys.add('n2:' + t)

    # 2. Address keys
    for num in nums:
        a_keys.add('num:' + num)
    for at in a_tokens:
        a_keys.add('a:' + at)
    if pin:
        a_keys.add('pin:' + pin)

    # 3. Compound name keys & squashed compound keys
    for i in range(len(n_tokens)):
        for j in range(i + 1, min(i + 4, len(n_tokens))):
            t1, t2 = n_tokens[i], n_tokens[j]
            pair = tuple(sorted([t1, t2]))
            c_keys.add(f'np:{pair[0]}_{pair[1]}')
            # Squashed compound key (e.g. westernoceanhawk)
            c_keys.add(f'nsq:{t1}{t2}')
            c_keys.add(f'nsq:{t2}{t1}')

    # 4. Name + number cross keys
    for t in n_tokens[:3]:
        for num in nums:
            c_keys.add(f'n_num:{t}_{num}')

    # 5. Number + address token cross keys
    for num in nums:
        for at in a_tokens[:3]:
            c_keys.add(f'num_a:{num}_{at}')

    # 6. Name + address token cross keys
    target_a = list(dict.fromkeys(a_tokens[:2] + a_tokens[-2:])) if len(a_tokens) > 2 else a_tokens
    for t in n_tokens[:3]:
        for at in target_a:
            c_keys.add(f'n_a:{t}_{at}')

    # 7. PIN code + address token cross keys (vital for Indic script records where name doesn't match!)
    if pin:
        for at in a_tokens[:3]:
            c_keys.add(f'pin_a:{pin}_{at}')
        for num in nums:
            c_keys.add(f'pin_num:{pin}_{num}')

    # 8. Multi-number conjunction (for Indian door/plot numbers like 29-13-21)
    if len(nums) >= 2:
        sorted_nums = sorted(nums)[:4]
        for i in range(len(sorted_nums)):
            for j in range(i + 1, len(sorted_nums)):
                c_keys.add(f'num2:{sorted_nums[i]}_{sorted_nums[j]}')

    # 9. Single word character n-grams (lowered threshold from 8 to 6 chars)
    if len(n_tokens) == 1 and len(n_tokens[0]) >= 6:
        single_word = n_tokens[0]
        for i in range(min(12, len(single_word) - 3)):
            n_keys.add('ng:' + single_word[i:i+4])

    return n_keys, a_keys, c_keys


def build_country_index_v4(s23_df, s1_n_keys, s1_a_keys, s1_c_keys, s1_names,
                           max_p_comp=800, max_p_single=600):
    exact_idx, n_idx, a_idx, c_idx = {}, {}, {}, {}
    for r in s23_df.itertuples(index=False):
        eid, nc, ac = r.entity_id, r.name_clean, r.addr_clean
        if nc and nc in s1_names:
            exact_idx.setdefault(nc, []).append(eid)
        nk, ak, ck = extract_keys_v4(nc, ac)
        for k in nk:
            if k in s1_n_keys:
                lst = n_idx.setdefault(k, [])
                if len(lst) < max_p_single: lst.append(eid)
        for k in ak:
            if k in s1_a_keys:
                lst = a_idx.setdefault(k, [])
                if len(lst) < max_p_single: lst.append(eid)
        for k in ck:
            if k in s1_c_keys:
                lst = c_idx.setdefault(k, [])
                if len(lst) < max_p_comp: lst.append(eid)
    return {'exact': exact_idx, 'name': n_idx, 'addr': a_idx, 'comp': c_idx}


def retrieve_candidates_v4(nc, ac, idx, comp_k=70, name_k=45, addr_k=45, exact_k=100):
    cands = set()
    exact_idx = idx['exact']
    c_idx = idx['comp']
    n_idx = idx['name']
    a_idx = idx['addr']

    if nc and nc in exact_idx:
        cands.update(exact_idx[nc][:exact_k])

    nk, ak, ck = extract_keys_v4(nc, ac)

    if ck:
        c_scores = {}
        for k in ck:
            postings = c_idx.get(k)
            if postings:
                w = 1.0 / np.log1p(len(postings))
                for mid in postings:
                    c_scores[mid] = c_scores.get(mid, 0.0) + w
        if c_scores:
            cands.update(m for m, _ in heapq.nlargest(comp_k, c_scores.items(), key=lambda x: x[1]))

    if nk:
        n_scores = {}
        for k in nk:
            postings = n_idx.get(k)
            if postings:
                w = 1.0 / np.log1p(len(postings))
                for mid in postings:
                    n_scores[mid] = n_scores.get(mid, 0.0) + w
        if n_scores:
            cands.update(m for m, _ in heapq.nlargest(name_k, n_scores.items(), key=lambda x: x[1]))

    if ak:
        a_scores = {}
        for k in ak:
            postings = a_idx.get(k)
            if postings:
                w = 1.0 / np.log1p(len(postings))
                for mid in postings:
                    a_scores[mid] = a_scores.get(mid, 0.0) + w
        if a_scores:
            cands.update(m for m, _ in heapq.nlargest(addr_k, a_scores.items(), key=lambda x: x[1]))

    return cands
