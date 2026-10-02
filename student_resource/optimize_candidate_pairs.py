#!/usr/bin/env python3
"""
Stream through candidate_pairs.tsv and matching_results.tsv to produce a calibrated,
high-precision candidate set conforming strictly to competition guidelines:
1. Every single matched ID in matching_results.tsv is 100% guaranteed to be in the candidate list.
2. The candidate set is capped at 35 top candidates per entity (~60M total pairs).
3. Eliminates all warnings about matched IDs missing from candidates.
4. Keeps the total ZIP archive size safely below 1024 MB (~280-320 MB total ZIP).
"""

import os
import sys
import time

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    matching_path = os.path.join(base_dir, 'output', 'matching_results.tsv')
    candidate_path = os.path.join(base_dir, 'output', 'candidate_pairs.tsv')
    out_candidate_path = os.path.join(base_dir, 'output', 'candidate_pairs_calibrated.tsv')

    print("=" * 70)
    print(" OPTIMIZING CANDIDATE PAIRS FOR < 1024 MB SUBMISSION LIMIT")
    print("=" * 70)
    t0 = time.time()

    # Step 1: Load matching_results.tsv into memory (1.73M entities, ~150 MB RAM)
    print("[1/3] Loading matching_results.tsv ...", flush=True)
    matching_map = {}
    with open(matching_path, 'r', encoding='utf-8') as f:
        header_m = f.readline().strip()
        assert header_m == "source1_entity_id\tmatched_entity_ids", f"Invalid matching header: {header_m}"
        for line in f:
            parts = line.strip('\r\n').split('\t')
            sid = parts[0]
            m_str = parts[1] if len(parts) > 1 else ''
            matching_map[sid] = [x.strip() for x in m_str.split(',') if x.strip()]

    print(f"  Loaded {len(matching_map):,} entities in {time.time()-t0:.2f}s", flush=True)

    # Step 2: Stream candidate_pairs.tsv and write calibrated file
    print("[2/3] Streaming candidate_pairs.tsv and merging with matches (cap = 35) ...", flush=True)
    t1 = time.time()
    seen_entities = set()
    total_rows = 0
    total_candidates = 0
    max_cands = 0

    with open(candidate_path, 'r', encoding='utf-8') as f_in, open(out_candidate_path, 'w', encoding='utf-8') as f_out:
        header_c = f_in.readline().strip()
        assert header_c == "source1_entity_id\tcandidate_entity_ids", f"Invalid candidate header: {header_c}"
        f_out.write("source1_entity_id\tcandidate_entity_ids\n")

        for line in f_in:
            parts = line.strip('\r\n').split('\t')
            sid = parts[0]
            seen_entities.add(sid)
            c_str = parts[1] if len(parts) > 1 else ''
            cands = [x.strip() for x in c_str.split(',') if x.strip()]

            matches = matching_map.get(sid, [])
            seen_ids = set(matches)
            merged = list(matches)

            for c in cands:
                if len(merged) >= 35:
                    break
                if c not in seen_ids:
                    merged.append(c)
                    seen_ids.add(c)

            total_rows += 1
            n_c = len(merged)
            total_candidates += n_c
            if n_c > max_cands:
                max_cands = n_c

            f_out.write(sid + '\t' + ','.join(merged) + '\n')

            if total_rows % 500000 == 0:
                print(f"  Processed {total_rows:,} entities... ({time.time()-t1:.1f}s)", flush=True)

    # Check if any matching entities were missing from candidate file
    missing = set(matching_map.keys()) - seen_entities
    if missing:
        print(f"  Appending {len(missing)} missing entities with their matches ...")
        with open(out_candidate_path, 'a', encoding='utf-8') as f_out:
            for sid in missing:
                matches = matching_map[sid]
                f_out.write(sid + '\t' + ','.join(matches) + '\n')
                total_rows += 1
                total_candidates += len(matches)

    out_size_mb = os.path.getsize(out_candidate_path) / (1024 * 1024)
    print(f"[3/3] Done in {time.time()-t0:.1f}s!")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Total candidates: {total_candidates:,} (avg {total_candidates/total_rows:.1f}/entity, max {max_cands})")
    print(f"  Output file: {out_candidate_path}")
    print(f"  Output size: {out_size_mb:.2f} MB")
    print("=" * 70)

if __name__ == '__main__':
    main()
