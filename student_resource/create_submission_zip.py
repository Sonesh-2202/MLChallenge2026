#!/usr/bin/env python3
"""
Create the official final submission zip archive: AstraCloud_submission.zip
According to the specification in student_resource/README.md:

AstraCloud_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
"""

import os
import sys
import zipfile
import time

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    zip_path = os.path.join(base_dir, 'AstraCloud_submission.zip')

    if os.path.exists(zip_path):
        os.remove(zip_path)

    print("=" * 70)
    print(" CREATING FINAL SUBMISSION PACKAGE: AstraCloud_submission.zip")
    print("=" * 70)
    t0 = time.time()

    files_to_add = [
        # (source_path, archive_arcname)
        (os.path.join(base_dir, 'Documentation_template.md'), 'Documentation_template.md'),
        (os.path.join(base_dir, 'output', 'matching_results.tsv'), 'output/matching_results.tsv'),
        (os.path.join(base_dir, 'output', 'candidate_pairs.tsv'), 'output/candidate_pairs.tsv'),
        (os.path.join(base_dir, 'code', 'business_entity_resolution', 'README.md'), 'code/business_entity_resolution/README.md'),
        (os.path.join(base_dir, 'code', 'business_entity_resolution', 'requirements.txt'), 'code/business_entity_resolution/requirements.txt'),
    ]

    # Add all files in code/business_entity_resolution/src
    src_dir = os.path.join(base_dir, 'code', 'business_entity_resolution', 'src')
    for root, _, files in os.walk(src_dir):
        if '__pycache__' in root:
            continue
        for f in files:
            if f.endswith('.pyc') or f.endswith('.pyo'):
                continue
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, base_dir)
            files_to_add.append((full_path, rel_path.replace('\\', '/')))

    print(f"Total files to package: {len(files_to_add)}")
    for src, arc in files_to_add:
        size = os.path.getsize(src)
        print(f"  Adding {arc} ({size / 1024 / 1024:.2f} MB) ...", flush=True)

    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for src, arc in files_to_add:
            zf.write(src, arc)

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print("=" * 70)
    print(f" Package created successfully in {time.time()-t0:.1f}s!")
    print(f" Location: {zip_path}")
    print(f" Archive Size: {zip_size_mb:.2f} MB")
    print("=" * 70)

if __name__ == '__main__':
    main()
