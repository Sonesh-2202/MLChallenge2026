import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("Counting S2 and S3 train countries...")
s2_train_counts = {}
for chunk in pd.read_csv('dataset/train/train_source2.tsv', sep='\t', usecols=['country'], chunksize=500000):
    for c, cnt in chunk['country'].value_counts().items():
        s2_train_counts[c] = s2_train_counts.get(c, 0) + cnt
print("S2 train country counts:", s2_train_counts, "Total:", sum(s2_train_counts.values()))

s3_train_counts = {}
for chunk in pd.read_csv('dataset/train/train_source3.tsv', sep='\t', usecols=['country'], chunksize=500000):
    for c, cnt in chunk['country'].value_counts().items():
        s3_train_counts[c] = s3_train_counts.get(c, 0) + cnt
print("S3 train country counts:", s3_train_counts, "Total:", sum(s3_train_counts.values()))

print("\nCounting S2 and S3 test countries...")
s2_test_counts = {}
for chunk in pd.read_csv('dataset/test/test_source2.tsv', sep='\t', usecols=['country'], chunksize=500000):
    for c, cnt in chunk['country'].value_counts().items():
        s2_test_counts[c] = s2_test_counts.get(c, 0) + cnt
print("S2 test country counts:", s2_test_counts, "Total:", sum(s2_test_counts.values()))

s3_test_counts = {}
for chunk in pd.read_csv('dataset/test/test_source3.tsv', sep='\t', usecols=['country'], chunksize=500000):
    for c, cnt in chunk['country'].value_counts().items():
        s3_test_counts[c] = s3_test_counts.get(c, 0) + cnt
print("S3 test country counts:", s3_test_counts, "Total:", sum(s3_test_counts.values()))
