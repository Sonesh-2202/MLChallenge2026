import pandas as pd
import sys

sys.stdout.reconfigure(encoding='utf-8')

print("Reading train_source1...")
s1 = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
print("Reading ground truth...")
gt = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t')

print(f"Total S1 train: {len(s1)}")
print("Country distribution in S1 train:")
print(s1['country'].value_counts())

print("\nReading test_source1...")
ts1 = pd.read_csv('dataset/test/test_source1.tsv', sep='\t')
print(f"Total S1 test: {len(ts1)}")
print("Country distribution in S1 test:")
print(ts1['country'].value_counts())

# Ground truth statistics
gt['num_matches'] = gt['matched_entity_ids'].apply(lambda x: 0 if pd.isna(x) or str(x).strip()=='' else len(str(x).split(',')))
print("\nGround truth matches per S1:")
print(gt['num_matches'].value_counts().sort_index().head(15))
singletons = (gt['num_matches'] == 0).sum()
print(f"\nSingletons: {singletons:,} ({singletons / len(gt) * 100:.2f}%)")
print(f"Entities with matches: {len(gt) - singletons:,} ({(len(gt) - singletons) / len(gt) * 100:.2f}%)")
print(f"Total true match pairs: {gt['num_matches'].sum():,}")
print(f"Average matches per non-singleton: {gt[gt['num_matches'] > 0]['num_matches'].mean():.2f}")
