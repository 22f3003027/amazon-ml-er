"""EDA v2 stage A — chunk-scan FULL files: counts + nulls + countries.
Run: python 01a_scan.py. Saves WORK/eda_scan.csv."""
import pandas as pd, os
BASE = '/content/drive/MyDrive/AMAZON_ML'
TRAIN = f'{BASE}/data/student_resource/dataset/train'
TEST = f'{BASE}/data/student_resource/dataset/test'
WORK = f'{BASE}/work'; os.makedirs(WORK, exist_ok=True)
files = {'train_s1': f'{TRAIN}/train_source1.tsv', 'train_s2': f'{TRAIN}/train_source2.tsv',
 'train_s3': f'{TRAIN}/train_source3.tsv', 'test_s1': f'{TEST}/test_source1.tsv',
 'test_s2': f'{TEST}/test_source2.tsv', 'test_s3': f'{TEST}/test_source3.tsv'}
rows = []
for name, p in files.items():
    n = no = na = nc = 0; co = {}
    for ch in pd.read_csv(p, sep='\t', chunksize=200000, dtype=str, keep_default_na=True):
        n += len(ch); no += ch['business_name'].isna().sum(); na += ch['business_address'].isna().sum(); nc += ch['country'].isna().sum()
        for k, v in ch['country'].value_counts(dropna=False).items():
            co[str(k)] = co.get(str(k), 0) + int(v)
    print(f'{name}: rows={n:,} null_name={no:,} null_addr={na:,} null_cty={nc:,} countries={co}', flush=True)
    rows.append({'file': name, 'rows': n, 'null_name': int(no), 'null_addr': int(na), 'null_cty': int(nc), 'countries': str(co)})
pd.DataFrame(rows).to_csv(f'{WORK}/eda_scan.csv', index=False)
print('saved', f'{WORK}/eda_scan.csv')
