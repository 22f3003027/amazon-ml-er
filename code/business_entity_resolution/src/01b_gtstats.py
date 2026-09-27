"""EDA v2 stage B — ground-truth FULL stats. Saves WORK/eda_gtstats.txt."""
import pandas as pd, os
from collections import Counter
BASE = '/content/drive/MyDrive/AMAZON_ML'
TRAIN = f'{BASE}/data/student_resource/dataset/train'
WORK = f'{BASE}/work'; os.makedirs(WORK, exist_ok=True)
c = Counter(); n = ns2 = ns3 = gr = 0
for ch in pd.read_csv(f'{TRAIN}/train_ground_truth.tsv', sep='\t', chunksize=200000, dtype=str, keep_default_na=False):
    gr += len(ch)
    for m in ch['matched_entity_ids']:
        m = str(m).strip()
        ids = [x.strip() for x in m.split(',') if x.strip()] if m else []
        c[len(ids)] += 1; n += 1
        ns2 += sum(1 for x in ids if x.startswith('S2-')); ns3 += sum(1 for x in ids if x.startswith('S3-'))
out = f'gt_rows: {gr}\nmatch-count dist: {dict(sorted(c.items()))}\nsingleton rate: {c[0]/n:.3f} | S2 links={ns2:,} S3 links={ns3:,}\n'
print(out); open(f'{WORK}/eda_gtstats.txt', 'w').write(out)
print('saved', f'{WORK}/eda_gtstats.txt')
