"""EDA v2 stage C — REAL eyeball pairs. Saves WORK/eda_pairs.txt."""
import pandas as pd, os
BASE = '/content/drive/MyDrive/AMAZON_ML'
TRAIN = f'{BASE}/data/student_resource/dataset/train'
WORK = f'{BASE}/work'; os.makedirs(WORK, exist_ok=True)
gt = pd.read_csv(f'{TRAIN}/train_ground_truth.tsv', sep='\t', nrows=15, dtype=str, keep_default_na=False)
need1 = set(gt['source1_entity_id']); need23 = set()
for m in gt['matched_entity_ids']:
    for x in str(m).split(',')[:4]:
        if x.strip():
            need23.add(x.strip())
print(f'need {len(need1)} S1 + {len(need23)} S2/S3', flush=True)
idx = {}
for pth, need in [(f'{TRAIN}/train_source1.tsv', need1), (f'{TRAIN}/train_source2.tsv', need23), (f'{TRAIN}/train_source3.tsv', need23)]:
    for ch in pd.read_csv(pth, sep='\t', chunksize=200000, dtype=str, keep_default_na=True):
        for _, r in ch[ch['entity_id'].isin(need)].iterrows():
            idx[r['entity_id']] = r.to_dict()
        if all(k in idx for k in need):
            break
lines = [f"resolved {sum(k in idx for k in need1)}/{len(need1)} S1, {sum(k in idx for k in need23)}/{len(need23)} S2/S3"]
for _, row in gt.head(15).iterrows():
    lines.append('=' * 110)
    lines.append(f"S1: {row['source1_entity_id']} | {idx.get(row['source1_entity_id'], 'NOT-FOUND')}")
    for m in str(row['matched_entity_ids']).split(',')[:4]:
        if m.strip():
            lines.append(f"  MTCH: {m.strip()} | {idx.get(m.strip(), 'NOT-FOUND')}")
txt = '\n'.join(lines); print(txt)
open(f'{WORK}/eda_pairs.txt', 'w', encoding='utf-8').write(txt)
print('saved', f'{WORK}/eda_pairs.txt')
