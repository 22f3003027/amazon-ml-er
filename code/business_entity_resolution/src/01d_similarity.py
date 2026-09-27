"""EDA v2 cell D — standalone, auto-runnable on Colab.
Run:  python 01d_similarity.py   (after drive mount + BASE set, or edit paths below)
Writes: WORK/eda_truepair_sim.csv + prints medians.
"""
import pandas as pd, statistics, os
from rapidfuzz.fuzz import token_set_ratio
BASE = '/content/drive/MyDrive/AMAZON_ML'
TRAIN = f'{BASE}/data/student_resource/dataset/train'
WORK = f'{BASE}/work'; os.makedirs(WORK, exist_ok=True)
gt = pd.read_csv(f'{TRAIN}/train_ground_truth.tsv', sep='\t', nrows=60, dtype=str, keep_default_na=False)
need1 = set(gt['source1_entity_id']); need23 = set(); pairs = []
for _, r in gt.iterrows():
    ids = [x.strip() for x in str(r['matched_entity_ids']).split(',') if x.strip()]
    if ids:
        pairs.append((r['source1_entity_id'], ids[0])); need23.add(ids[0])
idx = {}
for pth, need in [(f'{TRAIN}/train_source1.tsv', need1), (f'{TRAIN}/train_source2.tsv', need23), (f'{TRAIN}/train_source3.tsv', need23)]:
    for ch in pd.read_csv(pth, sep='\t', chunksize=200000, dtype=str, keep_default_na=True):
        for _, r in ch[ch['entity_id'].isin(need)].iterrows():
            idx[r['entity_id']] = r.to_dict()
rows = []
for a, b in pairs:
    ok = a in idx and b in idx
    na = str(idx[a].get('business_name') or '') if a in idx else ''
    nb = str(idx[b].get('business_name') or '') if b in idx else ''
    aa = str(idx[a].get('business_address') or '') if a in idx else ''
    ab = str(idx[b].get('business_address') or '') if b in idx else ''
    s1 = token_set_ratio(na, nb) if ok else None
    s2 = token_set_ratio(aa, ab) if ok else None
    rows.append({'s1': a, 's2s3': b, 'name_sim': s1, 'addr_sim': s2, 's1_name': na[:100], 'm_name': nb[:100]})
    print(f'{a} <-> {b} | name_sim={s1} addr_sim={s2}')
    print(f'  {na[:70]!r} <-> {nb[:70]!r}')
out = pd.DataFrame(rows); out.to_csv(f'{WORK}/eda_truepair_sim.csv', index=False)
ns = [r['name_sim'] for r in rows if r['name_sim'] is not None]
ad = [r['addr_sim'] for r in rows if r['addr_sim'] is not None]
print(f'n={len(ns)} name_med={statistics.median(ns):.1f} name_mean={statistics.mean(ns):.1f} addr_med={statistics.median(ad):.1f} addr_mean={statistics.mean(ad):.1f}')
print('saved', f'{WORK}/eda_truepair_sim.csv')
