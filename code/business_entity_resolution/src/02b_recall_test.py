"""Stage 02b-v2 — recall test against sharded hash index (loads ONE shard at a time).
Saves WORK/block_recall.txt.
"""
import pandas as pd, numpy as np, os, re, unicodedata, joblib, gc
from sklearn.feature_extraction.text import HashingVectorizer
from scipy import sparse
BASE = '/content/drive/MyDrive/AMAZON_ML'
TRAIN = f'{BASE}/data/student_resource/dataset/train'
WORK = f'{BASE}/work'
SUFFIX = r'\b(inc|incorporated|corp|corporation|llc|ltd|limited|pvt|private|llp|pllc|co|company|services|enterprises|solutions|holdings|group|associates|partners)\b\.?'
def norm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return ''
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('ascii')
    s = s.lower(); s = re.sub(r'\.com|\.in|\.org|\.net|\.co\b', ' ', s)
    s = re.sub(r'[^a-z0-9 ]', ' ', s); s = re.sub(SUFFIX, ' ', s)
    return re.sub(r'\s+', ' ', s).strip()
vec = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 5), n_features=2**18, alternate_sign=False, norm='l2')
shards = {}
for sh in ['US', 'IN']:
    X = sparse.load_npz(f'{WORK}/block_matrix_{sh}.npz')
    ids = pd.read_parquet(f'{WORK}/block_ids_{sh}.parquet')['entity_id'].tolist()
    shards[sh] = (X.T.tocsr(), ids)
    print(f'shard {sh}: {X.shape}', flush=True)
    del X; gc.collect()
gt = pd.read_csv(f'{TRAIN}/train_ground_truth.tsv', sep='\t', nrows=3000, dtype=str, keep_default_na=False)
need1 = set(gt['source1_entity_id'])
s1map = {}
for ch in pd.read_csv(f'{TRAIN}/train_source1.tsv', sep='\t', chunksize=200000, dtype=str, keep_default_na=True):
    for _, r in ch[ch['entity_id'].isin(need1)].iterrows():
        s1map[r['entity_id']] = (r['business_name'], r['country'])
    if len(s1map) >= len(need1): break
print(f's1 resolved {len(s1map)}/{len(need1)}', flush=True)
K = 100
hits = {10: 0, 30: 0, 50: 0, 100: 0}; tot = 0; tot_links = 0; found = {10: 0, 30: 0, 50: 0, 100: 0}
for _, row in gt.iterrows():
    if row['source1_entity_id'] not in s1map: continue
    nm, cty = s1map[row['source1_entity_id']]
    sh = 'US' if cty == 'US' else 'IN'
    Xt, ids = shards[sh]
    q = vec.transform([norm(nm)])
    scores = (q @ Xt).toarray().ravel()
    top = np.argpartition(-scores, K)[:K]
    top = top[np.argsort(-scores[top])]
    cand = [ids[i] for i in top]
    truth = [x.strip() for x in str(row['matched_entity_ids']).split(',') if x.strip()]
    if not truth: continue
    tot += 1; tot_links += len(truth)
    for k in hits:
        f = sum(1 for t in truth if t in cand[:k])
        found[k] += f
        if f == len(truth): hits[k] += 1
lines = [f'entities={tot} links={tot_links} K={K}']
for k in [10, 30, 50, 100]:
    lines.append(f'K={k}: link_recall={found[k]/tot_links:.3f} full_entity_recall={hits[k]/tot:.3f} avg_cand={k}')
txt = '\n'.join(lines); print(txt)
open(f'{WORK}/block_recall.txt', 'w').write(txt + '\n')
print('saved WORK/block_recall.txt')
