"""Stage 02d — dual-channel recall test: name top-50 + address top-50 merged.
Saves WORK/block_recall_dual.txt. Decides final K + channel weights.
"""
import pandas as pd, numpy as np, os, re, unicodedata, gc
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
def anorm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return ''
    s = str(s).lower(); s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = re.sub(r'\b(null|nan|none)\b', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()
nvec = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 5), n_features=2**18, alternate_sign=False, norm='l2')
avec = HashingVectorizer(analyzer='word', ngram_range=(1, 2), n_features=2**18, alternate_sign=False, norm='l2')
sh = {}
for c in ['US', 'IN']:
    Xn = sparse.load_npz(f'{WORK}/block_matrix_{c}.npz').T.tocsr()
    Xa = sparse.load_npz(f'{WORK}/addr_matrix_{c}.npz').T.tocsr()
    ids = pd.read_parquet(f'{WORK}/block_ids_{c}.parquet')['entity_id'].tolist()
    sh[c] = (Xn, Xa, ids); print(f'{c}: name {Xn.shape} addr {Xa.shape}', flush=True)
    gc.collect()
gt = pd.read_csv(f'{TRAIN}/train_ground_truth.tsv', sep='\t', nrows=3000, dtype=str, keep_default_na=False)
need1 = set(gt['source1_entity_id'])
s1map = {}
for ch in pd.read_csv(f'{TRAIN}/train_source1.tsv', sep='\t', chunksize=200000, dtype=str, keep_default_na=True):
    for _, r in ch[ch['entity_id'].isin(need1)].iterrows():
        s1map[r['entity_id']] = (r['business_name'], r['business_address'], r['country'])
    if len(s1map) >= len(need1): break
KN, KA = 50, 50
st = {'n_only': 0, 'a_only': 0, 'both': 0, 'miss': 0}
tot = tl = 0; full = 0
for _, row in gt.iterrows():
    if row['source1_entity_id'] not in s1map: continue
    nm, ad, cty = s1map[row['source1_entity_id']]
    c = 'US' if cty == 'US' else 'IN'
    Xn, Xa, ids = sh[c]
    qn = nvec.transform([norm(nm)])
    qa = avec.transform([anorm(ad)])
    sn = (qn @ Xn).toarray().ravel(); sa = (qa @ Xa).toarray().ravel()
    tn = set(np.argpartition(-sn, KN)[:KN].tolist()); ta = set(np.argpartition(-sa, KA)[:KA].tolist())
    cand = tn | ta
    cand_ids = {ids[i] for i in cand}
    truth = [x.strip() for x in str(row['matched_entity_ids']).split(',') if x.strip()]
    if not truth: continue
    tot += 1; tl += len(truth)
    f = sum(1 for t in truth if t in cand_ids)
    if f == len(truth): full += 1
    for t in truth:
        try: i = ids.index(t)
        except ValueError: st['miss'] += 1; continue
        inn, ina = i in tn, i in ta
        if inn and ina: st['both'] += 1
        elif inn: st['n_only'] += 1
        elif ina: st['a_only'] += 1
        else: st['miss'] += 1
rec = 1 - st['miss'] / max(tl, 1)
lines = [f'entities={tot} links={tl} KN={KN} KA={KA} avg_cand~{(KN+KA)} (pre-dedup)',
 f'link_recall_dual={rec:.3f} full_entity_recall={full/tot:.3f}',
 f"channel attribution: both={st['both']/tl:.3f} name_only={st['n_only']/tl:.3f} addr_only={st['a_only']/tl:.3f} missed={st['miss']/tl:.3f}"]
txt = '\n'.join(lines); print(txt)
open(f'{WORK}/block_recall_dual.txt', 'w').write(txt + '\n')
print('saved WORK/block_recall_dual.txt')
