"""Stage 03a-v3 SAFE.
v2 died: Q(1000x262k) @ X.T(262kx6.2M) = 25GB dense.
v3: per-query matmul against CSC index (25MB each), chunked cdist.
Saves WORK/feat_train.parquet (same schema).
"""
import pandas as pd, numpy as np, os, re, unicodedata, gc, time
from sklearn.feature_extraction.text import HashingVectorizer
from scipy import sparse
from rapidfuzz import fuzz
from rapidfuzz.process import cdist as rf_cdist
BASE = '/content/drive/MyDrive/AMAZON_ML'
TRAIN = f'{BASE}/data/student_resource/dataset/train'
WORK = f'{BASE}/work'
t0 = time.time()
SUFFIX = r'\b(inc|corp|llc|ltd|limited|pvt|private|llp|pllc|co|company|services|enterprises|solutions|holdings|group|associates|partners)\b\.?'
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
N_S1, KN, KA = 12000, 30, 30
gt = pd.read_csv(f'{TRAIN}/train_ground_truth.tsv', sep='\t', nrows=N_S1, dtype=str, keep_default_na=False)
gmap = {r['source1_entity_id']: set(x.strip() for x in str(r['matched_entity_ids']).split(',') if x.strip()) for _, r in gt.iterrows()}
s1map = {}
for ch in pd.read_csv(f'{TRAIN}/train_source1.tsv', sep='\t', chunksize=200000, dtype=str, keep_default_na=True, usecols=['entity_id', 'business_name', 'business_address', 'country']):
    hit = ch[ch['entity_id'].isin(gmap)]
    for _, r in hit.iterrows():
        s1map[r['entity_id']] = (r['business_name'], r['business_address'], r['country'])
print(f's1 {len(s1map)}/{len(gmap)} t={time.time()-t0:.0f}s', flush=True)
cand_rows, need23 = [], set()

for c in ['US', 'IN']:
    sub = [s for s in gmap if s in s1map and ('US' if s1map[s][2] == 'US' else 'IN') == c]
    if not sub: continue
    Xn = sparse.load_npz(f'{WORK}/block_matrix_{c}.npz').tocsc()
    Xa = sparse.load_npz(f'{WORK}/addr_matrix_{c}.npz').tocsc()
    ids = pd.read_parquet(f'{WORK}/block_ids_{c}.parquet')['entity_id'].tolist()
    print(f'{c}: {len(sub)} queries vs {Xn.shape[0]:,} docs t={time.time()-t0:.0f}s', flush=True)
    for qi, s in enumerate(sub):
        qn = nvec.transform([norm(s1map[s][0])]); qa = avec.transform([anorm(s1map[s][1])])
        sn = (qn @ Xn.T).toarray().ravel(); sa = (qa @ Xa.T).toarray().ravel()
        tn = set(np.argpartition(-sn, KN)[:KN].tolist()) if (sn > 0).any() else set()
        ta = set(np.argpartition(-sa, KA)[:KA].tolist()) if (sa > 0).any() else set()
        for j in tn | ta:
            cand_rows.append((s, ids[j])); need23.add(ids[j])
        del qn, qa, sn, sa
        if (qi + 1) % 1000 == 0:
            print(f'  {c} {qi+1}/{len(sub)} t={time.time()-t0:.0f}s', flush=True); gc.collect()
    del Xn, Xa, ids; gc.collect()
print(f'pairs: {len(cand_rows):,} need23: {len(need23):,} t={time.time()-t0:.0f}s', flush=True)
m23 = {}
for p in [f'{TRAIN}/train_source2.tsv', f'{TRAIN}/train_source3.tsv']:
    for ch in pd.read_csv(p, sep='\t', chunksize=200000, dtype=str, keep_default_na=True):
        hit = ch[ch['entity_id'].isin(need23)]
        for _, r in hit.iterrows():
            m23[r['entity_id']] = (r['business_name'], r['business_address'], r['country'])
        if len(m23) >= len(need23): break
    if len(m23) >= len(need23): break
print(f'm23 {len(m23)}/{len(need23)} t={time.time()-t0:.0f}s', flush=True)
s1n = [str(s1map[s][0] or '') for s, _ in cand_rows]
s1a = [str(s1map[s][1] or '') for s, _ in cand_rows]
c2n = [str(m23.get(c, ('', '', ''))[0] or '') for _, c in cand_rows]
c2a = [str(m23.get(c, ('', '', ''))[1] or '') for _, c in cand_rows]
del s1map, m23; gc.collect()
print('rapidfuzz cdist chunked...', flush=True)
N = len(cand_rows); CH = 100000
o_ts, o_r, o_p, o_ats, o_ar, o_ap = [], [], [], [], [], []
for i in range(0, N, CH):
    j = min(N, i + CH)
    o_ts.append(np.diag(rf_cdist(s1n[i:j], c2n[i:j], scorer=fuzz.token_set_ratio, workers=-1)).astype(np.float32))
    o_r.append(np.diag(rf_cdist(s1n[i:j], c2n[i:j], scorer=fuzz.ratio, workers=-1)).astype(np.float32))
    o_p.append(np.diag(rf_cdist(s1n[i:j], c2n[i:j], scorer=fuzz.partial_ratio, workers=-1)).astype(np.float32))
    o_ats.append(np.diag(rf_cdist(s1a[i:j], c2a[i:j], scorer=fuzz.token_set_ratio, workers=-1)).astype(np.float32))
    o_ar.append(np.diag(rf_cdist(s1a[i:j], c2a[i:j], scorer=fuzz.ratio, workers=-1)).astype(np.float32))
    o_ap.append(np.diag(rf_cdist(s1a[i:j], c2a[i:j], scorer=fuzz.partial_ratio, workers=-1)).astype(np.float32))
    print(f'  feats {j}/{N} t={time.time()-t0:.0f}s', flush=True)
del s1n, s1a, c2n, c2a; gc.collect()
import itertools
fn_ts = np.concatenate(o_ts); fn_r = np.concatenate(o_r); fn_p = np.concatenate(o_p)
fa_ts = np.concatenate(o_ats); fa_r = np.concatenate(o_ar); fa_p = np.concatenate(o_ap)
del o_ts, o_r, o_p, o_ats, o_ar, o_ap; gc.collect()
label = np.array([1 if c in gmap[s] else 0 for s, c in cand_rows], dtype=np.int8)
df = pd.DataFrame({'s1': [s for s, _ in cand_rows], 'cid': [c for _, c in cand_rows], 'label': label,
 'country': ['US'] * len(cand_rows),
 'n_tset': fn_ts, 'n_ratio': fn_r, 'n_part': fn_p, 'a_tset': fa_ts, 'a_ratio': fa_r, 'a_part': fa_p,
 'same_cty': np.ones(len(cand_rows), dtype=np.int8), 'miss_n': np.zeros(len(cand_rows), dtype=np.int8), 'miss_a': np.zeros(len(cand_rows), dtype=np.int8)})
df.to_parquet(f'{WORK}/feat_train.parquet', index=False)
print(f'saved {df.shape} pos_rate={df.label.mean():.4f} total_t={time.time()-t0:.0f}s')
print(df.groupby('label')[['n_tset', 'a_tset']].median())
