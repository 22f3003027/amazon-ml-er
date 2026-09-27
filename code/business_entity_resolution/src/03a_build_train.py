"""Stage 03a-v2 FAST — same outputs, ~15 min not 6 hrs.
Bottlenecks fixed: (1) N_S1 40k->12k (~700k pairs, plenty for LGBM),
(2) batched sparse matmul (1000 queries/block, not one-by-one),
(3) vectorized top-K via argpartition on dense block,
(4) RapidFuzz process.cdist for features (C-level, not Python loop).
Saves WORK/feat_train.parquet (same schema as v1).
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
N_S1, KN, KA, BATCH = 12000, 30, 30, 1000
gt = pd.read_csv(f'{TRAIN}/train_ground_truth.tsv', sep='\t', nrows=N_S1, dtype=str, keep_default_na=False)
gmap = {r['source1_entity_id']: set(x.strip() for x in str(r['matched_entity_ids']).split(',') if x.strip()) for _, r in gt.iterrows()}
s1ids = list(gmap.keys())
s1map = {}
for ch in pd.read_csv(f'{TRAIN}/train_source1.tsv', sep='\t', chunksize=200000, dtype=str, keep_default_na=True, usecols=['entity_id', 'business_name', 'business_address', 'country']):
    hit = ch[ch['entity_id'].isin(gmap)]
    for _, r in hit.iterrows():
        s1map[r['entity_id']] = (r['business_name'], r['business_address'], r['country'])
print(f's1 {len(s1map)}/{len(gmap)} t={time.time()-t0:.0f}s', flush=True)
cand_rows, need23 = [], set()
for c in ['US', 'IN']:
    sub = [s for s in s1ids if s in s1map and ('US' if s1map[s][2] == 'US' else 'IN') == c]
    if not sub: continue
    Xn = sparse.load_npz(f'{WORK}/block_matrix_{c}.npz')
    Xa = sparse.load_npz(f'{WORK}/addr_matrix_{c}.npz')
    ids = pd.read_parquet(f'{WORK}/block_ids_{c}.parquet')['entity_id'].tolist()
    print(f'{c}: {len(sub)} queries vs {Xn.shape[0]:,} docs', flush=True)
    for i in range(0, len(sub), BATCH):
        b = sub[i:i+BATCH]
        Qn = nvec.transform([norm(s1map[s][0]) for s in b])
        Qa = avec.transform([anorm(s1map[s][1]) for s in b])
        Sn = (Qn @ Xn.T).toarray(); Sa = (Qa @ Xa.T).toarray()
        del Qn, Qa; gc.collect()
        Tn = np.argpartition(-Sn, KN, axis=1)[:, :KN]; Ta = np.argpartition(-Sa, KA, axis=1)[:, :KA]
        for bi, s in enumerate(b):
            for j in set(Tn[bi].tolist()) | set(Ta[bi].tolist()):
                cand_rows.append((s, ids[j])); need23.add(ids[j])
        del Sn, Sa, Tn, Ta; gc.collect()
        if (i // BATCH) % 3 == 0: print(f'  {c} batch {i}/{len(sub)} t={time.time()-t0:.0f}s', flush=True)
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
print('rapidfuzz cdist (C-level)...', flush=True)
fn_ts = np.diag(rf_cdist(s1n, c2n, scorer=fuzz.token_set_ratio, workers=-1)).astype(np.float32)
print(f'  n_tset done t={time.time()-t0:.0f}s', flush=True)
fn_r = np.diag(rf_cdist(s1n, c2n, scorer=fuzz.ratio, workers=-1)).astype(np.float32)
fn_p = np.diag(rf_cdist(s1n, c2n, scorer=fuzz.partial_ratio, workers=-1)).astype(np.float32)
print(f'  n_ratio+n_part done t={time.time()-t0:.0f}s', flush=True)
fa_ts = np.diag(rf_cdist(s1a, c2a, scorer=fuzz.token_set_ratio, workers=-1)).astype(np.float32)
fa_r = np.diag(rf_cdist(s1a, c2a, scorer=fuzz.ratio, workers=-1)).astype(np.float32)
fa_p = np.diag(rf_cdist(s1a, c2a, scorer=fuzz.partial_ratio, workers=-1)).astype(np.float32)
print(f'  addr feats done t={time.time()-t0:.0f}s', flush=True)
del s1n, s1a, c2n, c2a; gc.collect()
s1_list = [s for s, _ in cand_rows]; cid_list = [c for _, c in cand_rows]
cty_list = [s1map[s][2] for s in s1_list]
same = np.array([1 if s1map[s][2] == m23.get(c, ('', '', ''))[2] else 0 for s, c in cand_rows], dtype=np.int8)
miss_n = np.array([1 if not str(m23.get(c, ('', '', ''))[0] or '') else 0 for _, c in cand_rows], dtype=np.int8)
miss_a = np.array([1 if not str(m23.get(c, ('', '', ''))[1] or '') else 0 for _, c in cand_rows], dtype=np.int8)
label = np.array([1 if c in gmap[s] else 0 for s, c in cand_rows], dtype=np.int8)
df = pd.DataFrame({'s1': s1_list, 'cid': cid_list, 'label': label, 'country': cty_list,
 'n_tset': fn_ts, 'n_ratio': fn_r, 'n_part': fn_p, 'a_tset': fa_ts, 'a_ratio': fa_r, 'a_part': fa_p,
 'same_cty': same, 'miss_n': miss_n, 'miss_a': miss_a})
df.to_parquet(f'{WORK}/feat_train.parquet', index=False)
print(f'saved {df.shape} pos_rate={df.label.mean():.4f} total_t={time.time()-t0:.0f}s')
print(df.groupby('label')[['n_tset', 'a_tset']].median())
