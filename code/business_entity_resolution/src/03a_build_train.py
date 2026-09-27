"""Stage 03a — build matching training set: dual-channel candidates for sampled S1 + labels + features.
For each sampled S1: name top-30 + addr top-30 (60/S1, dedup) -> join S2/S3 rows ->
label from ground truth -> string-sim features. Saves WORK/feat_train.parquet.
Sample: 40k S1 -> ~2.4M pairs (fits RAM; LightGBM handles it).
"""
import pandas as pd, numpy as np, os, re, unicodedata, gc
from sklearn.feature_extraction.text import HashingVectorizer
from scipy import sparse
from rapidfuzz import fuzz
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
    sh[c] = (Xn, Xa, ids); print(f'{c} loaded', flush=True); gc.collect()
N_S1 = 40000
gt = pd.read_csv(f'{TRAIN}/train_ground_truth.tsv', sep='\t', nrows=N_S1, dtype=str, keep_default_na=False)
gmap = {r['source1_entity_id']: set(x.strip() for x in str(r['matched_entity_ids']).split(',') if x.strip()) for _, r in gt.iterrows()}
need1 = set(gmap)
s1map = {}
for ch in pd.read_csv(f'{TRAIN}/train_source1.tsv', sep='\t', chunksize=200000, dtype=str, keep_default_na=True):
    for _, r in ch[ch['entity_id'].isin(need1)].iterrows():
        s1map[r['entity_id']] = (r['business_name'], r['business_address'], r['country'])
    if len(s1map) >= len(need1): break
print(f's1 {len(s1map)}/{len(need1)}', flush=True)
# collect candidate ids per s1
KN, KA = 30, 30
cand_rows = []  # (s1, cand_id)
need23 = set()
for s1, (nm, ad, cty) in s1map.items():
    c = 'US' if cty == 'US' else 'IN'
    Xn, Xa, ids = sh[c]
    sn = (nvec.transform([norm(nm)]) @ Xn).toarray().ravel()
    sa = (avec.transform([anorm(ad)]) @ Xa).toarray().ravel()
    tn = set(np.argpartition(-sn, KN)[:KN].tolist()) if (sn > 0).any() else set()
    ta = set(np.argpartition(-sa, KA)[:KA].tolist()) if (sa > 0).any() else set()
    for i in tn | ta:
        cand_rows.append((s1, ids[i])); need23.add(ids[i])
print(f'pairs: {len(cand_rows):,} need23: {len(need23):,}', flush=True)
# fetch S2/S3 rows for needed ids
m23 = {}
for p in [f'{TRAIN}/train_source2.tsv', f'{TRAIN}/train_source3.tsv']:
    for ch in pd.read_csv(p, sep='\t', chunksize=200000, dtype=str, keep_default_na=True):
        for _, r in ch[ch['entity_id'].isin(need23)].iterrows():
            m23[r['entity_id']] = (r['business_name'], r['business_address'], r['country'])
        if len(m23) >= len(need23): break
    if len(m23) >= len(need23): break
print(f'm23 {len(m23)}/{len(need23)}', flush=True)
def feats(a, b):
    a = a or ''; b = b or ''
    return [fuzz.token_set_ratio(a, b), fuzz.ratio(a, b), fuzz.partial_ratio(a, b)]
import time; t0 = time.time()
out = []
for s1, cid in cand_rows:
    nm, ad, cty = s1map[s1]
    rec = m23.get(cid)
    if rec is None: continue
    cn, ca, ccty = rec
    fn = feats(str(nm or ''), str(cn or ''))
    fa = feats(str(ad or ''), str(ca or ''))
    label = 1 if cid in gmap[s1] else 0
    out.append([s1, cid, label, cty, fn[0], fn[1], fn[2], fa[0], fa[1], fa[2],
                1 if (cty == ccty) else 0, 1 if not str(cn or '') else 0, 1 if not str(ca or '') else 0])
cols = ['s1', 'cid', 'label', 'country', 'n_tset', 'n_ratio', 'n_part', 'a_tset', 'a_ratio', 'a_part', 'same_cty', 'miss_n', 'miss_a']
df = pd.DataFrame(out, columns=cols)
df.to_parquet(f'{WORK}/feat_train.parquet', index=False)
print(f'saved {df.shape} pos_rate={df.label.mean():.4f} time={time.time()-t0:.0f}s')
print(df.groupby('label')[['n_tset', 'a_tset']].median())
