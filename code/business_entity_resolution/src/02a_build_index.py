"""Stage 02a-v2 — memory-safe blocking index: HashingVectorizer (stateless) + per-country shards.
Exit -9 was OOM: 10.3M-row TF-IDF vocab matrix doesn't fit 12GB Colab RAM.
Fix: HashingVectorizer (no vocab dict), float32, 100k chunks, per-country shards.
Saves: WORK/block_hash_cfg.joblib, WORK/block_matrix_{US,IN}.npz, WORK/block_ids_{US,IN}.parquet
"""
import pandas as pd, numpy as np, os, re, unicodedata, joblib, gc
from sklearn.feature_extraction.text import HashingVectorizer
from scipy import sparse
BASE = '/content/drive/MyDrive/AMAZON_ML'
TRAIN = f'{BASE}/data/student_resource/dataset/train'
WORK = f'{BASE}/work'; os.makedirs(WORK, exist_ok=True)
SUFFIX = r'\b(inc|incorporated|corp|corporation|llc|ltd|limited|pvt|private|llp|pllc|co|company|services|enterprises|solutions|holdings|group|associates|partners)\b\.?'
def norm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return ''
    s = unicodedata.normalize('NFKD', str(s)).encode('ascii', 'ignore').decode('ascii')
    s = s.lower(); s = re.sub(r'\.com|\.in|\.org|\.net|\.co\b', ' ', s)
    s = re.sub(r'[^a-z0-9 ]', ' ', s); s = re.sub(SUFFIX, ' ', s)
    return re.sub(r'\s+', ' ', s).strip()
print('building stateless hasher (2^18 char_wb 3-5gram)...', flush=True)
vec = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 5), n_features=2**18, alternate_sign=False, norm='l2')
joblib.dump({'n_features': 2**18, 'ngram': (3, 5)}, f'{WORK}/block_hash_cfg.joblib')
for shard, cty in [('US', 'US'), ('IN', 'India')]:
    print(f'shard {shard}: streaming {cty} rows...', flush=True)
    ids, Xp = [], []
    for p in [f'{TRAIN}/train_source2.tsv', f'{TRAIN}/train_source3.tsv']:
        for ch in pd.read_csv(p, sep='\t', chunksize=100000, dtype=str, keep_default_na=True, usecols=['entity_id', 'business_name', 'country']):
            m = ch[ch['country'] == cty]
            if len(m) == 0: continue
            ids.extend(m['entity_id'].tolist())
            Xp.append(vec.transform([norm(x) for x in m['business_name'].tolist()]))
            del m
        gc.collect()
    X = sparse.vstack(Xp).tocsr().astype(np.float32)
    del Xp; gc.collect()
    print(f'shard {shard}: {len(ids):,} rows matrix {X.shape} nnz={X.nnz:,} mem~{X.data.nbytes/1e9:.2f}GB', flush=True)
    sparse.save_npz(f'{WORK}/block_matrix_{shard}.npz', X)
    pd.DataFrame({'entity_id': ids}).to_parquet(f'{WORK}/block_ids_{shard}.parquet', index=False)
    del X, ids; gc.collect()
print('saved per-shard index')
