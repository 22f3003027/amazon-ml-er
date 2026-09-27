"""Stage 02c — address-channel index (per-country shards, word 1-2gram + digit-preserving norm).
Catches transliterated + trade-name pairs via shared address tokens/house numbers/PINs.
Saves: WORK/addr_matrix_{US,IN}.npz (ids reuse block_ids shards).
"""
import pandas as pd, numpy as np, os, re, joblib, gc
from sklearn.feature_extraction.text import HashingVectorizer
from scipy import sparse
BASE = '/content/drive/MyDrive/AMAZON_ML'
TRAIN = f'{BASE}/data/student_resource/dataset/train'
WORK = f'{BASE}/work'; os.makedirs(WORK, exist_ok=True)
def anorm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return ''
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = re.sub(r'\b(null|nan|none)\b', ' ', s)
    return re.sub(r'\s+', ' ', s).strip()
vec = HashingVectorizer(analyzer='word', ngram_range=(1, 2), n_features=2**18, alternate_sign=False, norm='l2')
for shard, cty in [('US', 'US'), ('IN', 'India')]:
    print(f'addr shard {shard}...', flush=True)
    Xp = []
    for p in [f'{TRAIN}/train_source2.tsv', f'{TRAIN}/train_source3.tsv']:
        for ch in pd.read_csv(p, sep='\t', chunksize=100000, dtype=str, keep_default_na=True, usecols=['business_address', 'country']):
            m = ch[ch['country'] == cty]['business_address']
            if len(m) == 0: continue
            Xp.append(vec.transform([anorm(x) for x in m.tolist()]))
        gc.collect()
    X = sparse.vstack(Xp).tocsr().astype(np.float32)
    del Xp; gc.collect()
    print(f'addr shard {shard}: {X.shape} nnz={X.nnz:,} mem~{X.data.nbytes/1e9:.2f}GB', flush=True)
    sparse.save_npz(f'{WORK}/addr_matrix_{shard}.npz', X)
    del X; gc.collect()
print('saved address index')
