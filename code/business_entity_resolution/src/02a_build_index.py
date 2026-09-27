"""Stage 02a — build normalized blocking corpus + TF-IDF index (S2+S3), save to Drive.
Memory-safe: streams S2/S3 in chunks, char-ngram TF-IDF on normalized name.
Saves: WORK/block_vectorizer.joblib, WORK/block_matrix.npz, WORK/block_ids.parquet
"""
import pandas as pd, numpy as np, os, re, unicodedata, joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
BASE = '/content/drive/MyDrive/AMAZON_ML'
TRAIN = f'{BASE}/data/student_resource/dataset/train'
WORK = f'{BASE}/work'; os.makedirs(WORK, exist_ok=True)
SUFFIX = r'\b(inc|incorporated|corp|corporation|llc|ltd|limited|pvt|private|llp|pllc|co|company|corp\.|inc\.|ltd\.|llc\.|services|enterprises|solutions|holdings|group|associates|partners)\b\.?'
def norm(s):
    if s is None or (isinstance(s, float) and np.isnan(s)): return ''
    s = str(s)
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')
    s = s.lower()
    s = re.sub(r'\.com|\.in|\.org|\.net|\.co\b', ' ', s)
    s = re.sub(r'[^a-z0-9 ]', ' ', s)
    s = re.sub(SUFFIX, ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s
print('pass 1: collect normalized names...', flush=True)
ids, texts = [], []
for p in [f'{TRAIN}/train_source2.tsv', f'{TRAIN}/train_source3.tsv']:
    for ch in pd.read_csv(p, sep='\t', chunksize=200000, dtype=str, keep_default_na=True, usecols=['entity_id', 'business_name']):
        ids.extend(ch['entity_id'].tolist())
        texts.extend([norm(x) for x in ch['business_name'].tolist()])
print(f'corpus: {len(ids):,} rows', flush=True)
print('pass 2: fit TF-IDF char_wb 3-5gram...', flush=True)
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2, sublinear_tf=True)
X = vec.fit_transform(texts)
print('matrix:', X.shape, f'nnz={X.nnz:,}', flush=True)
joblib.dump(vec, f'{WORK}/block_vectorizer.joblib')
sparse.save_npz(f'{WORK}/block_matrix.npz', X)
pd.DataFrame({'entity_id': ids}).to_parquet(f'{WORK}/block_ids.parquet', index=False)
print('saved vectorizer + matrix + ids')
