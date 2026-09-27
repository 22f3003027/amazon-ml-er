"""Stage 03b — train LightGBM matcher + tune threshold for F_0.5 on holdout S1 split.
Saves WORK/lgbm_matcher.txt + WORK/threshold.txt + prints val F_0.5 / P / R.
"""
import pandas as pd, numpy as np, os, lightgbm as lgb
BASE = '/content/drive/MyDrive/AMAZON_ML'
WORK = f'{BASE}/work'
FEATS = ['n_tset', 'n_ratio', 'n_part', 'a_tset', 'a_ratio', 'a_part', 'same_cty', 'miss_n', 'miss_a']
df = pd.read_parquet(f'{WORK}/feat_train.parquet')
print(f'train rows {df.shape} pos={df.label.mean():.4f}', flush=True)
s1s = df['s1'].unique()
rng = np.random.RandomState(42); rng.shuffle(s1s)
sp = int(len(s1s) * 0.8)
tr, va = set(s1s[:sp]), set(s1s[sp:])
dtr = df[df.s1.isin(tr)]; dva = df[df.s1.isin(va)]
print(f's1 split train={len(tr)} val={len(va)} rows {len(dtr)}/{len(dva)}', flush=True)
md = lgb.Dataset(dtr[FEATS], label=dtr['label'])
mv = lgb.Dataset(dva[FEATS], label=dva['label'], reference=md)
params = {'objective': 'binary', 'metric': 'binary_logloss', 'num_leaves': 63, 'min_data_in_leaf': 200,
          'feature_fraction': 0.9, 'bagging_fraction': 0.9, 'bagging_freq': 1, 'learning_rate': 0.05, 'verbose': -1}
bo = lgb.train(params, md, num_boost_round=500, valid_sets=[mv])
bo.save_model(f'{WORK}/lgbm_matcher.txt')
pv = bo.predict(dva[FEATS])
g = dva.assign(p=pv).groupby('s1')
best = (0, 0, 0, 0)
for th in [round(x, 2) for x in np.arange(0.3, 0.96, 0.05)]:
    P = R = F = n = 0
    for s1, grp in g:
        pred = set(grp[grp.p >= th]['cid']); true = set(grp[grp.label == 1]['cid'])
        if not true and not pred: f = 1.0; p = r = 1.0
        elif not true or not pred: f = 0.0; p = 0.0 if pred else 1.0; r = 0.0 if true else 1.0
        else:
            tp = len(pred & true); p = tp / len(pred); r = tp / len(true)
            f = 1.25 * p * r / (0.25 * p + r) if (p + r) > 0 else 0.0
        P += p; R += r; F += f; n += 1
    F /= n; P /= n; R /= n
    print(f'th={th:.2f} F05={F:.4f} P={P:.4f} R={R:.4f}')
    if F > best[1]: best = (th, F, P, R)
th, F, P, R = best
open(f'{WORK}/threshold.txt', 'w').write(f'{th}\n')
print(f'BEST th={th} F05={F:.4f} P={P:.4f} R={R:.4f}')
print('saved matcher + threshold')
print('importance:', dict(zip(FEATS, bo.feature_importance(importance_type='gain'))))
