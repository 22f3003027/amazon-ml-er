"""RUN_ALL EDA v2 — one click on Colab: executes A→D, saves results to Drive.
Paste this whole file into ONE Colab cell and run. No per-cell clicking.
Pulls latest src from GitHub so fixes auto-apply."""
import subprocess, sys, os
REPO = 'https://github.com/22f3003027/amazon-ml-er.git'
DST = '/content/amazon-ml-er'
if not os.path.exists(DST):
    subprocess.run(['git', 'clone', '--depth', '1', REPO, DST], check=True)
else:
    subprocess.run(['git', '-C', DST, 'pull', '--ff-only'], check=False)
sys.path.insert(0, DST)
from google.colab import drive
drive.mount('/content/drive')
# pip deps (quiet, fast if cached)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'pandas', 'numpy', 'matplotlib', 'rapidfuzz', 'tqdm'], check=True)
# run stages A..D as scripts from repo (each prints + saves to Drive work/)
for stage in ['01a_scan.py', '01b_gtstats.py', '01c_pairs.py', '01d_similarity.py']:
    print('\n' + '=' * 70 + '\n>>> STAGE', stage)
    r = subprocess.run([sys.executable, f'{DST}/code/business_entity_resolution/src/{stage}'])
    print('<<< exit', r.returncode)
    if r.returncode != 0:
        print('STOPPED — fix error above, then re-run this cell'); break
else:
    print('\nALL EDA STAGES DONE — paste the printed summaries back to chat.')
