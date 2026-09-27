# Amazon ML Challenge 2026 — Business Entity Resolution
Colab-first project. Laptop = code editing only. Colab + Drive = all heavy compute.

## The deal (why this setup)
- Dataset is ~2.5GB unzipped, ~3.5M rows. Your laptop (15.7GB RAM, i5-13500H, no GPU)
  can edit code but shouldn't process full data.
- All `.tsv` files + notebooks run on **Google Colab** (free T4 GPU / high-RAM).
- **Google Drive** is the shared disk between laptop and Colab.
- **GitHub** versions the code. Data NEVER goes to GitHub (too big).

## One-time setup (15 min, do now)

### 1. Drive (via browser at drive.google.com)
- Create folder `MyDrive/AMAZON_ML/`
- Upload `6ab10eb3b23ba_student_resource.zip` (~1.1GB) into it. That's it —
  do NOT unzip locally, Colab unzips it on Drive.

### 2. This repo → GitHub
```powershell
cd 'd:\Amazon ML'
git init; git add .gitignore README.md notebooks/ code/ utils/ Documentation_template.md output/.gitkeep
git commit -m "colab-first scaffold"
gh repo create amazon-ml-er --public --source=. --push
```

### 3. Colab (first run, ~10 min)
- Go to colab.research.google.com → Upload → `notebooks/00_setup_colab.ipynb`
- Runtime → Change runtime type → T4 GPU (for later embedding steps)
- Run all cells: mounts Drive, installs deps, unzips data on Drive, prints row counts.
- Then open `notebooks/01_eda.ipynb` → Run all (sampled EDA, memory-safe).

## Daily workflow (the loop you'll live in)
1. **I write code here** (VS Code, in `code/business_entity_resolution/src/` or new notebook cells).
2. **You sync to Colab**: either `git push` then `!git pull` in Colab, or copy-paste the cell.
3. **Run heavy cells on Colab**, outputs land in `MyDrive/AMAZON_ML/output/` + `work/`.
4. **Colab auto-saves** `.ipynb` back to Drive/GitHub — I read results from your pasted output.
5. Small files (`matching_results.tsv` ~ tens of MB) you download from Drive to
   `output/` here for validation + portal upload. Big files stay on Drive.

## Folders
- `notebooks/` — numbered Colab notebooks (00 setup, 01 EDA, 02 blocking, ...). Versioned.
- `code/business_entity_resolution/src/` — final runnable pipeline (what goes in the zip).
- `utils/validate_submission.py` — format validator (stdlib only, run anywhere).
- `output/` — ONLY small final TSVs for portal upload. Never commit big data.
- `data/` — NOT in repo. Lives only on Drive (`MyDrive/AMAZON_ML/data/`).

## Colab one-cell runner (copy-paste into ONE empty cell, Run)

```python
import subprocess, sys, os
REPO='https://github.com/22f3003027/amazon-ml-er.git'; DST='/content/amazon-ml-er'
if not os.path.exists(DST):
    subprocess.run(['git','clone','--depth','1',REPO,DST],check=True)
else:
    subprocess.run(['git','-C',DST,'pull','--ff-only'],check=False)
subprocess.run([sys.executable,'-m','pip','install','-q','pandas','numpy','scikit-learn','scipy','rapidfuzz','lightgbm','pyarrow','joblib','tqdm'],check=True)
STAGES=[]  # e.g. ['02a_build_index.py','02b_recall_test.py']
for s in STAGES:
    print('\n'+'='*70+'\n>>> '+s, flush=True)
    r=subprocess.run([sys.executable,f'{DST}/code/business_entity_resolution/src/{s}'],capture_output=True,text=True)
    print(r.stdout[-6000:]); print(r.stderr[-6000:]); print('<<< exit',r.returncode)
    if r.returncode!=0: break
```

Why `capture_output`: Colab otherwise swallows tracebacks. `exit 2` = script file
missing (fresh runtime lost the git clone — the `clone-if-missing` lines above fix it).
1. `00_setup_colab` ✅ (created) — mount, deps, unzip, row counts
2. `01_eda` ✅ (created) — sampled EDA, noise patterns, France check
3. `02_blocking` (next) — TF-IDF + country-aware blocking, recall ceiling + reduction ratio
4. `03_features_model` — string-similarity features + LightGBM, threshold tuned for F_0.5
5. `04_predict_submit` — full test inference → matching_results.tsv + candidate_pairs.tsv → validate

## Rules (don't break)
- No external APIs/DBs/geocoding for entity lookup (disqualification).
- Model ≤8B params, MIT/Apache-2.0 license.
- `matched_entity_ids` must be S2/S3 IDs from test set; one row per S1; empty = singleton.
- Final matches ⊆ candidates (validator warns otherwise).
