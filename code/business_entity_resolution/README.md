# Business Entity Resolution — runnable pipeline
Reproduces `output/matching_results.tsv` + `output/candidate_pairs.tsv` from raw data.
Flesh out as notebooks mature (02 blocking → 03 model → 04 predict).

## Run on Colab (from repo root after git clone/pull)
```bash
pip install -r requirements.txt
python -m src.blocking --input $DATA/student_resource/dataset --out $WORK/candidates.parquet
python -m src.matching --candidates $WORK/candidates.parquet --out $OUT/matching_results.tsv
python utils/validate_submission.py --matching $OUT/matching_results.tsv --candidate $OUT/candidate_pairs.tsv --test-dir $DATA/student_resource/dataset/test
```
See `/notebooks` for the notebook versions of each stage.
