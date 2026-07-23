# Deliverable pack

## Contents
- `predict.py` — loads `starter/models/unified.joblib` (run from `starter/` or set paths)
- `predictions.csv` — **single CSV for both languages** (EN + HI, 496 pauses)
- `SUMMARY.html` — method, graphs, results
- `RUNLOG.md` — scored runs
- `NOTES.md` — ≤10 sentences

## Scores (handout / in-sample ship)
- English: **1000 ms** (AUC 0.838)
- Hindi: **781 ms** (AUC 0.857)

## `predictions.csv` schema
```
turn_id,pause_index,p_eot
en__000,0,0.14...
...
hi__099,1,0.60...
```
496 data rows = 248 English + 248 Hindi. No separate `predictions_hi.csv`.

## How to regenerate
```bash
cd starter
python3 predict.py --data_dir ../eot_data/english --out predictions.csv
python3 predict.py --data_dir ../eot_data/hindi   --out predictions_hi.csv
# merge:
python3 -c "import csv; rows=[]; 
[rows.extend(csv.DictReader(open(p))) for p in ['predictions.csv','predictions_hi.csv']];
w=csv.DictWriter(open('../deleieverable/predictions.csv','w',newline=''),fieldnames=['turn_id','pause_index','p_eot']);
w.writeheader(); w.writerows(rows)"
```
