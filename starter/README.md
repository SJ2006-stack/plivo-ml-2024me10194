# Starter kit

- `baseline.py` — the silence-only baseline; also shows the exact predict.py
  interface you must ship.
- `features.py` — audio loading, framing, energy, autocorrelation pitch
  tracker. Utilities only; the features are your job.
- `train.py` — runnable skeleton (weak on purpose).
- `score.py` — the official scorer. Your dev loop:

```
python baseline.py --data_dir ../eot_data/english --out base.csv
python score.py    --data_dir ../eot_data/english --pred base.csv
python train.py    --data_dir ../eot_data/english --out mine.csv
python score.py    --data_dir ../eot_data/english --pred mine.csv
```

Log every score in RUNLOG.md. Listen to your errors — that is where the
points are.
