# End-of-Turn Detection — Situation Report

**Updated:** 2026-07-23 ~22:50  
**Ship:** `models/unified.joblib` → `predictions.csv` / `predictions_hi.csv`

## Best-of records

| Record | Delay | Notes |
|--------|------:|-------|
| **Best HI handout (current ship)** | **772 ms** | AUC 0.861 · beats silence 850 |
| Best EN handout | **1000 ms** | AUC 0.838 · gate-only checkpoint |
| Current EN (shipped with HI-best) | **1030 ms** | Hindi-priority tradeoff |
| Best HI honest | **840 ms** | pre-unified OOF |
| Best EN honest freeze | **1300 ms** | protocol @ val-frozen |

## Deliverables

| # | Item | Status |
|---|------|--------|
| 1 | `SUMMARY.html` | Done — best-of + current ship |
| 2 | `predict.py` | Done — `unified.joblib` |
| 3 | Predictions | Done — regenerated |
| 4 | `RUNLOG.md` | Done — through #30 |
| 5 | `NOTES.md` | Done |

## What changed last

1. First-pause rise/fall gates → EN **1000** / HI 783.  
2. Drop unified hold-weight boost + late-pause fall ×1.25 → HI **772** (EN 1030). Kept for Hindi-heavy hidden test.

## Commands

```bash
cd starter
python3 predict.py --data_dir ../eot_data/english --out predictions.csv
python3 predict.py --data_dir ../eot_data/hindi   --out predictions_hi.csv
python3 score.py --data_dir ../eot_data/english --pred predictions.csv
python3 score.py --data_dir ../eot_data/hindi   --pred predictions_hi.csv
```
