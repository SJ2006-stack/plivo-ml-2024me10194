# End-of-Turn (EOT) Detection — `plivo-ml-2024me10194`

Causal pause classifier for voice agents: at each annotated pause, output
`p_eot` = P(the human turn is over), using **only audio up to `pause_start`**.

**Current ship (handout):** English **1000 ms** / Hindi **783 ms** mean response
delay @ ≤5% interrupted turns (`models/unified.joblib`).

---

## Quick start

```bash
# from repo root
cd starter

# Predict on a data folder (never-seen turns OK if schema matches)
python3 predict.py --data_dir ../eot_data/english --out predictions.csv
python3 predict.py --data_dir ../eot_data/hindi   --out predictions_hi.csv

# Score
python3 score.py --data_dir ../eot_data/english --pred predictions.csv
python3 score.py --data_dir ../eot_data/hindi   --pred predictions_hi.csv
```

Requirements: Python 3 with `numpy`, `scipy`, `scikit-learn`, `librosa`,
`soundfile`, `joblib`, `pandas` (optional). Use the project venv if present:

```bash
../env/bin/python predict.py --data_dir ../eot_data/english --out predictions.csv
```

---

## Deliverables

Graded pack lives in **`deleieverable/`**:

| File | What it is |
|------|------------|
| `SUMMARY.html` | Method, results, graphs, human vs agent, vs silence |
| `predict.py` | CLI: `python predict.py --data_dir <folder> --out predictions.csv` |
| `predictions.csv` | Both languages in **one** CSV (496 rows) |
| `predictions_hi.csv` | Hindi-only copy (convenience) |
| `RUNLOG.md` | Every scoring run: score + 1–2 lines changed/why |
| `NOTES.md` | ≤10 sentences: signal, failures, one more day |

Open `deleieverable/SUMMARY.html` in a browser for charts.

You can also run predict from that folder (it resolves `../starter` for features/models):

```bash
cd deleieverable
python3 predict.py --data_dir ../eot_data/english --out /tmp/pred_en.csv
```

---

## Data layout (expected)

```
<data_dir>/
  labels.csv          # turn_id, pause_index, pause_start, pause_end, label, audio_file, ...
  audio/
    *.wav             # 16 kHz mono
```

`predict.py` writes:

```
turn_id,pause_index,p_eot
```

---

## Model

- Saved weights: `starter/models/unified.joblib` (preferred), else
  `english.joblib` / `hindi.joblib` / `default.joblib`
- Features: causal Tier-1 prosody (`starter/features.py`) — `librosa.piptrack`
  F0, energy decay, lengthening, relative pitch, spectral + turn structure
- Train (only if you must rebuild):

```bash
cd starter
python3 train.py --unified --data_root ../eot_data \
  --model_out models/unified.joblib --out mine_unified_full.csv
```

Prefer **not** retraining blindly — see `models/DO_NOT_RETRAIN` and `RUNLOG.md`.

---

## Honest eval (optional)

Turn-grouped 60/20/20 splits under `eot_splits/`:

```bash
cd starter
python3 eval_holdout.py --mode protocol \
  --splits_dir ../eot_splits/english --out_prefix pred_en
python3 eval_holdout.py --mode protocol \
  --splits_dir ../eot_splits/hindi --out_prefix pred_hi
```

Trusted generalization bars (from `RUNLOG.md`): EN protocol freeze ~**1300 ms**,
best Hindi OOF ~**840 ms**. Handout 1000/783 is the submission artifact (trained
on the labeled set).

---

## Repo map

```
plivo-ml-2024me10194/
  README.md                 # this file
  deleieverable/            # graded submission pack
  eot_data/{english,hindi}/ # handout audio + labels
  eot_splits/               # local train/val/test
  starter/
    predict.py  train.py  score.py  features.py  baseline.py
    models/unified.joblib
  RUNLOG.md  NOTES.md  SUMMARY.html  observation.md  commands.md
```

---

## Causality rule (non-negotiable)

For a pause at `pause_start`, features may use **only** audio from time `0`
to `pause_start`. Never use audio after the pause, and never use pause
duration / `pause_end` as features.
