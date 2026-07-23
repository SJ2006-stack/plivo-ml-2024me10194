# Commands — `python3` only

```bash
cd /Users/shrianshjaiswal/plivo-ml-2024me10194/starter
```

## Current best ship — EN 1000 / HI 783

```bash
python3 predict.py --data_dir ../eot_data/english --out predictions.csv
python3 predict.py --data_dir ../eot_data/hindi   --out predictions_hi.csv
python3 score.py --data_dir ../eot_data/english --pred predictions.csv
python3 score.py --data_dir ../eot_data/hindi   --pred predictions_hi.csv
```

## Retrain

```bash
python3 train.py --unified --data_root ../eot_data \
  --model_out models/unified.joblib --out mine_unified_full.csv
```
