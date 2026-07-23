"""Predict EOT probabilities from a SAVED model (no refitting).

    python predict.py --data_dir <folder> --out predictions.csv

Works on any folder with the same structure/labels schema (never-seen turns OK).
predictions.csv columns: turn_id,pause_index,p_eot

Prefers models/unified.joblib (EN+HI), then models/<lang>.joblib,
then models/default.joblib.
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

# Allow running from deleieverable/ or starter/
_HERE = os.path.dirname(os.path.abspath(__file__))
_STARTER = _HERE if os.path.isfile(os.path.join(_HERE, "features.py")) else os.path.join(_HERE, "..", "starter")
_STARTER = os.path.abspath(_STARTER)
if _STARTER not in sys.path:
    sys.path.insert(0, _STARTER)

import joblib
import numpy as np

from features import (
    detect_lang_id,
    extract_features,
    load_wav,
    precompute_contours,
)
from train import MODEL_DIR, predict_proba_bundle

_DEFAULT_MODEL_DIR = MODEL_DIR if os.path.isdir(MODEL_DIR) else os.path.join(_STARTER, "models")


def _resolve_model_path(model_dir, data_dir, rows):
    lang_id = detect_lang_id(data_dir, rows)
    lang_name = "hindi" if lang_id >= 0.5 else "english"
    unified = os.path.join(model_dir, "unified.joblib")
    specific = os.path.join(model_dir, f"{lang_name}.joblib")
    default = os.path.join(model_dir, "default.joblib")
    if os.path.isfile(unified):
        return unified, lang_id, "unified"
    if os.path.isfile(specific):
        return specific, lang_id, lang_name
    if os.path.isfile(default):
        return default, lang_id, lang_name
    raise SystemExit(
        f"No saved model found in {model_dir}. "
        f"Expected unified.joblib under starter/models/."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model_dir", default=_DEFAULT_MODEL_DIR)
    args = ap.parse_args()

    labels_path = os.path.join(args.data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))
    model_path, lang_id, lang_name = _resolve_model_path(
        args.model_dir, args.data_dir, rows
    )
    bundle = joblib.load(model_path)
    print(f"loaded model {model_path} (lang hint={lang_name})")

    by_turn = defaultdict(list)
    for r in rows:
        by_turn[r["turn_id"]].append(r)
    last_end = {}
    for tid, prs in by_turn.items():
        prs = sorted(prs, key=lambda r: int(r["pause_index"]))
        prev = 0.0
        for r in prs:
            last_end[(tid, int(r["pause_index"]))] = prev
            prev = float(r["pause_end"])

    cache = {}
    contour_cache = {}
    X, rise, fall, keys = [], [], [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
            contour_cache[path] = precompute_contours(*cache[path])
        x, sr = cache[path]
        tid, pi = r["turn_id"], int(r["pause_index"])
        feat, rs, fs = extract_features(
            x, sr, float(r["pause_start"]), pi,
            last_pause_end=last_end[(tid, pi)],
            contours=contour_cache[path],
            lang_id=lang_id,
        )
        X.append(feat)
        rise.append(rs)
        fall.append(fs)
        keys.append((tid, pi))

    X = np.asarray(X, dtype=np.float32)
    rise = np.asarray(rise, dtype=np.float64)
    fall = np.asarray(fall, dtype=np.float64)
    p = predict_proba_bundle(bundle, X, rise, fall)

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pv in zip(keys, p):
            w.writerow([tid, pi, f"{pv:.6f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
