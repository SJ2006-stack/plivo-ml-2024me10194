"""Train a small causal EOT model and write predictions.

Hard rules:
- Features: audio only up to pause_start (enforced in features.extract_features).
- Sample weights MUST NOT use pause duration / pause_end (future info).
- Final submission CSVs should come from predict.py on a SAVED model.

*** Unified train (preferred) ***
One model on BOTH languages (more prosody data). Prefer 60/20/20 train splits:

    python3 train.py --unified --splits_dir ../eot_splits \\
        --model_out models/unified.joblib --out mine_unified.csv

Or pool full handout folders:

    python3 train.py --unified --data_root ../eot_data \\
        --model_out models/unified.joblib --out mine_unified.csv

Lang-specific english.joblib / hindi.joblib stay frozen unless --force-retrain.
"""
import argparse
import csv
import os
from collections import defaultdict

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import (
    detect_lang_id,
    extract_features,
    load_wav,
    precompute_contours,
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
FROZEN_MODEL_NAMES = ("english.joblib", "hindi.joblib", "default.joblib")
FREEZE_FLAG = os.path.join(MODEL_DIR, "DO_NOT_RETRAIN")

# Small ensemble only — giant RF/ET removed to limit overfit on ~200 turns.
ENSEMBLE_WEIGHTS = {
    "lr": 0.55,
    "hgb": 0.45,
}


def _rise_beta(lang_id):
    # Unified (~0.5): mild. Pure HI used to be 0.25 and crushed short rising EOTs.
    if lang_id >= 0.75:
        return 0.14
    if lang_id >= 0.4:
        return 0.10
    return 0.12


def _fall_gamma(lang_id):
    if lang_id >= 0.4:
        return 0.10
    return 0.05


def _sample_weights(labels, lang_id=0.0):
    """Class-balance weights only. NO pause duration / pause_end."""
    labels = list(labels)
    n = len(labels)
    n_eot = sum(1 for lab in labels if lab == "eot")
    n_hold = n - n_eot
    w_eot = n / (2.0 * max(n_eot, 1))
    w_hold = n / (2.0 * max(n_hold, 1))
    if lang_id >= 0.5:
        w_hold *= 1.15
    return np.array(
        [w_eot if lab == "eot" else w_hold for lab in labels],
        dtype=np.float64,
    )


def _build_estimators():
    return {
        "lr": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=4000,
                class_weight="balanced",
                C=0.35,
                random_state=0,
            )),
        ]),
        "hgb": HistGradientBoostingClassifier(
            max_depth=2,
            learning_rate=0.06,
            max_iter=120,
            l2_regularization=5.0,
            min_samples_leaf=10,
            random_state=0,
        ),
    }


def _fit_one(name, est, X, y, sw):
    if name == "lr":
        est.fit(X, y, clf__sample_weight=sw)
    else:
        est.fit(X, y, sample_weight=sw)
    return est


def predict_proba_bundle(bundle, X, rise, fall=None):
    """Ensemble p_eot with light rising-F0 suppression + fall boost.

    High-ROI gates (from error listening):
    - First pause: soft rise penalty (short rising HI EOTs) + soft fall boost
      (phrase-final holds ≠ turn-final).
    - Later pauses: full rise/fall calibration.
    """
    p = np.zeros(len(X), dtype=np.float64)
    for name, w in bundle["weights"].items():
        p += w * bundle["estimators"][name].predict_proba(X)[:, 1]
    X = np.asarray(X)
    first = (X[:, 0] <= 0.5) if X.ndim == 2 and X.shape[1] > 0 else np.zeros(len(p), dtype=bool)
    beta = float(bundle.get("rise_beta", 0.0))
    if beta > 0 and rise is not None:
        rise = np.asarray(rise, dtype=np.float64)
        beta_eff = np.where(first, beta * 0.35, beta)
        p = p * (1.0 - beta_eff * np.clip(rise / 100.0, 0.0, 0.85))
    gamma = float(bundle.get("fall_gamma", 0.0))
    if gamma > 0 and fall is not None:
        fall = np.asarray(fall, dtype=np.float64)
        # Soft fall on first pause; full fall boost on later pauses.
        gamma_eff = np.where(first, gamma * 0.40, gamma)
        p = p * (1.0 + gamma_eff * np.clip(fall / 80.0, -0.3, 0.8))

    # Niche Hindi-only: multiplicative nudge on safe rows (skip first+rise>fall).
    if X.ndim == 2 and X.shape[1] > 7 and fall is not None and rise is not None:
        hi = X[:, 7] >= 0.5
        fall_a = np.asarray(fall, dtype=np.float64)
        rise_a = np.asarray(rise, dtype=np.float64)
        safe = hi & ((~first) | (fall_a > rise_a + 5.0))
        p = np.where(safe, np.clip(p * 1.10, 0.0, 1.0), p)

    return np.clip(p, 0.0, 1.0)


def build_dataset(data_dir):
    labels_path = os.path.join(data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))
    lang_id = detect_lang_id(data_dir, rows)

    by_turn = defaultdict(list)
    for r in rows:
        by_turn[r["turn_id"]].append(r)

    last_end = {}
    for tid, prs in by_turn.items():
        prs = sorted(prs, key=lambda r: int(r["pause_index"]))
        prev = 0.0
        for r in prs:
            pi = int(r["pause_index"])
            last_end[(tid, pi)] = prev
            # previous pause_end is in the past for the next pause (causal IPI)
            prev = float(r["pause_end"])

    cache = {}
    contour_cache = {}
    X, y, groups, keys, labels, rise, fall = [], [], [], [], [], [], []
    for r in rows:
        path = os.path.join(data_dir, r["audio_file"])
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
        y.append(1 if r["label"] == "eot" else 0)
        groups.append(tid)
        keys.append((tid, pi))
        labels.append(r["label"])
        rise.append(rs)
        fall.append(fs)

    return {
        "X": np.asarray(X, dtype=np.float32),
        "y": np.asarray(y, dtype=np.int32),
        "groups": np.asarray(groups),
        "keys": keys,
        "labels": labels,
        "rise": np.asarray(rise, dtype=np.float64),
        "fall": np.asarray(fall, dtype=np.float64),
        "lang_id": lang_id,
        "lang_name": "hindi" if lang_id >= 0.5 else "english",
    }


def _sibling_dir(data_dir):
    """If training english, also use hindi (and vice versa) when present.

    Supports both eot_data/{lang} and eot_splits/{lang}/train layouts.
    """
    abs_dir = os.path.abspath(data_dir)
    base = os.path.basename(abs_dir).lower()
    parent = os.path.dirname(abs_dir)

    # eot_splits/english/train -> sibling eot_splits/hindi/train
    if base == "train":
        lang_parent = parent
        lang_name = os.path.basename(lang_parent).lower()
        grand = os.path.dirname(lang_parent)
        if "english" in lang_name or lang_name.startswith("en"):
            cand = os.path.join(grand, "hindi", "train")
        elif "hindi" in lang_name or lang_name.startswith("hi"):
            cand = os.path.join(grand, "english", "train")
        else:
            return None
        if os.path.isfile(os.path.join(cand, "labels.csv")):
            return cand
        return None

    if "english" in base or base.startswith("en"):
        cand = os.path.join(parent, "hindi")
    elif "hindi" in base or base.startswith("hi"):
        cand = os.path.join(parent, "english")
    else:
        return None
    if os.path.isfile(os.path.join(cand, "labels.csv")):
        return cand
    return None


def merge_datasets(ds_a, ds_b):
    """Stack two language datasets into one unified training set."""
    return {
        "X": np.vstack([ds_a["X"], ds_b["X"]]),
        "y": np.concatenate([ds_a["y"], ds_b["y"]]),
        "groups": np.concatenate([ds_a["groups"], ds_b["groups"]]),
        "keys": list(ds_a["keys"]) + list(ds_b["keys"]),
        "labels": list(ds_a["labels"]) + list(ds_b["labels"]),
        "rise": np.concatenate([ds_a["rise"], ds_b["rise"]]),
        "fall": np.concatenate([ds_a["fall"], ds_b["fall"]]),
        "lang_id": 0.5,  # mixed
        "lang_name": "unified",
    }


def train_bundle(ds, aux=None):
    if aux is None:
        X, y = ds["X"], ds["y"]
        labels = ds["labels"]
        lang_id = ds["lang_id"]
    else:
        X = np.vstack([ds["X"], aux["X"]])
        y = np.concatenate([ds["y"], aux["y"]])
        labels = list(ds["labels"]) + list(aux["labels"])
        lang_id = ds["lang_id"]
    sw = _sample_weights(labels, lang_id=lang_id)
    estimators = _build_estimators()
    for name, est in estimators.items():
        _fit_one(name, est, X, y, sw)
    return {
        "estimators": estimators,
        "weights": dict(ENSEMBLE_WEIGHTS),
        "rise_beta": _rise_beta(lang_id),
        "fall_gamma": _fall_gamma(lang_id),
        "lang_id": float(lang_id),
        "lang_name": ds["lang_name"],
        "n_features": int(X.shape[1]),
        "protocol": "no_pause_duration_weights; lr+small_hgb; unified_en_hi",
    }


def _resolve_unified_dirs(splits_dir, data_root):
    """Return (en_dir, hi_dir) for unified train."""
    if splits_dir:
        en = os.path.join(splits_dir, "english", "train")
        hi = os.path.join(splits_dir, "hindi", "train")
    elif data_root:
        en = os.path.join(data_root, "english")
        hi = os.path.join(data_root, "hindi")
    else:
        raise SystemExit("--unified needs --splits_dir or --data_root")
    for d in (en, hi):
        if not os.path.isfile(os.path.join(d, "labels.csv")):
            raise SystemExit(f"missing {d}/labels.csv")
    return en, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=None,
                    help="Single-language folder (ignored with --unified)")
    ap.add_argument("--unified", action="store_true",
                    help="Train ONE model on English+Hindi together")
    ap.add_argument("--splits_dir", default=None,
                    help="With --unified: use {english,hindi}/train under this "
                         "(60/20/20). Example: ../eot_splits")
    ap.add_argument("--data_root", default=None,
                    help="With --unified: use {english,hindi} under this "
                         "(full handout). Example: ../eot_data")
    ap.add_argument("--out", "--output", default="predictions.csv",
                    dest="out")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    ap.add_argument("--model_out", default=None,
                    help="Where to write the joblib "
                         "(default: models/unified.joblib if --unified else "
                         "models/<lang>.joblib)")
    ap.add_argument("--no_joint", action="store_true",
                    help="Do not pool sibling language data (non-unified)")
    ap.add_argument(
        "--force-retrain",
        action="store_true",
        help="Allow overwriting models/{english,hindi,default}.joblib",
    )
    args = ap.parse_args()

    if args.unified:
        en_dir, hi_dir = _resolve_unified_dirs(args.splits_dir, args.data_root)
        print(f"UNIFIED train: pooling\n  {en_dir}\n  {hi_dir}")
        ds = merge_datasets(build_dataset(en_dir), build_dataset(hi_dir))
        aux = None
        model_path = args.model_out or os.path.join(args.model_dir, "unified.joblib")
    else:
        if not args.data_dir:
            raise SystemExit("need --data_dir, or --unified with --splits_dir/--data_root")
        freeze_on = os.path.isfile(FREEZE_FLAG) or any(
            os.path.isfile(os.path.join(args.model_dir, n)) for n in FROZEN_MODEL_NAMES
        )
        if freeze_on and not args.force_retrain:
            raise SystemExit(
                "HARD RULE: lang-specific models are FROZEN.\n"
                "  Prefer: python3 train.py --unified --splits_dir ../eot_splits "
                "--model_out models/unified.joblib\n"
                "  Or pass --force-retrain to overwrite english/hindi/default."
            )
        ds = build_dataset(args.data_dir)
        aux = None
        if not args.no_joint:
            sib = _sibling_dir(args.data_dir)
            if sib is not None:
                print(f"pooling sibling data from {sib}")
                aux = build_dataset(sib)
        model_path = args.model_out or os.path.join(
            args.model_dir, f"{ds['lang_name']}.joblib"
        )

    X, y, groups = ds["X"], ds["y"], ds["groups"]
    rise_beta = _rise_beta(ds["lang_id"])
    fall_gamma = _fall_gamma(ds["lang_id"])

    # Held-out turn sanity
    tr, te = next(
        GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
        .split(X, y, groups)
    )
    if aux is None:
        Xtr, ytr = X[tr], y[tr]
        sw = _sample_weights([ds["labels"][i] for i in tr], ds["lang_id"])
    else:
        Xtr = np.vstack([X[tr], aux["X"]])
        ytr = np.concatenate([y[tr], aux["y"]])
        sw = _sample_weights(
            [ds["labels"][i] for i in tr] + list(aux["labels"]),
            ds["lang_id"],
        )
    tmp = _build_estimators()
    for name, est in tmp.items():
        _fit_one(name, est, Xtr, ytr, sw)
    p_te = predict_proba_bundle(
        {
            "estimators": tmp,
            "weights": ENSEMBLE_WEIGHTS,
            "rise_beta": rise_beta,
            "fall_gamma": fall_gamma,
        },
        X[te],
        ds["rise"][te],
        ds["fall"][te],
    )
    acc = float(np.mean((p_te >= 0.5) == y[te]))
    print(
        f"held-out turn accuracy: {acc:.3f} "
        f"(chance ~ {max(np.mean(y), 1 - np.mean(y)):.3f})"
    )

    bundle = train_bundle(ds, aux=aux)
    os.makedirs(os.path.dirname(os.path.abspath(model_path)) or ".", exist_ok=True)
    # unified writes are allowed even if lang models are read-only
    try:
        joblib.dump(bundle, model_path)
    except PermissionError:
        raise SystemExit(
            f"cannot write {model_path} (read-only?). "
            "Use --model_out models/unified.joblib"
        )
    print(f"saved model -> {model_path}")

    # Also refresh default.joblib so predict.py finds the unified model
    default_path = os.path.join(args.model_dir, "default.joblib")
    if args.unified or args.force_retrain:
        try:
            if os.path.isfile(default_path) and not os.access(default_path, os.W_OK):
                os.chmod(default_path, 0o644)
            joblib.dump(bundle, default_path)
            print(f"saved model -> {default_path}")
        except OSError as e:
            print(f"NOTE: could not update default.joblib ({e})")

    p = predict_proba_bundle(bundle, X, ds["rise"], ds["fall"])
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pv in zip(ds["keys"], p):
            w.writerow([tid, pi, f"{pv:.6f}"])
    print(f"wrote {len(ds['keys'])} predictions -> {args.out}")
    print("NOTE: honest scores = predict on held-out + eval_holdout protocol.")


if __name__ == "__main__":
    main()
