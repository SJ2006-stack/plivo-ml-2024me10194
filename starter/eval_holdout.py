"""Honest evaluation with FROZEN models (Action 2: do not retrain).

Protocol (preferred):
  1. Load models/{lang}.joblib (never overwrite)
  2. Score val → freeze OP (highest thr in {0.5..0.8} with FCR<=4.5%)
  3. Score test ONLY at that frozen OP (no re-sweep)

    python eval_holdout.py --mode protocol \\
        --splits_dir ../eot_splits/hindi --out_prefix pred_hi

Legacy 2-way (also loads frozen model; ignores --train_dir fit):
    python eval_holdout.py --mode split \\
        --train_dir ../eot_splits/hindi/train \\
        --test_dir ../eot_splits/hindi/test

OOF mode is DISABLED under freeze (would refit).
"""
import argparse
import csv
import json
import os

import joblib
import numpy as np

from score import (
    DELAYS,
    FCR_CAP,
    THRESHOLDS,
    VAL_THRESHOLDS,
    evaluate,
    load,
    score,
    select_val_operating_point,
)
from train import (
    MODEL_DIR,
    build_dataset,
    predict_proba_bundle,
)


def _write_pred(path, keys, p):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pv in zip(keys, p):
            w.writerow([tid, pi, f"{float(pv):.6f}"])


def _fmt(r, label):
    return (
        f"[{label}] turns={r['n_turns']} pauses={r['n_pauses']} "
        f"AUC={r['auc']:.3f} delay={r['latency']*1000:.0f} ms "
        f"cutoffs={r['cutoff']*100:.1f}% thr={r['threshold']} "
        f"action_delay={r['delay']*1000:.0f} ms"
    )


def _score_at_op(labels_csv, pred_csv, threshold, delay):
    """Evaluate at a frozen operating point (no re-sweep)."""
    pauses = load(labels_csv, pred_csv)
    cut, lat = evaluate(pauses, threshold, delay)
    y = np.array([1 if p["label"] == "eot" else 0 for p in pauses])
    s = np.array([p["p"] for p in pauses])
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = ((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
    return {
        "latency": lat,
        "cutoff": cut,
        "threshold": threshold,
        "delay": delay,
        "auc": float(auc),
        "n_turns": len({p["turn_id"] for p in pauses}),
        "n_pauses": len(pauses),
    }


def run_split(train_dir, test_dir, out, model_dir, no_joint, pool_with=None):
    """Legacy: score test_dir with FROZEN model (no retrain / no overwrite)."""
    del no_joint, pool_with, train_dir  # kept for CLI compat; ignored under freeze
    ds_te = build_dataset(test_dir)
    lang = ds_te["lang_name"]
    model_path = os.path.join(model_dir, f"{lang}.joblib")
    if not os.path.isfile(model_path):
        model_path = os.path.join(model_dir, "default.joblib")
    if not os.path.isfile(model_path):
        raise SystemExit(
            f"HARD RULE: no frozen model at {model_dir}/{{{lang},default}}.joblib — "
            "do not retrain; restore the saved artifact."
        )
    bundle = joblib.load(model_path)
    print(f"loaded FROZEN model {model_path} (split mode; no train)")
    p = predict_proba_bundle(bundle, ds_te["X"], ds_te["rise"], ds_te["fall"])
    _write_pred(out, ds_te["keys"], p)
    r = score(os.path.join(test_dir, "labels.csv"), out)
    print(_fmt(r, f"SPLIT held-out lang={lang}"))
    return r


def run_protocol(splits_dir, out_prefix, model_dir, no_joint, pool_with=None):
    """Load FROZEN model → val (freeze OP) → test at frozen OP only.

    HARD RULE (Action 2): do NOT retrain. Uses models/{lang}.joblib as-is.

    Val: sweep thr in {0.5,0.6,0.7,0.8}; pick HIGHEST with FCR <= 4.5%;
         then pick delay under that cap (minimize latency). Freeze (thr, delay).
    Test: evaluate ONLY at the val-frozen OP — do NOT re-sweep on test.
    """
    del no_joint, pool_with  # ignored under freeze — no sibling retrain
    train_dir = os.path.join(splits_dir, "train")
    val_dir = os.path.join(splits_dir, "val")
    test_dir = os.path.join(splits_dir, "test")
    for d in (train_dir, val_dir, test_dir):
        if not os.path.isfile(os.path.join(d, "labels.csv")):
            raise SystemExit(f"missing {d}/labels.csv — run make_splits.py first")

    # Infer language from folder name (avoid loading train audio just for lang)
    base = os.path.basename(os.path.abspath(splits_dir)).lower()
    if "hindi" in base or base.startswith("hi"):
        lang = "hindi"
    elif "english" in base or base.startswith("en"):
        lang = "english"
    else:
        lang = "english"
    # Prefer unified EN+HI model, then lang-specific, then default
    candidates = [
        os.path.join(model_dir, "unified.joblib"),
        os.path.join(model_dir, f"{lang}.joblib"),
        os.path.join(model_dir, "default.joblib"),
    ]
    model_path = next((p for p in candidates if os.path.isfile(p)), None)
    if model_path is None:
        raise SystemExit(
            f"No model in {model_dir}. Train unified first:\n"
            "  python3 train.py --unified --splits_dir ../eot_splits "
            "--model_out models/unified.joblib"
        )
    bundle = joblib.load(model_path)
    print(f"loaded model {model_path} (protocol; no retrain)")

    # --- VAL: conservative threshold rule (no latency-chasing on thr) ---
    ds_va = build_dataset(val_dir)
    p_va = predict_proba_bundle(bundle, ds_va["X"], ds_va["rise"], ds_va["fall"])
    out_va = f"{out_prefix}_val.csv"
    _write_pred(out_va, ds_va["keys"], p_va)
    pauses_va = load(os.path.join(val_dir, "labels.csv"), out_va)
    r_va = select_val_operating_point(pauses_va, fcr_cap=FCR_CAP)
    print(_fmt(r_va, f"VAL freeze-OP lang={lang}"))
    print(
        f"  rule: thr in {list(VAL_THRESHOLDS)}, pick HIGHEST with FCR<={FCR_CAP*100:.1f}%; "
        f"feasible={r_va.get('feasible_thresholds')}"
    )
    print(
        f"  -> freeze OP from val: thr={r_va['threshold']}, "
        f"delay={r_va['delay']*1000:.0f} ms, FCR={r_va['cutoff']*100:.1f}%"
    )

    # --- TEST: frozen OP only (do NOT re-sweep) ---
    ds_te = build_dataset(test_dir)
    p_te = predict_proba_bundle(bundle, ds_te["X"], ds_te["rise"], ds_te["fall"])
    out_te = f"{out_prefix}_test.csv"
    _write_pred(out_te, ds_te["keys"], p_te)

    r_te_frozen = _score_at_op(
        os.path.join(test_dir, "labels.csv"),
        out_te,
        r_va["threshold"],
        r_va["delay"],
    )
    print(_fmt(r_te_frozen, f"TEST @ val-frozen OP lang={lang} [TRUST THIS]"))
    print("  note: no thr×delay re-sweep on test; model was not refit")

    summary = {
        "lang": lang,
        "splits_dir": os.path.abspath(splits_dir),
        "frozen_model": os.path.abspath(model_path),
        "retrained": False,
        "val": r_va,
        "test_frozen_op": r_te_frozen,
        "val_pred": os.path.abspath(out_va),
        "test_pred": os.path.abspath(out_te),
        "op_rule": {
            "val_thresholds": list(VAL_THRESHOLDS),
            "fcr_cap": FCR_CAP,
            "pick": "highest_threshold_with_fcr_le_cap",
            "test": "frozen_op_only_no_resweep",
        },
        "n_delays": len(DELAYS),
        "n_thresh_legacy_score_py": len(THRESHOLDS),
    }
    summary_path = f"{out_prefix}_protocol_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {summary_path}")
    return summary


def run_oof(data_dir, out, n_splits=5, no_joint=False):
    raise SystemExit(
        "HARD RULE: OOF refits models — disabled under freeze.\n"
        "  Use existing models/*.joblib + predict.py, or --mode protocol "
        "(loads frozen model, no retrain)."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        choices=("protocol", "split", "oof"),
        required=True,
    )
    ap.add_argument("--splits_dir",
                    help="eot_splits/{lang} with train/val/test (protocol mode)")
    ap.add_argument("--train_dir")
    ap.add_argument("--test_dir")
    ap.add_argument("--data_dir")
    ap.add_argument("--pool_with", default=None,
                    help="Optional extra train dir to pool (e.g. EN train when scoring HI)")
    ap.add_argument("--out", default="holdout_pred.csv")
    ap.add_argument("--out_prefix", default="pred",
                    help="Prefix for protocol val/test pred CSVs")
    ap.add_argument("--model_dir", default=MODEL_DIR)
    ap.add_argument("--no_joint", action="store_true")
    ap.add_argument("--n_splits", type=int, default=5)
    args = ap.parse_args()

    if args.mode == "protocol":
        if not args.splits_dir:
            raise SystemExit("protocol mode needs --splits_dir")
        run_protocol(
            args.splits_dir, args.out_prefix, args.model_dir,
            args.no_joint, pool_with=args.pool_with,
        )
    elif args.mode == "split":
        if not args.train_dir or not args.test_dir:
            raise SystemExit("split mode needs --train_dir and --test_dir")
        run_split(
            args.train_dir, args.test_dir, args.out, args.model_dir,
            args.no_joint, pool_with=args.pool_with,
        )
    else:
        if not args.data_dir:
            raise SystemExit("oof mode needs --data_dir")
        run_oof(args.data_dir, args.out, n_splits=args.n_splits, no_joint=args.no_joint)


if __name__ == "__main__":
    main()
