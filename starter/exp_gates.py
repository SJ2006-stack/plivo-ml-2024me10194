"""ONE-OFF gate experiments. Does NOT touch deleieverable/ or ship paths.

Runs each post-process gate ALONE on raw ensemble probs, scores:
  - full EN handout, full HI handout
  - HI 5-fold GroupKFold OOF

    python3 exp_gates.py
"""
from __future__ import annotations

import csv
import os

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold

from score import score
from train import (
    ENSEMBLE_WEIGHTS,
    MODEL_DIR,
    _fall_gamma,
    _fit_one,
    _rise_beta,
    _sample_weights,
    build_dataset,
    _build_estimators,
)

ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(ROOT, "_exp_gate_cache.joblib")
OUT_DIR = os.path.join(ROOT, "_exp_gates_out")
os.makedirs(OUT_DIR, exist_ok=True)


def _raw_p(bundle, X):
    p = np.zeros(len(X), dtype=np.float64)
    for name, w in bundle["weights"].items():
        p += w * bundle["estimators"][name].predict_proba(X)[:, 1]
    return np.clip(p, 0.0, 1.0)


def _meta(X, rise, fall):
    X = np.asarray(X)
    rise = np.asarray(rise, dtype=np.float64)
    fall = np.asarray(fall, dtype=np.float64)
    first = X[:, 0] <= 0.5
    hi = X[:, 7] >= 0.5 if X.shape[1] > 7 else np.zeros(len(X), dtype=bool)
    # "short turn so far": pause_start < 4s (feature col 1)
    short = X[:, 1] < 4.0
    # single-pause-ish: first pause AND short
    singleish = first & short
    return first, hi, short, singleish, rise, fall


def gate_shipped(bundle, X, rise, fall):
    """Current train.predict_proba_bundle (for baseline compare)."""
    from train import predict_proba_bundle
    return predict_proba_bundle(bundle, X, rise, fall)


def gate_raw(bundle, X, rise, fall):
    return _raw_p(bundle, X)


def gate_soft_rise_hi_short(bundle, X, rise, fall):
    """Softer rise penalty on Hindi single-pause / short turns only."""
    p = _raw_p(bundle, X)
    first, hi, short, singleish, rise_a, fall_a = _meta(X, rise, fall)
    beta = float(bundle.get("rise_beta", _rise_beta(1.0)))
    # only HI short/singleish: strong soft rise; others untouched by rise
    mask = hi & (singleish | short)
    soften = np.where(mask, beta * 0.15, 0.0)  # only apply on mask
    p = p * (1.0 - soften * np.clip(rise_a / 100.0, 0.0, 0.85))
    return np.clip(p, 0.0, 1.0)


def gate_strong_first_not_done(bundle, X, rise, fall):
    """Stronger not-done on first pause: cut fall boost hard + mild p damp."""
    p = _raw_p(bundle, X)
    first, hi, short, singleish, rise_a, fall_a = _meta(X, rise, fall)
    gamma = float(bundle.get("fall_gamma", _fall_gamma(0.0)))
    # only first pauses get tiny fall; later get full
    gamma_eff = np.where(first, gamma * 0.15, gamma)
    p = p * (1.0 + gamma_eff * np.clip(fall_a / 80.0, -0.3, 0.8))
    # extra not-done: first-pause p *= 0.90
    p = np.where(first, p * 0.90, p)
    return np.clip(p, 0.0, 1.0)


def gate_hi_nudge_105(bundle, X, rise, fall):
    return _hi_nudge(bundle, X, rise, fall, 1.05)


def gate_hi_nudge_108(bundle, X, rise, fall):
    return _hi_nudge(bundle, X, rise, fall, 1.08)


def gate_hi_nudge_112(bundle, X, rise, fall):
    return _hi_nudge(bundle, X, rise, fall, 1.12)


def _hi_nudge(bundle, X, rise, fall, mult):
    """HI p *= mult only when NOT (first-pause AND rising>falling)."""
    p = _raw_p(bundle, X)
    first, hi, short, singleish, rise_a, fall_a = _meta(X, rise, fall)
    rising = rise_a > fall_a + 5.0
    safe = hi & ~ (first & rising)
    p = np.where(safe, p * mult, p)
    return np.clip(p, 0.0, 1.0)


def gate_hi_fall_boost_first_cut(bundle, X, rise, fall):
    """Hindi-only: pi>=1 & fall>rise → *1.08; pi==0 → cut fall hard."""
    p = _raw_p(bundle, X)
    first, hi, short, singleish, rise_a, fall_a = _meta(X, rise, fall)
    gamma = float(bundle.get("fall_gamma", _fall_gamma(1.0)))
    # apply fall with hard cut on first (HI only); others raw
    gamma_eff = np.zeros(len(p))
    gamma_eff = np.where(hi & first, gamma * 0.10, gamma_eff)
    gamma_eff = np.where(hi & (~first), gamma, gamma_eff)
    p = p * (1.0 + gamma_eff * np.clip(fall_a / 80.0, -0.3, 0.8))
    boost = hi & (~first) & (fall_a > rise_a)
    p = np.where(boost, p * 1.08, p)
    return np.clip(p, 0.0, 1.0)


def _write(path, keys, p):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pv in zip(keys, p):
            w.writerow([tid, pi, f"{float(pv):.6f}"])


def _fmt(r):
    return (
        f"delay={r['latency']*1000:.0f} ms  AUC={r['auc']:.3f}  "
        f"cut={100*r['cutoff']:.1f}%  thr={r['threshold']} d={r['delay']*1000:.0f}"
    )


def score_dir(data_dir, keys, p, tag):
    path = os.path.join(OUT_DIR, f"{tag}.csv")
    _write(path, keys, p)
    r = score(os.path.join(data_dir, "labels.csv"), path)
    return r, path


def load_or_build():
    if os.path.isfile(CACHE):
        print(f"cache hit {CACHE}")
        return joblib.load(CACHE)
    print("building feature cache (once)...")
    en = build_dataset(os.path.join(ROOT, "..", "eot_data", "english"))
    hi = build_dataset(os.path.join(ROOT, "..", "eot_data", "hindi"))
    bundle = joblib.load(os.path.join(MODEL_DIR, "unified.joblib"))
    # val for calibration
    val_dir = os.path.join(ROOT, "..", "eot_splits", "hindi", "val")
    train_dir = os.path.join(ROOT, "..", "eot_splits", "hindi", "train")
    hi_val = build_dataset(val_dir) if os.path.isfile(os.path.join(val_dir, "labels.csv")) else None
    hi_tr = build_dataset(train_dir) if os.path.isfile(os.path.join(train_dir, "labels.csv")) else None
    blob = {"en": en, "hi": hi, "bundle": bundle, "hi_val": hi_val, "hi_tr": hi_tr}
    joblib.dump(blob, CACHE)
    print(f"wrote {CACHE}")
    return blob


def oof_hi(hi, gate_fn, bundle_template):
    """5-fold OOF with raw ensemble per fold, then gate_fn (gate gets a fake bundle)."""
    X, y, groups = hi["X"], hi["y"], hi["groups"]
    rise, fall = hi["rise"], hi["fall"]
    gkf = GroupKFold(n_splits=5)
    p_oof = np.zeros(len(y), dtype=np.float64)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        sw = _sample_weights([hi["labels"][i] for i in tr], hi["lang_id"])
        ests = _build_estimators()
        for name, est in ests.items():
            _fit_one(name, est, X[tr], y[tr], sw)
        fold_bundle = {
            "estimators": ests,
            "weights": ENSEMBLE_WEIGHTS,
            "rise_beta": _rise_beta(hi["lang_id"]),
            "fall_gamma": _fall_gamma(hi["lang_id"]),
        }
        p_oof[te] = gate_fn(fold_bundle, X[te], rise[te], fall[te])
    return p_oof


def calibrate_isotonic_hi(blob, gate_base=gate_raw):
    """Fit isotonic on HI val raw/base probs → apply to handout HI + return calibrator."""
    from sklearn.isotonic import IsotonicRegression

    hi_tr, hi_val = blob["hi_tr"], blob["hi_val"]
    if hi_tr is None or hi_val is None:
        return None
    # train a fold-like model on train split only
    sw = _sample_weights(hi_tr["labels"], hi_tr["lang_id"])
    # pool EN train if exists
    en_tr_dir = os.path.join(ROOT, "..", "eot_splits", "english", "train")
    Xtr, ytr = hi_tr["X"], hi_tr["y"]
    labs = list(hi_tr["labels"])
    if os.path.isfile(os.path.join(en_tr_dir, "labels.csv")):
        en_tr = build_dataset(en_tr_dir)
        Xtr = np.vstack([Xtr, en_tr["X"]])
        ytr = np.concatenate([ytr, en_tr["y"]])
        labs = labs + list(en_tr["labels"])
        sw = _sample_weights(labs, hi_tr["lang_id"])
    ests = _build_estimators()
    for name, est in ests.items():
        _fit_one(name, est, Xtr, ytr, sw)
    b = {
        "estimators": ests,
        "weights": ENSEMBLE_WEIGHTS,
        "rise_beta": _rise_beta(1.0),
        "fall_gamma": _fall_gamma(1.0),
    }
    p_val = gate_base(b, hi_val["X"], hi_val["rise"], hi_val["fall"])
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_val, hi_val["y"])

    def gate_cal(bundle, X, rise, fall):
        # use SHIPPED unified bundle's raw then calibrate — apples for handout
        # For OOF we recalibrate isn't ideal; apply iso to gate_base output
        return np.clip(iso.predict(gate_base(bundle, X, rise, fall)), 0.0, 1.0)

    return gate_cal, b, iso


def main():
    blob = load_or_build()
    en, hi, bundle = blob["en"], blob["hi"], blob["bundle"]
    en_dir = os.path.join(ROOT, "..", "eot_data", "english")
    hi_dir = os.path.join(ROOT, "..", "eot_data", "hindi")

    experiments = [
        ("00_shipped_baseline", gate_shipped),
        ("01_raw_ensemble", gate_raw),
        ("02_soft_rise_hi_short", gate_soft_rise_hi_short),
        ("03_strong_first_not_done", gate_strong_first_not_done),
        ("04_hi_nudge_1.05", gate_hi_nudge_105),
        ("05_hi_nudge_1.08", gate_hi_nudge_108),
        ("06_hi_nudge_1.12", gate_hi_nudge_112),
        ("07_hi_fallboost_firstcut", gate_hi_fall_boost_first_cut),
    ]

    # calibration as separate experiment (alone on raw)
    try:
        cal = calibrate_isotonic_hi(blob, gate_base=gate_raw)
        if cal is not None:
            gate_cal, _, _ = cal
            experiments.append(("08_isotonic_on_raw", gate_cal))
    except Exception as e:
        print(f"calibration skipped: {e}")

    rows_out = []
    print("\n" + "=" * 78)
    print("GATE EXPERIMENTS (each ALONE on raw, except 00=shipped)")
    print("Deliverables NOT modified.")
    print("=" * 78)

    for name, fn in experiments:
        print(f"\n--- {name} ---")
        p_en = fn(bundle, en["X"], en["rise"], en["fall"])
        p_hi = fn(bundle, hi["X"], hi["rise"], hi["fall"])
        r_en, _ = score_dir(en_dir, en["keys"], p_en, f"{name}_en")
        r_hi, _ = score_dir(hi_dir, hi["keys"], p_hi, f"{name}_hi")
        print(f"  EN handout: {_fmt(r_en)}")
        print(f"  HI handout: {_fmt(r_hi)}")

        # HI OOF (slow-ish but no deliverable change)
        p_oof = oof_hi(hi, fn, bundle)
        r_oof, _ = score_dir(hi_dir, hi["keys"], p_oof, f"{name}_oof_hi")
        print(f"  HI OOF:     {_fmt(r_oof)}")

        rows_out.append({
            "name": name,
            "en_delay": r_en["latency"] * 1000,
            "en_auc": r_en["auc"],
            "hi_delay": r_hi["latency"] * 1000,
            "hi_auc": r_hi["auc"],
            "oof_hi_delay": r_oof["latency"] * 1000,
            "oof_hi_auc": r_oof["auc"],
            "sum_handout": (r_en["latency"] + r_hi["latency"]) * 1000,
        })

    print("\n" + "=" * 78)
    print("SUMMARY (lower delay better; prefer HI OOF not rising to 850)")
    print(f"{'exp':<28} {'EN':>7} {'HI':>7} {'sum':>7} {'OOF_HI':>7} {'OOF_AUC':>8}")
    base_en = rows_out[0]["en_delay"]
    base_oof = rows_out[0]["oof_hi_delay"]
    for r in rows_out:
        flag = ""
        if r["name"] != "00_shipped_baseline":
            if r["en_delay"] > base_en + 1:
                flag += " EN↑"
            if r["oof_hi_delay"] >= 849.0:
                flag += " OOF~850"
            if (
                r["sum_handout"] < rows_out[0]["sum_handout"] - 1
                and r["oof_hi_delay"] <= base_oof + 5
                and r["en_delay"] <= base_en + 1
            ):
                flag += " ★"
        print(
            f"{r['name']:<28} {r['en_delay']:7.0f} {r['hi_delay']:7.0f} "
            f"{r['sum_handout']:7.0f} {r['oof_hi_delay']:7.0f} {r['oof_hi_auc']:8.3f}{flag}"
        )

    summary_path = os.path.join(OUT_DIR, "SUMMARY.txt")
    with open(summary_path, "w") as f:
        f.write("GATE EXPERIMENTS — each alone; deliverables untouched\n")
        for r in rows_out:
            f.write(
                f"{r['name']}: EN={r['en_delay']:.0f} HI={r['hi_delay']:.0f} "
                f"sum={r['sum_handout']:.0f} OOF_HI={r['oof_hi_delay']:.0f} "
                f"OOF_AUC={r['oof_hi_auc']:.3f}\n"
            )
    print(f"\nwrote {summary_path}")
    print("CSVs under starter/_exp_gates_out/ (safe to delete)")


if __name__ == "__main__":
    main()
