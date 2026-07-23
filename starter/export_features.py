"""Export Tier-1 causal feature CSVs for analysis (not for training).

Reuses the same extract_features call path as train.py.build_dataset.
pause_end / label are written as metadata only; features remain causal
(audio 0..pause_start only, enforced inside extract_features).

    python export_features.py
    python export_features.py --data_root ../eot_data --out_dir feature_data
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np

from features import (
    FEATURE_NAMES,
    N_FEATURES,
    detect_lang_id,
    extract_features,
    load_wav,
    precompute_contours,
)

META_COLS = [
    "turn_id",
    "pause_index",
    "label",
    "audio_file",
    "pause_start",
    "pause_end",
]


def _last_pause_ends(rows):
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
            prev = float(r["pause_end"])
    return last_end


def export_language(data_dir, out_path, language=None):
    """Write one language's feature CSV. Returns list of row dicts."""
    labels_path = os.path.join(data_dir, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))
    lang_id = detect_lang_id(data_dir, rows)
    if language is None:
        language = "hindi" if lang_id >= 0.5 else "english"
    last_end = _last_pause_ends(rows)

    cache = {}
    contour_cache = {}
    out_rows = []
    for r in rows:
        path = os.path.join(data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
            contour_cache[path] = precompute_contours(*cache[path])
        x, sr = cache[path]
        tid, pi = r["turn_id"], int(r["pause_index"])
        feat, _rs, _fs = extract_features(
            x, sr, float(r["pause_start"]), pi,
            last_pause_end=last_end[(tid, pi)],
            contours=contour_cache[path],
            lang_id=lang_id,
        )
        if len(feat) != N_FEATURES:
            raise RuntimeError(
                f"feature length {len(feat)} != N_FEATURES {N_FEATURES}"
            )
        row = {
            "turn_id": tid,
            "pause_index": pi,
            "label": r["label"],
            "audio_file": r["audio_file"],
            "pause_start": float(r["pause_start"]),
            "pause_end": float(r["pause_end"]),
            "language": language,
        }
        for name, val in zip(FEATURE_NAMES, feat):
            row[name] = float(val)
        out_rows.append(row)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    # Header: metadata, then FEATURE_NAMES (may repeat pause_index/pause_start).
    # When writing, emit meta then features with FEATURE_NAMES column names.
    fieldnames = META_COLS + list(FEATURE_NAMES)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for row in out_rows:
            meta = [row[c] for c in META_COLS]
            feats = [row[name] for name in FEATURE_NAMES]
            w.writerow(meta + feats)

    return out_rows, fieldnames


def _summarize(name, rows, fieldnames):
    n = len(rows)
    labels = [r["label"] for r in rows]
    n_hold = sum(1 for lab in labels if lab == "hold")
    n_eot = sum(1 for lab in labels if lab == "eot")
    n_meta = len(META_COLS)
    n_feat = len(FEATURE_NAMES)
    print(f"\n=== {name} ===")
    print(f"shape: {n} rows x {len(fieldnames)} cols "
          f"({n_meta} metadata + {n_feat} features)")
    print(f"feature columns ({n_feat}): {FEATURE_NAMES}")
    print(f"label balance: hold={n_hold} ({n_hold / n:.1%}), "
          f"eot={n_eot} ({n_eot / n:.1%})")
    key_feats = [
        "energy_decay_slope",
        "f0_slope_last_voiced",
        "rel_final_pitch_med",
        "final_lengthening",
        "ipi",
    ]
    for kf in key_feats:
        if kf not in FEATURE_NAMES:
            continue
        by_lab = {"hold": [], "eot": []}
        for r in rows:
            by_lab.setdefault(r["label"], []).append(r[kf])
        parts = []
        for lab in ("hold", "eot"):
            vals = by_lab.get(lab) or []
            if vals:
                parts.append(f"{lab} mean={np.mean(vals):.4f}")
        print(f"  {kf}: {', '.join(parts)}")


def main():
    ap = argparse.ArgumentParser(description="Export Tier-1 feature CSVs")
    ap.add_argument(
        "--data_root",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "eot_data"
        ),
    )
    ap.add_argument(
        "--out_dir",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "feature_data"
        ),
    )
    args = ap.parse_args()
    data_root = os.path.abspath(args.data_root)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    langs = [
        ("english", os.path.join(data_root, "english"),
         os.path.join(out_dir, "english_features.csv")),
        ("hindi", os.path.join(data_root, "hindi"),
         os.path.join(out_dir, "hindi_features.csv")),
    ]

    all_rows = []
    for language, data_dir, out_path in langs:
        if not os.path.isdir(data_dir):
            print(f"SKIP missing data_dir: {data_dir}")
            continue
        n_labels = sum(1 for _ in open(os.path.join(data_dir, "labels.csv"))) - 1
        rows, fieldnames = export_language(data_dir, out_path, language=language)
        if len(rows) != n_labels:
            print(
                f"MISMATCH {language}: exported {len(rows)} rows "
                f"vs labels.csv {n_labels}"
            )
        else:
            print(f"OK {language}: {len(rows)} rows == labels.csv")
        expected_cols = len(META_COLS) + N_FEATURES
        if len(fieldnames) != expected_cols:
            print(
                f"MISMATCH cols: got {len(fieldnames)}, "
                f"expected {expected_cols}"
            )
        print(f"wrote -> {out_path}")
        _summarize(language, rows, fieldnames)
        all_rows.extend(rows)

    if all_rows:
        all_path = os.path.join(out_dir, "all_features.csv")
        # Combined file: language + metadata + FEATURE_NAMES
        all_fields = ["language"] + META_COLS + list(FEATURE_NAMES)
        with open(all_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(all_fields)
            for row in all_rows:
                w.writerow(
                    [row["language"]]
                    + [row[c] for c in META_COLS]
                    + [row[name] for name in FEATURE_NAMES]
                )
        print(f"\nwrote -> {all_path}")
        _summarize("all", all_rows, all_fields)

    print(
        f"\nDone. N_FEATURES={N_FEATURES}, "
        f"META_COLS={len(META_COLS)}, "
        f"note: pause_index/pause_start appear in both metadata and FEATURE_NAMES."
    )


if __name__ == "__main__":
    main()
