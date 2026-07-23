"""Split eot_data into turn-level 60/20/20 train/val/test folders.

Uses GroupShuffleSplit by turn_id so pauses from the same turn never
cross splits (avoids speaker/rhythm leakage).

Target fractions are of *turns*, not pause rows:
  ~60% train / ~20% val / ~20% test

    python make_splits.py
    # -> ../eot_splits/{english,hindi}/{train,val,test}/{labels.csv,audio/}
    # + ../eot_splits/{lang}/split_meta.json
"""
import argparse
import csv
import json
import os
import shutil
from collections import defaultdict

import numpy as np
from sklearn.model_selection import GroupShuffleSplit


ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(ROOT, "..", "eot_data")
DEFAULT_DST = os.path.join(ROOT, "..", "eot_splits")

# Fractions of turns
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20


def _link_or_copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _write_split(src_lang, out_dir, turn_set, by_turn):
    audio_out = os.path.join(out_dir, "audio")
    os.makedirs(audio_out, exist_ok=True)
    split_rows = []
    for tid in sorted(turn_set):
        for r in sorted(by_turn[tid], key=lambda z: int(z["pause_index"])):
            split_rows.append(r)
            src_wav = os.path.join(src_lang, r["audio_file"])
            dst_wav = os.path.join(out_dir, r["audio_file"])
            _link_or_copy(src_wav, dst_wav)

    with open(os.path.join(out_dir, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "turn_id", "audio_file", "pause_index",
                "pause_start", "pause_end", "label",
            ],
        )
        w.writeheader()
        w.writerows(split_rows)
    return split_rows


def _clear_split_dirs(dst_lang):
    """Remove old train/val/test so a 70/30 layout cannot linger."""
    for name in ("train", "val", "test"):
        path = os.path.join(dst_lang, name)
        if os.path.isdir(path):
            shutil.rmtree(path)


def split_language(src_lang, dst_lang, seed=0):
    """60/20/20 by turn_id via two GroupShuffleSplits.

    1) hold out 40% of turns -> temp
    2) split temp 50/50 -> val / test  (each ~20% of all turns)
    Remaining ~60% -> train
    """
    labels_path = os.path.join(src_lang, "labels.csv")
    rows = list(csv.DictReader(open(labels_path)))
    by_turn = defaultdict(list)
    for r in rows:
        by_turn[r["turn_id"]].append(r)

    turns = np.array(sorted(by_turn.keys()))
    X = np.zeros((len(turns), 1))
    y = np.zeros(len(turns))

    # Step 1: 60% train vs 40% held-out
    tr_idx, hold_idx = next(
        GroupShuffleSplit(n_splits=1, test_size=VAL_FRAC + TEST_FRAC, random_state=seed)
        .split(X, y, turns)
    )
    train_turns = set(turns[tr_idx])
    hold_turns = turns[hold_idx]

    # Step 2: split held-out into val/test (~20%/20% of all turns)
    Xh = np.zeros((len(hold_turns), 1))
    yh = np.zeros(len(hold_turns))
    # test_size relative to hold set: TEST_FRAC / (VAL_FRAC + TEST_FRAC) = 0.5
    va_idx, te_idx = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
            random_state=seed,
        ).split(Xh, yh, hold_turns)
    )
    val_turns = set(hold_turns[va_idx])
    test_turns = set(hold_turns[te_idx])

    # Safety: no turn in more than one split
    assert not (train_turns & val_turns)
    assert not (train_turns & test_turns)
    assert not (val_turns & test_turns)
    assert train_turns | val_turns | test_turns == set(turns)

    _clear_split_dirs(dst_lang)
    os.makedirs(dst_lang, exist_ok=True)

    meta = {
        "src": os.path.abspath(src_lang),
        "protocol": "GroupShuffleSplit by turn_id; 60/20/20 train/val/test of turns",
        "rationale": (
            "Never put pauses from the same turn in different splits — "
            "if pause #2 of en__001 is train and pause #5 of the same turn "
            "is test, the model memorizes voice/rhythm (speaker leakage)."
        ),
        "train_frac_target": TRAIN_FRAC,
        "val_frac_target": VAL_FRAC,
        "test_frac_target": TEST_FRAC,
        "seed": seed,
        "n_total_turns": len(turns),
        "n_total_pauses": len(rows),
        "n_train_turns": len(train_turns),
        "n_val_turns": len(val_turns),
        "n_test_turns": len(test_turns),
        "train_turns": sorted(train_turns),
        "val_turns": sorted(val_turns),
        "test_turns": sorted(test_turns),
    }

    for split_name, turn_set in (
        ("train", train_turns),
        ("val", val_turns),
        ("test", test_turns),
    ):
        out_dir = os.path.join(dst_lang, split_name)
        split_rows = _write_split(src_lang, out_dir, turn_set, by_turn)
        meta[f"n_{split_name}_pauses"] = len(split_rows)
        print(
            f"{os.path.basename(src_lang)}/{split_name}: "
            f"{len(turn_set)} turns, {len(split_rows)} pauses -> {out_dir}"
        )

    with open(os.path.join(dst_lang, "split_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dst", default=DEFAULT_DST)
    ap.add_argument("--seed", type=int, default=0,
                    help="Fixed seed; applied independently per language")
    args = ap.parse_args()

    for lang in ("english", "hindi"):
        src = os.path.join(args.src, lang)
        dst = os.path.join(args.dst, lang)
        if not os.path.isfile(os.path.join(src, "labels.csv")):
            raise SystemExit(f"missing {src}/labels.csv")
        split_language(src, dst, seed=args.seed)
    print(f"done -> {os.path.abspath(args.dst)} (seed={args.seed})")


if __name__ == "__main__":
    main()
