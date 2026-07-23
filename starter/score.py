"""Scorer: simulates a live voice agent using your p_eot scores.

Policy: at each pause the agent waits `delay` seconds of silence; if your
p_eot >= `threshold` it takes the floor at that moment, otherwise it keeps
waiting until a 1.6 s timeout forces a response (on true turn ends).

- Firing during a HOLD pause (before the user resumes) = FALSE CUTOFF.
  A turn with any false cutoff counts as an interrupted turn.
- On the true EOT pause, your response delay = the action delay (or the
  1.6 s timeout if your score never crossed the threshold).

We sweep (threshold x delay) and report the best MEAN DELAY achievable
while interrupting <= 5% of turns. Lower is better.

    python score.py --data_dir eot_data/english --pred predictions.csv
"""
import argparse
import csv
import os

import numpy as np

TIMEOUT_S = 1.6
THRESHOLDS = np.round(np.arange(0.05, 1.0, 0.05), 3)
DELAYS = np.round(np.arange(0.10, 1.65, 0.05), 3)

# Val-only OP rule (do NOT re-sweep on test):
# sweep these thresholds; pick the HIGHEST with FCR <= 4.5% (0.5% safety buffer).
VAL_THRESHOLDS = (0.5, 0.6, 0.7, 0.8)
FCR_CAP = 0.045


def load(labels_csv, pred_csv):
    preds = {}
    with open(pred_csv) as f:
        for r in csv.DictReader(f):
            preds[(r["turn_id"], int(r["pause_index"]))] = float(r["p_eot"])
    pauses = []
    with open(labels_csv) as f:
        for r in csv.DictReader(f):
            key = (r["turn_id"], int(r["pause_index"]))
            if key not in preds:
                raise SystemExit(f"missing prediction for {key}")
            pauses.append({
                "turn_id": r["turn_id"],
                "dur": float(r["pause_end"]) - float(r["pause_start"]),
                "label": r["label"],
                "p": preds[key],
            })
    return pauses


def evaluate(pauses, threshold, delay):
    turns_cut = set()
    turn_ids = set()
    latencies = []
    for pz in pauses:
        turn_ids.add(pz["turn_id"])
        fires = pz["p"] >= threshold
        if pz["label"] == "hold":
            # false cutoff only if the delay elapses before the user resumes
            if fires and delay < pz["dur"]:
                turns_cut.add(pz["turn_id"])
        else:  # true end of turn
            latencies.append(delay if fires else TIMEOUT_S)
    cutoff_rate = len(turns_cut) / max(1, len(turn_ids))
    return cutoff_rate, float(np.mean(latencies)) if latencies else TIMEOUT_S


def _auc_and_counts(pauses):
    y = np.array([1 if p["label"] == "eot" else 0 for p in pauses])
    s = np.array([p["p"] for p in pauses])
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), len(y) - y.sum()
    auc = ((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else float("nan")
    return float(auc), len({p["turn_id"] for p in pauses}), len(pauses)


def select_val_operating_point(pauses, fcr_cap=FCR_CAP, delays=DELAYS):
    """Freeze OP on validation only.

    Sweep thresholds in VAL_THRESHOLDS. Among those that can meet FCR <= fcr_cap
    for some delay, pick the HIGHEST threshold (not the lowest-delay one).
    Then, for that frozen threshold, pick the delay that minimizes mean latency
    while still meeting the FCR cap.
    """
    feasible = []
    for t in VAL_THRESHOLDS:
        for d in delays:
            cut, _ = evaluate(pauses, t, d)
            if cut <= fcr_cap:
                feasible.append(t)
                break

    if feasible:
        t_star = float(max(feasible))  # highest thr under FCR cap — safety over latency
    else:
        t_star = float(VAL_THRESHOLDS[-1])  # none meet: stay conservative

    best = None
    for d in delays:
        cut, lat = evaluate(pauses, t_star, d)
        if cut <= fcr_cap and (best is None or lat < best["latency"]):
            best = {
                "latency": float(lat),
                "cutoff": float(cut),
                "threshold": t_star,
                "delay": float(d),
            }
    if best is None:
        # threshold frozen but no delay under cap → safest never-fire-ish delay
        cut, lat = evaluate(pauses, t_star, TIMEOUT_S)
        best = {
            "latency": float(lat),
            "cutoff": float(cut),
            "threshold": t_star,
            "delay": float(TIMEOUT_S),
        }

    auc, n_turns, n_pauses = _auc_and_counts(pauses)
    best["auc"] = auc
    best["n_turns"] = n_turns
    best["n_pauses"] = n_pauses
    best["fcr_cap"] = float(fcr_cap)
    best["feasible_thresholds"] = [float(t) for t in feasible]
    return best


def score(labels_csv, pred_csv, budget=0.05):
    pauses = load(labels_csv, pred_csv)
    best = None
    for t in THRESHOLDS:
        for d in DELAYS:
            cut, lat = evaluate(pauses, t, d)
            if cut <= budget and (best is None or lat < best["latency"]):
                best = {"latency": lat, "cutoff": cut, "threshold": t, "delay": d}
    if best is None:  # nothing meets budget: report the never-fire policy
        best = {"latency": TIMEOUT_S, "cutoff": 0.0, "threshold": 1.0, "delay": TIMEOUT_S}
    auc, n_turns, n_pauses = _auc_and_counts(pauses)
    best["auc"] = auc
    best["n_turns"] = n_turns
    best["n_pauses"] = n_pauses
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--budget", type=float, default=0.05)
    args = ap.parse_args()
    r = score(os.path.join(args.data_dir, "labels.csv"), args.pred, args.budget)
    print(f"turns={r['n_turns']}  pauses={r['n_pauses']}  AUC={r['auc']:.3f}")
    print(f"BEST @ <= {int(args.budget*100)}% interrupted turns:")
    print(f"  mean response delay : {r['latency']*1000:.0f} ms   <-- your score, lower is better")
    print(f"  interrupted turns   : {r['cutoff']*100:.1f}%")
    print(f"  operating point     : threshold={r['threshold']}, delay={r['delay']*1000:.0f} ms")


if __name__ == "__main__":
    main()
