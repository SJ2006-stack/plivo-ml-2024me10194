"""Audio utilities + causal prosodic feature extraction for EOT.

HARD CAUSALITY RULE (non-negotiable)
------------------------------------
For a pause at `pause_start`, features may use ONLY audio from time 0 up to
`pause_start`. Never audio after the pause. Never use `pause_end` or pause
duration as a feature.

Enforcement: `extract_features` slices once
    x_causal = x[:int(pause_start * sr)]
and computes ALL features only from that prefix (or from contours truncated
to frames fully contained in that prefix). A live agent cannot hear the future.
"""
import io
import sys
import types

import numpy as np
import soundfile as sf

# This env's Python was built without _lzma; librosa.pyin → pooch/joblib imports
# lzma at module load. Stub is enough for pitch tracking (we never compress).
if "lzma" not in sys.modules:
    try:
        import lzma as _lzma  # noqa: F401
    except Exception:
        class _LZMAError(Exception):
            pass

        class _LZMAFile(io.BufferedIOBase):
            def __init__(self, *a, **k):
                raise _LZMAError("lzma unavailable in this Python build")

        _m = types.ModuleType("lzma")
        _m.LZMAError = _LZMAError
        _m.LZMAFile = _LZMAFile
        _m.FORMAT_AUTO = 0
        _m.FORMAT_XZ = 1
        _m.FORMAT_ALONE = 2
        _m.FORMAT_RAW = 3
        _m.CHECK_NONE = 0
        _m.CHECK_CRC32 = 1
        _m.CHECK_CRC64 = 4
        _m.CHECK_SHA256 = 10
        _m.open = lambda *a, **k: (_ for _ in ()).throw(_LZMAError("no lzma"))
        _m.compress = lambda *a, **k: (_ for _ in ()).throw(_LZMAError("no lzma"))
        _m.decompress = lambda *a, **k: (_ for _ in ()).throw(_LZMAError("no lzma"))
        sys.modules["lzma"] = _m

import librosa  # noqa: E402

FRAME_MS = 25
HOP_MS = 10
F0_FRAME_MS = 40  # legacy energy-frame helper only
HOP_S = HOP_MS / 1000.0

# Pitch tracker: librosa.piptrack (old working path; fmin=50 for Hindi, no pyin).
F0_FRAME_LENGTH = 2048
F0_HOP_LENGTH = 512
F0_FMIN = 50.0
F0_FMAX = 400.0

# Feature vector layout (see FEATURE_NAMES). Keep N_FEATURES in sync.
FEATURE_NAMES = [
    # structural (causal: known at pause onset)
    "pause_index",
    "pause_start",
    "ipi",
    "log_pause_start",
    "log_ipi",
    "is_first_pause",
    "is_late_pause",
    "lang_id",
    # energy / decay
    "energy_mean_local",
    "energy_final",
    "energy_decay_slope",
    "energy_decay_drop",
    "energy_slope_short",
    "pre_pause_rms_db",
    "rms_ratio_50_200",
    "energy_vs_turn_med",
    # F0 / prosody
    "f0_mean_turn",
    "f0_median_turn",
    "f0_final",
    "f0_std_local",
    "f0_slope_last_voiced",
    "f0_slope_terminal",
    "rel_final_pitch_med",
    "rel_final_pitch_mean",
    "fall_flag",
    "rise_flag",
    "fall_score",
    "rise_score",
    # lengthening / rate
    "final_voiced_dur",
    "mean_voiced_run_dur",
    "final_lengthening",
    "final_lengthening_med",
    "speaking_rate",
    # voicing / spectral
    "voiced_frac_turn",
    "voiced_frac_local",
    "n_voiced_runs",
    "trailing_unvoiced_s",
    "spec_centroid",
    "spec_bandwidth",
    "spec_flatness",
    "spec_low_high",
    "zcr",
]
N_FEATURES = len(FEATURE_NAMES)


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def speech_before(x, sr, pause_start, window_s=0.8):
    """The last `window_s` seconds of audio strictly before the pause."""
    end = int(pause_start * sr)
    start = max(0, end - int(window_s * sr))
    return x[start:end]


def frames(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x, sr):
    """Short-time energy per frame, in dB."""
    fr = frames(x, sr)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def autocorr_f0(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.30):
    """Legacy autocorrelation F0 (kept for reference; prefer f0_contour/piptrack)."""
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


def f0_hop_s(sr, hop_length=F0_HOP_LENGTH):
    return float(hop_length) / float(sr)


def f0_contour(
    x,
    sr,
    fmin=F0_FMIN,
    fmax=F0_FMAX,
    frame_length=F0_FRAME_LENGTH,
    hop_length=F0_HOP_LENGTH,
):
    """Per-frame F0 (Hz) via librosa.piptrack; 0.0 where unvoiced.

    OLD WORKING PITCH (replaces broken pyin). fmin=50 for Hindi male F0,
    without pyin's destructive voicing guard.
    """
    y = np.asarray(x, dtype=np.float32)
    if len(y) < 2:
        return np.zeros(0, dtype=np.float32)
    # OLD WORKING PITCH (Replaces the broken pyin)
    pitches, magnitudes = librosa.piptrack(
        y=y, sr=sr, fmin=fmin, fmax=fmax, hop_length=hop_length,
    )
    f0 = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        if magnitudes[index, t] > 0.5:
            f0.append(float(pitches[index, t]))
        else:
            f0.append(0.0)
    f0 = np.asarray(f0, dtype=np.float32)
    return f0


def _slope(y):
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return 0.0
    t = np.arange(len(y), dtype=float)
    return float(np.polyfit(t, y, 1)[0])


def _rms_db(a):
    rms = float(np.sqrt(np.mean(np.asarray(a, dtype=float) ** 2) + 1e-12))
    return 20.0 * np.log10(rms + 1e-12)


def precompute_contours(x, sr):
    """Cache per-file energy/F0.

    Contours are frame-local; truncating to pause_start is causal. Prefer
    computing on the causal prefix inside extract_features when no cache.
    Energy hop (10 ms) and F0 hop (piptrack hop_length/sr) differ on purpose.
    """
    return {
        "energy": frame_energy_db(x, sr),
        "f0": f0_contour(x, sr),
        "hop": HOP_S,
        "f0_hop": f0_hop_s(sr),
        "n_samples": len(x),
    }


def _n_causal_frames(n_samples, sr, frame_ms, hop_ms=HOP_MS):
    """Number of frames fully contained in a causal prefix of n_samples."""
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if n_samples < fl:
        return 0
    return 1 + (n_samples - fl) // hp


def _n_causal_f0_frames(n_samples, frame_length=F0_FRAME_LENGTH,
                        hop_length=F0_HOP_LENGTH):
    if n_samples < frame_length:
        return 0
    return 1 + (n_samples - frame_length) // hop_length


def _causal_contours(x_causal, sr, contours=None):
    """Energy/F0 for the causal prefix only (no frames past pause_start).

    Returns (energy, f0, energy_hop_s, f0_hop_s).
    """
    n = len(x_causal)
    hop_e = HOP_S
    hop_f = f0_hop_s(sr)
    if contours is not None:
        hop_e = contours.get("hop", HOP_S)
        hop_f = contours.get("f0_hop", hop_f)
        n_e = _n_causal_frames(n, sr, FRAME_MS)
        n_f = _n_causal_f0_frames(n)
        e = contours["energy"][:n_e]
        f0 = contours["f0"][:n_f]
        if len(e) < 2:
            e = frame_energy_db(x_causal, sr)
        if len(f0) < 2:
            f0 = f0_contour(x_causal, sr)
        return e, f0, hop_e, hop_f
    return frame_energy_db(x_causal, sr), f0_contour(x_causal, sr), hop_e, hop_f


def _voiced_runs(f0, hop):
    """Continuous voiced runs as (start_i, end_i_inclusive, duration_s)."""
    voiced = np.asarray(f0) > 0
    runs = []
    i = 0
    n = len(voiced)
    while i < n:
        if voiced[i]:
            j = i
            while j < n and voiced[j]:
                j += 1
            runs.append((i, j - 1, (j - i) * hop))
            i = j
        else:
            i += 1
    return runs


# ---------------------------------------------------------------------------
# Tier-1 causal helpers
# ---------------------------------------------------------------------------

def f0_slope_last_voiced(f0, hop=HOP_S):
    """F0 slope over the last continuous voiced stretch.

    Final fall (negative) → more EOT; flat/rise → more hold.
    Returns (slope_hz_per_frame, final_f0, run_duration_s).

    If <30% of frames are voiced, slope is forced to 0 (avoid garbage
    intonation from sparse / failed pitch tracks — common on low-F0 speech).
    """
    f0 = np.asarray(f0)
    if len(f0) == 0:
        return 0.0, 0.0, 0.0
    voiced_frac = float(np.mean(f0 > 0))
    runs = _voiced_runs(f0, hop)
    if not runs:
        return 0.0, 0.0, 0.0
    start, end, dur = runs[-1]
    run = f0[start: end + 1]
    run = run[run > 0]
    final_f0 = float(run[-min(5, len(run)):].mean()) if len(run) else 0.0
    if voiced_frac < 0.3:
        return 0.0, final_f0, dur
    if len(run) < 3:
        return 0.0, final_f0, dur
    return _slope(run), final_f0, dur


def energy_decay_into_pause(energy, hop=HOP_S, window_s=0.4):
    """Slope and drop of short-time energy in the last ~window_s before pause.

    Decay into the pause (negative slope / positive drop) → EOT cue.
    Returns (slope, drop=early_mean - late_mean, final_energy).
    """
    if energy is None or len(energy) == 0:
        return 0.0, 0.0, 0.0
    n = max(3, int(window_s / hop))
    w = energy[-n:]
    slope = _slope(w) if len(w) >= 3 else 0.0
    mid = max(1, len(w) // 2)
    early = float(w[:mid].mean())
    late = float(w[mid:].mean())
    drop = early - late
    final_e = float(w[-min(5, len(w)):].mean())
    return slope, drop, final_e


def final_lengthening_ratio(f0, hop=HOP_S):
    """Duration of last voiced run vs mean/median voiced-run length so far.

    Must be given the FULL causal F0 contour (0..pause_start), not a short
    local window. Returns (ratio_vs_mean, ratio_vs_median, final_dur, mean_dur).
    """
    runs = _voiced_runs(f0, hop)
    if not runs:
        return 1.0, 1.0, 0.0, 0.0
    durs = np.array([r[2] for r in runs], dtype=float)
    final_dur = float(durs[-1])
    # Compare against earlier runs when possible; else vs all (incl. final)
    prior = durs[:-1] if len(durs) > 1 else durs
    mean_dur = float(prior.mean())
    med_dur = float(np.median(prior))
    ratio_mean = final_dur / max(mean_dur, 1e-3)
    ratio_med = final_dur / max(med_dur, 1e-3)
    return ratio_mean, ratio_med, final_dur, mean_dur


def relative_final_pitch(f0_turn, hop=HOP_S, local_window_s=0.8):
    """Last voiced F0 vs mean/median F0 of the turn so far.

    Low relative pitch → EOT. Uses full causal turn contour for the reference;
    final F0 is taken from the last voiced stretch (optionally preferring the
    last local_window_s for the final estimate if that stretch is long).
    Returns (rel_med, rel_mean, f0_final, f0_mean, f0_median).
    """
    v = f0_turn[f0_turn > 0]
    if len(v) == 0:
        return 1.0, 1.0, 0.0, 0.0, 0.0
    f0_mean = float(v.mean())
    f0_median = float(np.median(v))
    _, f0_final, _ = f0_slope_last_voiced(f0_turn, hop)
    if f0_final <= 0:
        f0_final = float(v[-min(5, len(v)):].mean())
    # Optional: if last 0.8 s has voiced frames, use their trailing mean as final
    n_loc = int(local_window_s / hop)
    if n_loc > 0 and len(f0_turn) > 0:
        local = f0_turn[-n_loc:]
        lv = local[local > 0]
        if len(lv) >= 3:
            f0_final = float(lv[-min(5, len(lv)):].mean())
    rel_med = f0_final / max(f0_median, 1.0)
    rel_mean = f0_final / max(f0_mean, 1.0)
    return rel_med, rel_mean, f0_final, f0_mean, f0_median


def _spectral_tail(seg, sr, dur=0.12):
    """Centroid / bandwidth / flatness / low-high / ZCR on last `dur` seconds."""
    n = int(dur * sr)
    last = seg[-n:] if len(seg) > n else seg
    if len(last) < 64:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    w = last * np.hanning(len(last))
    spec = np.abs(np.fft.rfft(w)) + 1e-12
    freqs = np.fft.rfftfreq(len(last), 1.0 / sr)
    centroid = float((freqs * spec).sum() / spec.sum())
    bw = float(np.sqrt(((freqs - centroid) ** 2 * spec).sum() / spec.sum()))
    flat = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
    low = float(spec[freqs < 800].sum() / spec.sum())
    high = float(spec[freqs >= 3000].sum() / spec.sum())
    zcr = float(np.sum(np.abs(np.diff(np.sign(last)))) / (2 * len(last)))
    return centroid, bw, flat, low - high, zcr


def extract_features(x, sr, pause_start, pause_index, last_pause_end=0.0,
                     contours=None, lang_id=0.0):
    """Causal Tier-1 + spectral/rate features for one pause.

    HARD CAUSALITY: slices `x_causal = x[:int(pause_start * sr)]` once and
    derives every feature from that prefix only. `pause_end` / pause duration
    are never used.

    Returns (feat_vector[N_FEATURES], rise_score, fall_score).
    """
    # --- hard causal cut: nothing after pause_start may be used ---
    end = max(0, int(float(pause_start) * sr))
    x_causal = x[:end]

    hop = HOP_S
    ipi = max(0.0, float(pause_start) - float(last_pause_end))

    out = np.zeros(N_FEATURES, dtype=np.float32)
    out[0] = float(pause_index)
    out[1] = float(pause_start)
    out[2] = float(ipi)
    out[3] = float(np.log1p(max(pause_start, 0.0)))
    out[4] = float(np.log1p(ipi))
    out[5] = 1.0 if pause_index == 0 else 0.0
    out[6] = 1.0 if pause_index >= 2 else 0.0
    out[7] = float(lang_id)

    if len(x_causal) < max(sr // 20, 64):
        return out, 0.0, 0.0

    e_turn, f0_turn, hop_e, hop_f = _causal_contours(x_causal, sr, contours)

    window_sec = 1.5
    n_loc_e = max(1, int(window_sec / hop_e))
    n_loc_f = max(1, int(window_sec / hop_f))
    n_short = max(3, int(0.15 / hop_e))
    e_local = e_turn[-n_loc_e:] if len(e_turn) else e_turn
    f0_local = f0_turn[-n_loc_f:] if len(f0_turn) else f0_turn

    e_slope, e_drop, e_final = energy_decay_into_pause(e_turn, hop_e, window_s=0.4)
    e_slope_short = _slope(e_turn[-n_short:]) if len(e_turn) >= 3 else 0.0
    e_mean_local = float(e_local.mean()) if len(e_local) else 0.0
    pre_rms = _rms_db(x_causal[-int(0.05 * sr):]) if len(x_causal) > int(0.05 * sr) else _rms_db(x_causal)
    r50 = float(np.sqrt(np.mean(x_causal[-int(0.05 * sr):] ** 2) + 1e-12)) if len(x_causal) > int(0.05 * sr) else float(np.sqrt(np.mean(x_causal ** 2) + 1e-12))
    r200 = float(np.sqrt(np.mean(x_causal[-int(0.20 * sr):] ** 2) + 1e-12)) if len(x_causal) > int(0.20 * sr) else r50
    rms_ratio = 20.0 * np.log10(r50 / (r200 + 1e-12))
    e_vs_med = e_final - float(np.median(e_turn)) if len(e_turn) else 0.0

    f0_slope_run, f0_final_run, final_dur = f0_slope_last_voiced(f0_turn, hop_f)
    v_local = f0_local[f0_local > 0]
    voiced_frac_local = float(np.mean(f0_local > 0)) if len(f0_local) else 0.0
    if voiced_frac_local < 0.3:
        f0_slope_term = 0.0
    else:
        f0_slope_term = _slope(v_local[-min(10, len(v_local)):]) if len(v_local) >= 3 else 0.0
    f0_std_local = float(v_local.std()) if len(v_local) > 1 else 0.0

    rel_med, rel_mean, f0_final, f0_mean, f0_med = relative_final_pitch(
        f0_turn, hop_f, local_window_s=window_sec
    )
    if f0_final <= 0 and f0_final_run > 0:
        f0_final = f0_final_run

    fall_flag = 1.0 if (rel_med < 0.95 and f0_slope_run < 0) else 0.0
    rise_flag = 1.0 if (rel_med > 1.05 and f0_slope_run > 0) else 0.0
    fall_score = (-f0_slope_run) + 50.0 * (1.0 - min(rel_med, 1.5))
    rise_score = f0_slope_run + 50.0 * max(rel_med - 1.0, 0.0)

    len_mean, len_med, final_dur2, mean_run = final_lengthening_ratio(f0_turn, hop_f)
    if final_dur2 > 0:
        final_dur = final_dur2

    runs = _voiced_runs(f0_turn, hop_f)
    turn_dur = max(len(x_causal) / float(sr), 1e-3)
    speaking_rate = float(len(runs)) / turn_dur
    voiced_frac_turn = float(np.mean(f0_turn > 0)) if len(f0_turn) else 0.0
    trail = 0
    for b in (f0_turn > 0)[::-1]:
        if not b:
            trail += 1
        else:
            break
    trailing_uv = trail * hop_f

    centroid, bw, flat, low_high, zcr = _spectral_tail(x_causal, sr, dur=0.12)

    feats = [
        float(pause_index),
        float(pause_start),
        float(ipi),
        float(np.log1p(max(pause_start, 0.0))),
        float(np.log1p(ipi)),
        1.0 if pause_index == 0 else 0.0,
        1.0 if pause_index >= 2 else 0.0,
        float(lang_id),
        e_mean_local,
        e_final,
        e_slope,
        e_drop,
        e_slope_short,
        pre_rms,
        float(rms_ratio),
        float(e_vs_med),
        f0_mean,
        f0_med,
        f0_final,
        f0_std_local,
        f0_slope_run,
        f0_slope_term,
        rel_med,
        rel_mean,
        fall_flag,
        rise_flag,
        float(fall_score),
        float(rise_score),
        final_dur,
        mean_run,
        len_mean,
        len_med,
        speaking_rate,
        voiced_frac_turn,
        voiced_frac_local,
        float(len(runs)),
        trailing_uv,
        float(centroid),
        float(bw),
        float(flat),
        float(low_high),
        float(zcr),
    ]
    arr = np.asarray(feats, dtype=np.float32)
    assert len(arr) == N_FEATURES, (len(arr), N_FEATURES)
    out[:N_FEATURES] = arr
    return out, float(rise_score), float(fall_score)


def detect_lang_id(data_dir, rows=None):
    """0=english-like, 1=hindi-like. Prefer folder name, else turn_id prefix."""
    name = os_basename_lower(data_dir)
    if "hindi" in name or name.startswith("hi"):
        return 1.0
    if "english" in name or name.startswith("en"):
        return 0.0
    if rows:
        tid = rows[0].get("turn_id", "")
        if str(tid).startswith("hi"):
            return 1.0
        if str(tid).startswith("en"):
            return 0.0
    return 0.0


def os_basename_lower(path):
    import os
    return os.path.basename(os.path.abspath(path)).lower()
