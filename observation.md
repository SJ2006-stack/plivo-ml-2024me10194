# EOT listening observations

Human listening log over weak starter (`starter/mine.csv`) error / hard cases. Purpose: extract **causal** feature ideas — only signals available from audio **up to `pause_start`**. Pause durations below are for human understanding of silence-baseline failures, not candidate features at decision time.

---

## en__051

| pause_index | pause_start | pause_end | dur (s) | label | p_eot |
|---:|---:|---:|---:|---|---:|
| 0 | 8.4 | 8.8 | 0.40 | hold | 0.270 |
| 1 | 9.5 | 10.1 | 0.60 | hold | 0.328 |
| 2 | 10.9 | 12.257 | 1.36 | eot | 0.697 |

**Observation (human):** Speaker takes a long pause mid-turn and again at the end; delivery is slow; voice falls toward the final pause.

**Useful signals (causal):**
- Final F0 / pitch fall into the last pause → **eot** cue
- Slow speaking rate before pause → possible **eot** / turn-completion cue (vs rushed mid-turn holds)
- Long pause duration explains why a silence baseline may fire late — **not usable as a hold/eot feature at decision time** (duration only known after silence ends)

---

## en__066

| pause_index | pause_start | pause_end | dur (s) | label | p_eot |
|---:|---:|---:|---:|---|---:|
| 0 | 2.8 | 3.0 | 0.20 | hold | 0.349 |
| 1 | 3.8 | 5.103 | 1.30 | eot | 0.580 |

**Observation (human):** Early segment is full silence; then speech starts incomplete / “hangs” (trailing off or not finishing the thought) rather than a clean close.

**Useful signals (causal):**
- Incomplete start / mid-turn voice hang (trailing energy, unfinished phrasing) before pause → **hold** cue
- Short early pause with incomplete speech context → silence alone is a weak discriminator
- Again: pause **duration** is diagnostic for humans (why silence models confuse hold vs eot) but **not** a causal feature for holds at `pause_start`

---

## Patterns to turn into features

1. **Prosodic completion:** final F0 fall / energy decay before pause → lean **eot**
2. **Speaking rate:** slower, deliberate speech into pause → lean **eot**; contrast with abrupt mid-turn cuts
3. **Incompleteness / hang:** trailing or unfinished onset into silence → lean **hold**
4. **Do not use:** pause duration (or anything after `pause_start`) as a causal hold feature — only as post-hoc explanation of silence-baseline errors

*Source: human listening notes on starter weak predictions; for grading / feature narrative.*

---

## Honest-error listening (holdout / OOF, 2026-07-23)

Analyzed worst **false cutoffs** (hold + high `p_eot`) and **late EOTs** (eot + low `p_eot`) from `pred_*_holdout.csv` / `oof_*.csv`, with feature CSV + causal waveform windows (≤ `pause_start`). Pause durations below are post-hoc only.

### False cutoffs (model too eager)

| turn | pi | p_eot | pause_start | post-hoc dur | what we infer |
|------|---:|------:|------------:|-------------:|---------------|
| `en__082` | 0 | ~0.78 | 3.3 s | 0.6 s | **First pause is hold** but strong fall + lengthening + energy cliff in last 150 ms — phrase-final, not turn-final. Continues to EOT at 5.3 s. |
| `en__030` | 0 | ~0.74 | 3.3 s | 0.5 s | Same pattern: first of five holds; huge energy drop into silence looks “done.” |
| `en__002` | 4 | ~0.80 | 16.1 s | 1.8 s | Late mid-turn hold in a long 7-pause turn; completion-like taper; model treats as EOT while two more holds + final remain. |
| `hi__007` | 2 | ~0.79 | 6.2 s | 0.6 s | Mid-turn hold; energy into pause is flatter (hang-like) yet `p_eot` still high — fall features under-informed; need incompleteness cue. |
| `hi__073` | 3 | ~0.76 | 15.9 s | 0.5 s | Late hold with fall_flag + energy drop; turn continues ~7 s. |
| `hi__097` | 1 | ~0.75 (OOF) | 9.8 s | 0.4 s | Known ranking failure family: hold outranks true EOT; sharp energy cliff before short hold. |

### Late EOTs (model too timid)

| turn | pi | p_eot | pause_start | post-hoc dur | what we infer |
|------|---:|------:|------------:|-------------:|---------------|
| `hi__002` | 0 | ~0.12 | 3.3 s | 0.44 s | **Whole turn is one short pause**; energy does decay, but `rise_flag=1` / high rise_score → model reads “continuation.” |
| `hi__048` | 0 | ~0.19 | 2.0 s | 0.96 s | Same: single-pause short HI utterance with rising final; would sit near timeout. |
| `en__013` | 0 | ~0.21 | 2.3 s | 1.42 s | Short EN one-shot EOT; strong rise scores dominate despite clear energy drop. |
| `en__095` | 5 | ~0.20 | 29.5 s | 1.30 s | True final after long turn; high spectral centroid / rise-ish terminal — noisy or non-prototypical close; model under-confident. |
| `hi__033` | 2 | ~0.16 (OOF) | 28.3 s | 1.13 s | Long gap after last hold; energy falls hard at end but `p_eot` stays low — calibration / rise–fall conflict on HI finals. |

### Lessons → causal feature/model tweaks

1. **Phrase-final ≠ turn-final:** gate fall/lengthening by `is_first_pause` / early `pause_start`.
2. **Rise penalty hurts short true EOTs** (esp. Hindi): condition or shrink `rise_beta` when the turn has no prior pauses.
3. **Need explicit hold/hang features** beyond “missing fall” (trailing unvoiced, flat local energy, high speaking rate into pause).
4. **Late mid-holds** need stronger “not last” structure features (past pause count, time-so-far) without using future pause duration.
5. **Calibrate Hindi** so aggressive low thresholds from val don’t blow the 5% cutoff budget on unseen turns.

*Source: holdout/OOF error mining + feature_data + waveform RMS windows; no post-pause audio used.*
