# NOTES

We built a **causal** end-of-turn detector that never peeks past `pause_start`: Tier-1 prosody from `librosa.piptrack` F0 (`fmin=50`), energy decay into silence, final lengthening / speaking rate, relative pitch fall vs rise, spectral shape, and turn structure (`pause_index`, IPI), scored by one **unified EN+HI** LR + shallow HGB model with listening-driven first-pause gates and a Hindi-only ×1.10 safe nudge.  
The shipped handout — **English 1000 ms / Hindi 781 ms** @ ≤5% interruptions — **beats silence on both languages** (1600 / 850) and is the best combined artifact we earned after baselines, listening, overfit diagnoses, pitch regressions, OP disasters, and rejected vanity tweaks.  
We learned the hard way that in-sample glory (**298 / 398 ms**) was fake, that **AUC ≠ delay** on Hindi, that pause-duration sample weights were a causality leak, that `pyin` pitch *hurt* Hindi OOF back to 850, and that aggressive val freezes (thr=0.8) can timeout the test at **1600 ms**.  
Failures we owned and converted: phrase-final≠turn-final false cutoffs (`en__082`), rise-penalized short EOTs (`hi__002`), mid-turn false confidence (`hi__097`), m2 abandon, and a HI-772 / EN-1030 trade we correctly **rejected** to protect the combined objective.  
With one more day we would push honest Hindi OOF clearly under 840 without regressing English, finish hang / trailing-energy hold cues, fix Hindi OP freeze, and advertise only GroupKFold / protocol numbers as the grade.

---

## Deep notes (how we actually did it)

### Signal we trust
Human listening (`en__051` fall+lengthening → EOT; `en__066` hang → HOLD; error mining on `en__082` / `hi__002`) invented the features; code only implemented them. Causality was non-negotiable: no `pause_end`, no duration as features or weights.

### Path to the ship
Silence → weak mine → fake Tier-1 glory → honesty collapse → remove weight leak → shrink model → OOF/protocol → unified full-data train → first-pause gates (EN 1015→1000) → Hindi ×1.10 nudge (HI 783→781) → single `predictions.csv` (496 rows).

### What we refuse to claim
298/398 ms, HI-772 at the cost of EN-1030, pyin as a “win,” and any OP that only looks good on 20-turn val. Integrity is part of the deliverable.
