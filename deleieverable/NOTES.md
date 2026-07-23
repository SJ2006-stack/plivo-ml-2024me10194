# NOTES

We built a **causal** end-of-turn detector that never peeks past `pause_start`: Tier-1 prosody from `librosa.piptrack` F0 (fmin=50), energy decay, final lengthening, relative pitch fall/rise, spectral shape, and turn structure (`pause_index`, IPI), scored by a unified EN+HI LR+HGB model with listening-driven first-pause gates and a tiny Hindi-safe nudge.  
The shipped handout result — **English 1000 ms / Hindi 781 ms** @ ≤5% interruptions — **beats silence on both languages** (1600 / 850) and is the best combined artifact we earned after a full night of baselines, overfit diagnoses, pitch experiments, and rejected Hindi-only tweaks.  
We learned the hard way that in-sample glory (**298 / 398 ms**) was fake, that **AUC ≠ delay** on Hindi, that pause-duration sample weights were a causality leak, that `pyin` pitch *hurt* Hindi OOF back to 850, and that aggressive val freezes (thr=0.8) can timeout the test at **1600 ms**.  
Failures we owned and converted into wins: phrase-final≠turn-final false cutoffs (`en__082`), rise-penalized short EOTs (`hi__002`), mid-turn false confidence (`hi__097`), m2 abandon, and a HI-772 / EN-1030 trade we correctly **rejected**.  
With one more day we would push honest Hindi OOF clearly under 840 without regressing English, finish hang/trailing-energy cues, and advertise only GroupKFold / protocol numbers as the grade.
