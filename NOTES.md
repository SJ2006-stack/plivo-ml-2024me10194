# NOTES

Our model decides end-of-turn from **causal Tier-1 prosody only**: for each pause it uses audio from time 0 up to `pause_start` (never pause duration or post-pause audio) and extracts F0 via `librosa.piptrack` (`fmin=50`), energy decay, lengthening / rate, relative final pitch, spectral shape, and turn-structure features.  
Those features feed LR + shallow HGB trained as one **unified EN+HI** model on the full handout (`models/unified.joblib`), with first-pause rise/fall gates and stronger late-pause fall boost; `predict.py` loads the saved model with no refit.  
**Best handout records:** English **1000 ms** (AUC 0.838); Hindi **772 ms** (AUC 0.861) — current ship is the Hindi-priority config (EN 1030 / HI **772**, beating silence 850).  
Honest holdout bars remain weaker: protocol freeze EN **1300** / HI **1600**; best trusted Hindi delay is still pre-unified OOF **~840 ms**.  
Typical errors: phrase-final first-pause holds scored as EOT, mid-turn holds outranking the true end, and short rising Hindi EOTs under-scored before the rise gate.  
With one more day we would restore the best-EN gate-only checkpoint for English while keeping Hindi 772, and only claim OOF / 60/20/20 numbers as the grade.
