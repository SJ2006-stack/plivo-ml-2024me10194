# NOTES

Our model decides end-of-turn from **causal Tier-1 prosody only**: for each pause it uses audio from time 0 up to `pause_start` (never pause duration or post-pause audio) and extracts F0 via `librosa.piptrack` (`fmin=50`), energy decay, lengthening / rate, relative final pitch, spectral shape, and turn-structure features.  
Those features feed LR + shallow HGB trained as one **unified EN+HI** model on the full handout (`models/unified.joblib`), with first-pause rise/fall gates; `predict.py` loads the saved model with no refit.  
**Best shipped handout:** English **1000 ms** (AUC 0.838) and Hindi **783 ms** (AUC 0.864), both beating silence; a Hindi-only push to **772 ms** hurt English to 1030 so we kept the combined best.  
Honest holdout bars remain weaker: protocol freeze EN **1300** / HI **1600**; best trusted Hindi delay is still pre-unified OOF **~840 ms**.  
Typical errors: phrase-final first-pause holds scored as EOT, mid-turn holds outranking the true end, and short rising Hindi EOTs under-scored before the rise gate.  
With one more day we would improve Hindi delay without regressing English, and only claim OOF / 60/20/20 numbers as the grade.
