# NOTES

Our model decides end-of-turn from **causal Tier-1 prosody only**: for each pause it uses audio from time 0 up to `pause_start` (never pause duration or post-pause audio) and extracts F0 via `librosa.piptrack` (`fmin=50`), energy decay, final lengthening / speaking-rate, relative final pitch (fall vs rise), spectral shape, and turn-structure features (`pause_index`, IPI).  
Those features feed a shrunk LR + shallow HistGradientBoosting ensemble trained as one **unified EN+HI** model on the full handout (`models/unified.joblib`), with light first-pause gates (soften rise; cut fall boost on `pause_index==0`), scored only through `predict.py`.  
All-time best handout ship is **EN 1000 ms** (AUC 0.838) and **HI 783 ms** (AUC 0.864), pause-acc@0.5 ≈ **0.76** — Hindi beats silence **850 ms** on the labeled set, but those scores are in-sample.  
Honest bars are weaker: protocol @ val-frozen EN **1300 ms** / HI **1600 ms**, and the best trusted Hindi delay remains pre-unified OOF **~840 ms**.  
It still fails on phrase-final first-pause holds, mid-turn completion-like holds outranking true EOTs, short rising Hindi EOTs, and over-conservative Hindi OP freezes.  
With one more day we would recalibrate Hindi thresholds on val, finish listening-driven gates, and only advertise OOF / 60/20/20 numbers — never train-folder delay as the grade.
