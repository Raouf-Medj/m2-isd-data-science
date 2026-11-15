# Decision log

One entry per scientific choice, at least one per milestone. Keep entries short and factual: the oral
exam asks you to defend them, including the alternatives you rejected.

## D1 — Acoustic model: GMM-HMM with 5 states × 2 mixtures (S5M2)   (milestone: d1)
- Question: which (states × mixtures) configuration should be the acoustic model of the digit recogniser?
- Alternatives: 8 configurations, states ∈ {2, 3, 5, 8} × mixtures ∈ {1, 2} (S2M1 … S8M2).
- Protocol: FSDD, MFCC, one GMM-HMM per digit; standard split (300 test utterances) under clean, white noise 10 dB,
  babble 5 dB, reverberation (RT60 0.6 s) and telephone; plus a held-out speaker (theo, 500 utterances, clean).
  Metrics: accuracy, latency per utterance, number of parameters. Laptop CPU, 1 run per configuration.
- Evidence: `results/figures/isolated_digits.png`, `results/speech_classical/isolated_digits.csv`,
  rows `task=gmm_isolated, cfg_milestone=d1`.
  S5M2: 99.0 % clean, 79.0 % noise 10 dB, 83.3 % telephone, 13.7 ms, 8 633 params.
  S8M2: same 99.0 % clean, no better under noise (77.3 %), but 40 ms (×3) and 13 763 params (×1.6).
  S3M2: 8 ms but less robust (98.3 % clean, 71.7 % noise 10 dB, 73.3 % telephone).
- Decision: S5M2. It has the best accuracy, the best robustness to noise, and a latency well below real time
  (RTF ≈ 0.04). It is also the acoustic model used for connected digits.
- Would change our mind if: a latency budget below 10 ms per utterance (we would take S3M2), or held-out-speaker
  results becoming the priority: on the unseen speaker S5M2 drops to 97.8 % while S3M2 reaches 98.8 %, so larger
  models generalise slightly worse to a new voice.
