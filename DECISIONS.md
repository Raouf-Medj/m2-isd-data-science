# Decision log

One entry per scientific choice, at least one per milestone. Keep entries short and factual: the oral
exam asks you to defend them, including the alternatives you rejected.

## D1 — Acoustic model: S5M2 (5 states × 2 mixtures)   (milestone: d1)
- Question: which GMM-HMM configuration should we use as the acoustic model?
- Alternatives: 8 configurations: states ∈ {2, 3, 5, 8} × mixtures ∈ {1, 2}.
- Protocol: FSDD digits, 300 test files, tested clean and under 4 degradations (white noise, babble, reverb, telephone), plus one unseen speaker. We measured accuracy, latency and model size. One run per configuration.
- Evidence (`results/figures/isolated_digits.png`): S5M2 and S8M2 both reach 99.0 % clean. S5M2 is better on white noise (79.0 vs 77.3 %), S8M2 slightly better on babble and telephone (1–3 pts, within the error margin). S8M2 is 1.6× bigger and was about 3× slower. S3M2 is faster (8 ms) but clearly less robust (71.7 % on white noise).
- Decision: S5M2: same accuracy as S8M2, smaller and faster. We also use it for connected digits.
- Would change our mind if: we needed latency below 10 ms (we would take S3M2), or new runs showed S8M2 is not slower (latency varies between runs).
