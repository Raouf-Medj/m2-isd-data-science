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
  (RTF ≈ 0.04). It is also the acoustic model used for connected digits (D3, D4).
- Would change our mind if: a latency budget below 10 ms per utterance (we would take S3M2), or held-out-speaker
  results becoming the priority: on the unseen speaker S5M2 drops to 97.8 % while S3M2 reaches 98.8 %, so larger
  models generalise slightly worse to a new voice.

## D2 — Language model: 3-gram, interpolated smoothing when training data is small   (milestone: d1)
- Question: which n-gram order N and which smoothing for the digit language model?
- Alternatives: N = 1 … 5; add-k vs interpolation; training sizes 50, 500 and 5 000 sentences.
- Protocol: digit grammar with 2nd-order structure, perplexity on 1 000 held-out sentences, model size
  (number of entries). 1 run per configuration.
- Evidence: `results/speech_classical/ngram_perplexity.csv`, rows `task=ngram_lm, cfg_milestone=d1`.
  With 5 000 sentences (add-k): perplexity 10.69 (1-gram), 8.89 (2), **5.05 (3)**, 5.16 (4), 5.67 (5);
  sizes 1 035 entries (3-gram) vs 4 687 (4-gram) and 12 693 (5-gram).
  With 50 sentences: interpolation 7.76 vs add-k 8.08 (3-gram). At high order interpolation degrades less
  (5-gram, 5 000 sentences: 5.23 vs 5.67).
- Decision: N = 3. The grammar is 2nd-order (each digit depends on the two previous ones), which is exactly
  what a trigram models; higher orders add 4–12× more entries without lowering perplexity (sparsity).
  Interpolated smoothing when the corpus is small, add-k is enough with 5 000 sentences.
- Would change our mind if: a language with longer dependencies (4-gram would then pay off with enough data),
  or a decoder where graph size matters (a trigram needs one copy of each word per previous word: see D3).

## D3 — Connected digits: bigram LM with weight w = 32, insertion penalty tuned on dev   (milestone: d1)
- Question: how much weight should the language model get against the acoustic evidence?
- Alternatives: acoustic only (uniform LM) vs bigram with w ∈ {0.5, 1, 2, 4, 8, 16, 32, 64}; for each,
  word insertion penalty chosen in {-600 … -30} on a dev subset.
- Protocol: connected-digit utterances built from FSDD test words, word order from a Markov chain;
  bigram trained on 5 000 text-only sequences of the same domain; 150 utterances (30 dev for the penalty,
  120 test); SNR clean, 20, 15, 10 dB; acoustic model S5M2 (D1). Metrics: WER, Viterbi ms per utterance.
- Evidence: `results/figures/connected_digits_lm.png`, `results/speech_classical/connected_digits_lm.csv`,
  rows `task=gmm_connected, cfg_milestone=d1, domain=matched`.
  WER acoustic only → best bigram: clean 4.4 % → 1.5 % (w=32); 20 dB 26.4 % → 9.4 % (w=32);
  15 dB 45.0 % → 18.8 % (w=64); 10 dB 57.8 % → 28.8 % (w=64).
  Too much weight hurts: clean WER 1.5 % at w=32 vs 3.0 % at w=64 (deletions rise from 2 to 8).
  Cost: Viterbi time ≈ 6 ms per utterance with or without the LM.
- Decision: bigram with w = 32 and a dev-tuned insertion penalty: the best compromise across all SNRs.
  The LM divides the WER by 2–3 and helps more as the acoustics degrade, at no decoding cost
  (a bigram needs no state expansion: the source word-end state already identifies the previous word).
- Would change our mind if: the deployment is always noisy (≤ 15 dB), where w = 64 is better,
  or the spoken domain does not match the LM (see D4).

## D4 — Use a strong LM weight only when the LM matches the spoken domain   (milestone: d1)
- Question: what happens when the LM is trained on a different domain from what is actually said?
- Alternatives: same bigram (trained on the structured domain) decoded on matched utterances vs
  mismatched utterances (uniformly random digit strings).
- Protocol: same as D3, `domain=mismatched`.
- Evidence: `results/figures/connected_digits_lm.png`, `results/speech_classical/connected_digits_lm.csv`,
  rows `task=gmm_connected, cfg_milestone=d1, domain=mismatched`.
  Clean: 1.7 % acoustic only, unchanged up to w = 4, then 3.0 % at w = 32 and 8.1 % at w = 64.
  10 dB: 52.5 % acoustic only vs 60.4 % at w = 64 (worse than no LM; deletions 15 → 166).
  The only gain is at 20 dB (22.8 % → 16.3 % at w = 16), mostly from fewer insertions (102 → 54).
- Decision: a strong LM weight (w = 32) only when the LM domain matches the use case;
  when the domain is uncertain, a small weight (w ≤ 4) costs nothing and keeps the LM safe.
- Would change our mind if: an LM trained on a mixture of domains closes the gap, or a way to
  detect the domain at run time lets us adapt the weight.
