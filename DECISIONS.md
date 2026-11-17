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

## D2 — Neural model for push-to-talk: wav2vec2-base-960h at batch 1   (milestone: d2)
- Question: which pretrained recogniser of the model zoo should we use for an interactive (one utterance at a time) assistant on a laptop CPU?
- Alternatives: 3 families of small models: Whisper (`openai/whisper-tiny.en`, 37.8 M params, encoder-decoder), wav2vec2 CTC (`facebook/wav2vec2-base-960h`, 94.4 M), Moonshine (`UsefulSensors/moonshine-tiny`, 27.1 M).
- Protocol: the same 200 LibriSpeech test-clean utterances (22.5 min of audio), CPU only (Windows 11, 12 logical CPUs), fp32, one run per configuration, `python -m speech.asr_benchmark`. Metrics: WER, RTF, latency per utterance (p50/p90), parameters.
- Evidence (rows `task=asr, cfg_batch=1, cfg_milestone=d2`; `results/figures/asr_pareto.png`): wav2vec2-base-960h has the lowest WER (3.32 %) and is also the fastest (RTF 0.062, p90 814 ms); Moonshine-tiny 3.61 % (RTF 0.084, p90 1141 ms); Whisper-tiny.en 4.97 % (RTF 0.173, p90 1854 ms). wav2vec2 alone is on the (WER, lat_p90_ms) and (WER, RTF) fronts; Moonshine-tiny joins the front only on (WER, parameters), being 3.5× smaller.
- Decision: wav2vec2-base-960h at batch 1. It is also a CTC model, so it is the one we extend with an external n-gram LM in the afternoon.
- Would change our mind if: memory or download size became the constraint (Moonshine-tiny: +0.3 pt WER for 3.5× fewer parameters), or the larger models we have not measured yet (Whisper base/small, wav2vec2-large) gave a much lower WER within the latency budget. Caveat: `cpu_maxrss_mb` is the peak of the whole process, which benchmarks all models in turn, so it cannot be used to compare their memory.

## D3 — Batch size on CPU: batch 1, not 16   (milestone: d2)
- Question: does batching (16 utterances per call) pay off on a laptop CPU, as it does on a GPU?
- Alternatives: batch 1 (latency, interactive use) vs batch 16 (throughput, offline transcription), same 3 models and data as D2.
- Protocol: as D2; RTF = total processing time / audio duration.
- Evidence (rows `task=asr, cfg_milestone=d2`): batch 16 is slower in throughput for all three models: RTF 0.173 → 0.192 (Whisper-tiny.en), 0.062 → 0.160 (wav2vec2), 0.084 → 0.192 (Moonshine-tiny); a request then waits a whole batch (p90 22.8–29.9 s). Batching also padded the utterances to the longest one: wav2vec2-base-960h's WER rose from 3.32 % to 24.47 %, while Whisper-tiny.en (4.97 %) and Moonshine-tiny (3.61 → 3.64 %) were practically unchanged. Our hypothesis, not yet verified: this wav2vec2 checkpoint is used without an attention mask, so the zero padding is seen as audio.
- Decision: batch 1 everywhere on CPU. On a CPU one utterance already keeps the cores busy, and batching only adds padding.
- Would change our mind if: a GPU run showed batch 16 lowering the RTF (the expected result on a GPU), or batching utterances of similar length (less padding) closed the gap.
