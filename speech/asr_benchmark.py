"""Offline ASR Pareto front on LibriSpeech: WER vs real-time factor, latency, energy, size.

    python -m speech.asr_benchmark --librispeech /data/LibriSpeech --subset test-clean --n 200 \
        --models openai/whisper-tiny openai/whisper-small facebook/wav2vec2-base-960h --batch 1 16
    python -m speech.asr_benchmark --plot results/asr_bench.jsonl

batch=1 measures LATENCY (interactive use); batch=16 measures THROUGHPUT (offline transcription).
The two give different fronts -- one of the key messages of the project.
UNTESTED in the authoring sandbox (no model downloads possible there). Verify model ids on the Hub;
some families (e.g. Moonshine, Parakeet) may need a recent transformers or their own library.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common.bench import (BenchRecord, cuda_sync, energy_scope, log_record, peak_memory_mb, reset_peak_memory,
                          summarize_latencies)
from speech.data import load_librispeech, wer

DEFAULT_MODELS = [
    "openai/whisper-tiny.en", "openai/whisper-base.en", "openai/whisper-small.en", "openai/whisper-medium.en",
    "openai/whisper-large-v3-turbo", "distil-whisper/distil-large-v3",
    "facebook/wav2vec2-base-960h", "facebook/wav2vec2-large-960h-lv60-self",
    "UsefulSensors/moonshine-tiny", "UsefulSensors/moonshine-base",
]


def bench_model(model_id, data, batch, device, fp16):
    """Benchmark one Hugging Face ASR model at one batch size.

    data: list of dicts with "audio" (float32 array), "sr" and "text" (from speech.data.load_librispeech).
    Returns (metrics, hypotheses): metrics has wer, rtf, n_params_M, energy_j_per_audio_min (None when not
    measurable), the keys of summarize_latencies (lat_p50_ms, lat_p90_ms, ...) and of peak_memory_mb.
    - fp16 only on a GPU (float32 on CPU).
    - The pipeline takes a list of {"raw": audio, "sampling_rate": sr} dicts and CONSUMES them (it pops the keys):
      pass fresh copies at every call, e.g. [dict(x) for x in inputs[i:i + batch]], or the second call gets {}.
    - Batching pads sequences: a tokenizer without a pad token (Moonshine) refuses to batch; set
      asr.tokenizer.pad_token_id = asr.model.config.eos_token_id when it is None.
    - At batch 1 there is one latency per utterance; at batch > 1, one per batch (what a request in that batch
      waits). Compare throughput with rtf.
    """
    import torch
    from transformers import pipeline
    dtype = torch.float16 if (fp16 and device.startswith("cuda")) else torch.float32
    asr = pipeline("automatic-speech-recognition", model=model_id, device=device, dtype=dtype)
    if batch > 1 and getattr(asr, "tokenizer", None) is not None and asr.tokenizer.pad_token_id is None:
        asr.tokenizer.pad_token_id = asr.model.config.eos_token_id  # Moonshine: no pad token, needed to batch
    n_params = sum(p.numel() for p in asr.model.parameters())
    inputs = [{"raw": d["audio"], "sampling_rate": d["sr"]} for d in data]
    asr([dict(x) for x in inputs[:2]], batch_size=min(batch, 2))  # warm-up (the pipeline empties its inputs)
    reset_peak_memory()
    lats, hyps, res = [], [], {}
    with energy_scope(res):
        t_all = time.perf_counter()
        for i in range(0, len(inputs), batch):
            batch_inputs = [dict(x) for x in inputs[i:i + batch]]
            cuda_sync()
            t0 = time.perf_counter()
            outputs = asr(batch_inputs, batch_size=batch)
            cuda_sync()
            elapsed = time.perf_counter() - t0
            lats.append(elapsed)
            hyps.extend(output["text"].strip() for output in outputs)
        total = time.perf_counter() - t_all
    audio_s = sum(len(d["audio"]) / d["sr"] for d in data)  # seconds of audio transcribed
    wer_value = wer([d["text"] for d in data], hyps)
    rtf = total / audio_s if audio_s > 0 else float("inf")
    m = dict(wer=wer_value, rtf=rtf, n_params_M=n_params / 1e6,
            energy_j_per_audio_min=(res["energy_j"] / audio_s * 60)
            if res.get("energy_j") is not None and audio_s > 0 else None)
    m.update(summarize_latencies(lats))  # lat_p50_ms, lat_p90_ms, ... from the list of seconds
    m.update(peak_memory_mb())
    return m, hyps


def plot(path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.bench import load_records
    from common.pareto import plot_pareto
    df = load_records(path)
    df["label"] = df.model.str.split("/").str[-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, b in zip(axes, sorted(df.cfg_batch.unique())[:2]):
        plot_pareto(df[df.cfg_batch == b].reset_index(drop=True), "rtf", "wer", "min", "min", label="label",
                    ax=ax, logx=True, title=f"WER vs real-time factor, batch={b}")
    fig.tight_layout()
    fig.savefig(Path(path).with_suffix(".png"), dpi=130)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--librispeech", type=Path)
    ap.add_argument("--subset", default="test-clean")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 16])
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("results/asr_bench.jsonl"))
    ap.add_argument("--plot", type=Path)
    args = ap.parse_args()
    if args.plot:
        return plot(args.plot)
    data = load_librispeech(args.librispeech, args.subset, args.n)
    print(f"{len(data)} utterances, {sum(len(d['audio']) / d['sr'] for d in data) / 60:.1f} min of audio")
    for mid in args.models:
        for b in args.batch:
            try:
                m, _ = bench_model(mid, data, b, args.device, args.fp16)
            except Exception as e:
                print(f"[skip] {mid} batch={b}: {e}")
                continue
            log_record(BenchRecord("asr", mid, dict(subset=args.subset, batch=b, fp16=args.fp16, n=len(data)), m), args.out)
            print(mid, b, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)})


if __name__ == "__main__":
    main()
