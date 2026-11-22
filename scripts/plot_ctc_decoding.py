"""Day 2 afternoon figure: CTC decoding, WER vs decoding time, from the `ctc_decoding` rows of results.jsonl.

One panel per acoustic model. Dots = the dev sweep of `python -m speech.ctc_lm` (greedy, beam without LM,
beam + n-gram for every alpha x beta x beam); dashed line = the dev Pareto front (common/pareto.py), whose
points are labelled. Crosses = the best dev point decoded once on the test half (the numbers we report).

    python -m scripts.plot_ctc_decoding            # -> results/figures/ctc_decoding_pareto.png
"""
from __future__ import annotations

import argparse
from pathlib import Path


def short(r) -> str:
    """Label of one configuration: its decoder knobs (alpha = LM weight, beta = word bonus, N = beam width)."""
    if r.decoder.startswith("greedy"):
        return "greedy"
    if r.decoder == "beam, no LM":
        return f"no LM, N={int(r.cfg_beam)}"
    return f"α={r.cfg_alpha:g} β={r.cfg_beta:g} N={int(r.cfg_beam)}"


def plot(path: Path, out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from common.bench import load_records
    from common.pareto import plot_pareto

    df = load_records(path)
    df = df[df.task == "ctc_decoding"].copy()
    if df.empty:
        raise SystemExit(f"no 'ctc_decoding' rows in {path}: run `python -m speech.ctc_lm` first")
    df["acoustic"] = df.model.str.split("|").str[0].str.split("/").str[-1]  # e.g. wav2vec2-base-960h
    df["decoder"] = df.model.str.split("|").str[-1]
    df["wer_pct"] = 100 * df.wer
    # a configuration measured twice: keep the last run
    df = df.drop_duplicates(["acoustic", "decoder", "cfg_alpha", "cfg_beta", "cfg_beam", "cfg_split"], keep="last")

    models = sorted(df.acoustic.unique())
    fig, axes = plt.subplots(1, len(models), figsize=(7 * len(models), 5.5), squeeze=False)
    for ax, m in zip(axes[0], models):
        dev = df[(df.acoustic == m) & (df.cfg_split == "dev")].reset_index(drop=True)
        test = df[(df.acoustic == m) & (df.cfg_split == "test")]
        _, front = plot_pareto(dev, "ms_per_utt", "wer_pct", "min", "min", hue="decoder", ax=ax, logx=True,
                               annotate=False, title=f"{m}: dev sweep ({int(dev.cfg_n.iloc[0])} utterances)")
        for _, r in dev[front].iterrows():  # name only the configurations on the front
            ax.annotate(short(r), (r.ms_per_utt, r.wer_pct), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.scatter(test.ms_per_utt, test.wer_pct, marker="x", c="k", s=60, label="test, other half (reported)")
        lines = [f"{short(r)}: {r.wer_pct:.2f} %, {r.ms_per_utt:.1f} ms" for _, r in test.iterrows()]
        ax.text(0.02, 0.03, f"test ({int(test.cfg_n.iloc[0])} other utterances)\n" + "\n".join(lines),
                transform=ax.transAxes, fontsize=8, va="bottom", bbox=dict(fc="white", ec="0.7"))
        ax.set(xlabel="decoding time per utterance (ms, log scale; network forward not included)",
               ylabel="WER (%)")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    print(f"figure: {out_png}")


def main():
    ap = argparse.ArgumentParser(description="CTC decoding: WER vs decoding time (rows task=ctc_decoding).")
    ap.add_argument("--results", type=Path, default=Path("results/results.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("results/figures/ctc_decoding_pareto.png"))
    args = ap.parse_args()
    plot(args.results, args.out)


if __name__ == "__main__":
    main()
