"""Generate the paper's two main figures from existing results.

Figure 1: honest matched-rate RD comparison (frame, frame+RLE, billed adaptive,
billed uniform) over the measured rate range, with the matched window shaded.

Figure 2: timing-channel attribution (free / billed / none) at K=32, plus the
soft/hard x timing matrix from the ablation.

Outputs: figures/fig1_rd_crossover.png, figures/fig2_timing_ablation.png
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EX = Path(__file__).resolve().parent
FIG = EX.parent / "figures"
FIG.mkdir(exist_ok=True)


def summary_rows(path: Path):
    d = json.loads(path.read_text())
    return d["summary"], d["config"]


def main() -> None:
    ms_summary, ms_cfg = summary_rows(EX / "results_matched_rate_multiseed.json")
    rle_summary, _ = summary_rows(EX / "results_frame_rle.json")
    ms_points = json.loads((EX / "results_matched_rate_multiseed.json").read_text())["points"]

    # ---------------- Figure 1: RD crossover ----------------
    fig, ax = plt.subplots(figsize=(7.2, 5.0))

    def curve(arm, label, color, marker, agg=lambda r: (r["bits_per_sec"], r["mse_mean"]),
              err=None):
        rows = [r for r in ms_summary.values() if r["arm"] == arm] if arm in ("frame", "adaptive", "uniform") \
            else [r for r in rle_summary.values()]
        rows.sort(key=agg)
        xs = [agg(r)[0] for r in rows]
        ys = [agg(r)[1] for r in rows]
        if err:
            errs = [err(r) for r in rows]
            ax.errorbar(xs, ys, yerr=errs, label=label, color=color, marker=marker,
                        ms=4, lw=1.4, capsize=2, elinewidth=0.8)
        else:
            ax.plot(xs, ys, label=label, color=color, marker=marker, ms=4, lw=1.4)

    curve("frame", "frame-VQ (plain)", "#888888", "s",
          err=lambda r: r["mse_std"])
    curve("frame_rle", "frame-VQ + run-length (control)", "#2a7de1", "D",
          agg=lambda r: (r["bits_per_sec_mean"], r["mse_mean"]),
          err=lambda r: r["mse_std"])
    curve("adaptive", "adaptive segment-VQ (billed)", "#c1272d", "o",
          err=lambda r: r["mse_std"])
    curve("uniform", "uniform segment-VQ (billed)", "#f2a900", "^",
          err=lambda r: r["mse_std"])

    ax.axvspan(50, 132, color="#2a7de1", alpha=0.07)
    ax.axvspan(80, 100, color="#c1272d", alpha=0.10)
    ax.annotate("matched\nwindow", xy=(91, 0.425), ha="center", fontsize=8, color="#2a7de1")
    ax.annotate("crossover\nregion", xy=(90, 0.135), ha="center", fontsize=8, color="#c1272d")

    ax.set_xlabel("rate (bits/s, exact accounting)")
    ax.set_ylabel("reconstruction MSE (PCA-64 WavLM)")
    ax.set_title("Matched-rate RD comparison, billed timing, 3 seeds (mean $\\pm$ seed std)")
    ax.set_yscale("log")
    ax.set_xlim(0, 320)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_rd_crossover.png", dpi=200)
    plt.close(fig)

    # ---------------- Figure 2: timing attribution ----------------
    # Panel (a): adaptive K=32 decoder MSE under the three timing modes vs frame.
    modes = {"density (free timing)": {}, "billed": {}, "none": {}}
    frame = {}
    for p in ms_points:
        if p["seed"] != 7:
            continue
    # billed + none per (N) at K=32 come from the billed/none sweeps; free from lowrate
    billed_pts = json.loads((EX / "results_experiment_b_vq_lowrate_billed.json").read_text())["points"]
    none_pts = json.loads((EX / "results_experiment_b_vq_lowrate_nopd.json").read_text())["points"]
    free_pts = json.loads((EX / "results_experiment_b_vq_lowrate.json").read_text())["points"]
    frame_low = json.loads((EX / "results_experiment_b_vq_lowrate.json").read_text())["points"]

    def collect(pts, arm, store, key):
        for p in pts:
            if p["arm"] == arm:
                store.setdefault(p["codebook"], {})[key] = (p["bits_per_sec_exact"], p["mse"])

    collect(billed_pts, "adaptive", modes["billed"], "b")
    collect(billed_pts, "frame", frame, "b")
    collect(none_pts, "adaptive", modes["none"], "n")
    collect(free_pts, "adaptive", modes["density (free timing)"], "f")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axa = axes[0]
    colors = {"density (free timing)": "#2a9d3a", "billed": "#c1272d", "none": "#555555"}
    markers = {"density (free timing)": "o", "billed": "s", "none": "^"}
    for mode, store in modes.items():
        if not store:
            continue
        ns = sorted(store)
        axa.plot(ns, [store[n]["b" if "b" in store[n] else ("n" if "n" in store[n] else "f")][1]
                      for n in ns],
                 label=mode, color=colors[mode], marker=markers[mode], ms=5, lw=1.5)
    # frame arm: timing-independent; use the billed-sweep frame points (N=4..64)
    fb = {p["codebook"]: p["mse"] for p in billed_pts if p["arm"] == "frame"}
    axa.plot(sorted(fb), [fb[n] for n in sorted(fb)], label="frame-VQ", color="#888888",
             marker="D", ms=5, lw=1.5, ls="--")
    axa.set_xlabel("codebook size $N$ (K=32)")
    axa.set_ylabel("decoder MSE")
    axa.set_title("Timing-mode attribution (WavLM, seed 7)")
    axa.legend(fontsize=8)
    axa.grid(alpha=0.25)

    axb = axes[1]
    abl = json.loads((EX / "results_soft_hard_timing_ablation.json").read_text())["points"]
    order = ["soft_with_ud", "soft_billed_ud", "soft_no_ud"]
    labels = ["free\n(density u/d)", "billed\n(boundary u/d)", "none"]
    soft = [next(p["mse"] for p in abl if p["ablation"] == o) for o in order]
    hard = [next(p["hard_mse"] for p in abl if p["ablation"] == o) for o in order]
    x = range(3)
    axb.bar([i - 0.18 for i in x], soft, width=0.36, label="soft decoder MSE", color="#c1272d")
    axb.bar([i + 0.18 for i in x], hard, width=0.36, label="hard partition-mean MSE", color="#7f8c9a")
    axb.axhline(0.321, color="#444444", ls=":", lw=1)
    axb.text(2.45, 0.321, "hard family\n(exp B, 0.32)", fontsize=7, va="center")
    axb.set_xticks(list(x))
    axb.set_xticklabels(labels, fontsize=8)
    axb.set_ylabel("MSE")
    axb.set_title("Soft/hard $\\times$ timing matrix (K=32, N=256)")
    axb.legend(fontsize=8)
    axb.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_timing_ablation.png", dpi=200)
    plt.close(fig)

    print(f"saved {FIG/'fig1_rd_crossover.png'}")
    print(f"saved {FIG/'fig2_timing_ablation.png'}")


if __name__ == "__main__":
    main()
