import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

c = pd.read_csv("results/controls.csv"); pb = np.load("data/processed/per_beat_controls.npz"); cls = pb["cls"]
key = lambda m, t, s, sm, snr, met: f"{m}|{t}|{s}|{sm}|{snr}|{met}"
M = ["GradCAM", "IntGrad", "SHAP"]; col = {"GradCAM": "C3", "IntGrad": "C2", "SHAP": "C0"}

# --- Fig 1: resolution control (noise-augmented models, spearman)
fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
for a, met in zip(ax, ("spearman", "top20_jaccard")):
    for m in M:
        for sm, ls in ((False, "-"), (True, "--")):
            if m == "GradCAM" and sm: continue
            d = c[(c.method == m) & (c.training == "aug") & (c.series == "noisy_vs_clean") & (c.smoothed == sm) & (c.metric == met)].sort_values("snr_db")
            a.plot(d.snr_db, d["mean"], ls, color=col[m], marker="o", label=f"{m}{' (pooled to Grad-CAM resolution)' if sm else ''}")
            a.fill_between(d.snr_db, d.ci_lo, d.ci_hi, color=col[m], alpha=0.15)
    a.set_title(met); a.set_xlabel("SNR (dB)")
ax[0].set_ylabel("clean vs noisy explanation"); ax[0].legend(fontsize=7); ax[0].set_ylim(0, 1)
plt.tight_layout(); plt.savefig("results/fig_resolution_control.png", dpi=130)

# --- Fig 2: floors
fig, ax = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for a, m in zip(ax, M):
    for tr, ls in (("clean", ":"), ("aug", "-")):
        for ser, cc in (("noisy_vs_clean", "k"), ("noisy_vs_noisy", "C1")):
            d = c[(c.method == m) & (c.training == tr) & (c.series == ser) & (~c.smoothed) & (c.metric == "spearman")].sort_values("snr_db")
            a.plot(d.snr_db, d["mean"], ls, color=cc, marker="o", label=f"{tr}: {ser}")
        f = c[(c.method == m) & (c.training == tr) & (c.series == "seed_floor") & (~c.smoothed) & (c.metric == "spearman")]["mean"]
        a.axhline(f.iloc[0], color="gray", ls=ls, label=f"{tr}: seed floor")
    a.set_title(m); a.set_xlabel("SNR (dB)"); a.set_ylim(0, 1)
ax[0].set_ylabel("Spearman"); ax[0].legend(fontsize=6)
plt.tight_layout(); plt.savefig("results/fig_floors.png", dpi=130)

# --- Tables
def row(m, tr, ser, sm, snr, met="spearman"):
    r = c[(c.method == m) & (c.training == tr) & (c.series == ser) & (c.smoothed == sm) & (c.snr_db == snr) & (c.metric == met)].iloc[0]
    return f"{r['mean']:.2f} [{r.ci_lo:.2f}, {r.ci_hi:.2f}]"
out = ["### Spearman (clean vs noisy), noise-augmented models, 95% bootstrap CI over beats\n",
       "| Method | resolution | 24 dB | 12 dB | 0 dB |", "|---|---|---|---|---|"]
for m in M:
    for sm in (False, True):
        if m == "GradCAM" and sm: continue
        out.append(f"| {m} | {'pooled' if sm else 'native'} | " + " | ".join(row(m, "aug", "noisy_vs_clean", sm, s) for s in (24, 12, 0)) + " |")
out += ["\n### Floors (Spearman, native resolution, noise-augmented models)\n", "| Method | noisy vs clean @12 dB | noisy vs noisy @12 dB | seed floor (clean beats) |", "|---|---|---|---|"]
for m in M:
    sf = c[(c.method == m) & (c.training == "aug") & (c.series == "seed_floor") & (~c.smoothed) & (c.metric == "spearman")].iloc[0]
    out.append(f"| {m} | {row(m,'aug','noisy_vs_clean',False,12)} | {row(m,'aug','noisy_vs_noisy',False,12)} | {sf['mean']:.2f} [{sf.ci_lo:.2f}, {sf.ci_hi:.2f}] |")
out += ["\n### Paired Wilcoxon on per-beat Spearman @12 dB, noise-augmented, pooled resolution\n", "| Comparison | median diff | p |", "|---|---|---|"]
for a_, b_ in (("IntGrad", "GradCAM"), ("SHAP", "GradCAM"), ("SHAP", "IntGrad")):
    sa = lambda m: pb[key(m, "aug", "noisy_vs_clean", m != "GradCAM", 12, "spearman")]
    va, vb = sa(a_), sa(b_); k = ~(np.isnan(va) | np.isnan(vb))
    out.append(f"| {a_} - {b_} | {np.median(va[k] - vb[k]):+.2f} | {wilcoxon(va[k], vb[k]).pvalue:.1e} |")
out += ["\n### Per-class Spearman @12 dB (noise-augmented, native)\n", "| Method | N | S | V | F |", "|---|---|---|---|---|"]
for m in M:
    v = pb[key(m, "aug", "noisy_vs_clean", False, 12, "spearman")]
    out.append(f"| {m} | " + " | ".join(f"{np.nanmean(v[cls == k]):.2f}" for k in range(4)) + " |")
nan = {m: np.isnan(pb[key(m, "aug", "noisy_vs_clean", False, 12, "spearman")]).mean() for m in M}
out.append(f"\nFraction of beats with undefined Spearman (constant attribution): {nan}")
open("results/controls_tables.md", "w").write("\n".join(out)); print("\n".join(out))
