import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

s = pd.read_csv("results/explanation_stability.csv"); c = pd.read_csv("results/classification_vs_noise.csv")
c = c[c.snr_db != "clean"].assign(snr_db=lambda d: d.snr_db.astype(int))
noises = ["gaussian", "baseline_wander", "muscle"]
styles = {("cnn", "clean"): ("C3", "-"), ("cnn", "aug"): ("C3", "--"), ("rf", "clean"): ("C0", "-"), ("rf", "aug"): ("C0", "--")}
fig, ax = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
for j, n in enumerate(noises):
    for (m, t), (col, ls) in styles.items():
        lab = f"{m.upper()} {'noise-aug.' if t == 'aug' else 'clean-trained'}"
        a = c[(c.model == m) & (c.training == t) & (c.noise == n)].groupby("snr_db").accuracy.agg(["mean", "std"])
        ax[0, j].errorbar(a.index, a["mean"], a["std"], color=col, ls=ls, marker="o", label=lab, capsize=2)
        xai = "SHAP" if m == "rf" else "Grad-CAM"
        for k, metric in ((1, "spearman"), (2, "top20_jaccard")):
            b = s[(s.model == m) & (s.xai == xai) & (s.training == t) & (s.noise == n) & (s.metric == metric)]
            b = b.groupby("snr_db").mean_all.agg(["mean", "std"]).fillna(0)
            ax[k, j].errorbar(b.index, b["mean"], b["std"], color=col, ls=ls, marker="o", capsize=2)
    ax[0, j].set_title(n); ax[0, j].set_ylim(0, 1); ax[1, j].set_ylim(-0.1, 1); ax[2, j].set_ylim(0, 1)
    ax[2, j].axhline(0.111, color="gray", ls=":", label="chance"); ax[2, j].set_xlabel("SNR (dB)")
ax[0, 0].set_ylabel("DS2 accuracy"); ax[1, 0].set_ylabel("Spearman (clean vs noisy expl.)"); ax[2, 0].set_ylabel("Top-20% Jaccard")
ax[0, 0].legend(fontsize=7); ax[2, 0].legend(fontsize=7)
plt.suptitle("Error bars: std over 3 training seeds (CNN); RF explanations use seed 0 only")
plt.tight_layout(); plt.savefig("results/stability_vs_noise.png", dpi=130)
