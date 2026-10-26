import pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

s = pd.read_csv("results/explanation_stability.csv"); c = pd.read_csv("results/classification_vs_noise.csv")
c = c[c.snr_db != "clean"].assign(snr_db=lambda d: d.snr_db.astype(int))
noises = ["gaussian", "baseline_wander", "muscle"]
fig, ax = plt.subplots(3, 3, figsize=(14, 10), sharex=True)
for j, n in enumerate(noises):
    for m, xai in (("rf", "RF + SHAP"), ("cnn", "CNN + Grad-CAM")):
        a = c[(c.model == m) & (c.noise == n)].sort_values("snr_db")
        ax[0, j].plot(a.snr_db, a.accuracy, "o-", label=f"{xai} acc"); ax[0, j].plot(a.snr_db, a.pred_consistency, "x--", label=f"{xai} pred. consistency")
        for k, metric in ((1, "spearman"), (2, "top20_jaccard")):
            b = s[(s.model == m) & (s.noise == n) & (s.metric == metric)].sort_values("snr_db")
            ax[k, j].plot(b.snr_db, b.mean_all, "o-", label=xai)
    ax[0, j].set_title(n); ax[0, j].set_ylim(0, 1)
    ax[2, j].axhline(0.111, color="gray", ls=":", label="chance"); ax[2, j].set_xlabel("SNR (dB)")
    ax[1, j].set_ylim(-0.1, 1); ax[2, j].set_ylim(0, 1)
ax[0, 0].set_ylabel("Classification"); ax[1, 0].set_ylabel("Spearman (clean vs noisy expl.)"); ax[2, 0].set_ylabel("Top-20% Jaccard")
ax[0, 0].legend(fontsize=7); ax[2, 0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("results/stability_vs_noise.png", dpi=130)
