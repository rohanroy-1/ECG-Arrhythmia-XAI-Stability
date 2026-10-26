"""Explanation stability: compare the attribution of each clean beat with the attribution of the SAME beat after noise.
Both attributions explain the class predicted on the clean beat, so differences are due to the noise, not a changed target."""
import sys, pickle, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch
from ecg_xai.data import zscore
from ecg_xai.noise import add_noise, NOISE_TYPES, SNR_LEVELS
from ecg_xai.models import BeatCNN, predict_cnn
from ecg_xai.xai import shap_attr, gradcam
from ecg_xai import stability as st

N_PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 100
d = np.load("data/processed/beats.npz"); X, y = d["X_test"], d["y_test"]
rng = np.random.default_rng(0)
idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:N_PER_CLASS] for c in range(4)])
X, y = X[idx], y[idx]

rf = pickle.load(open("data/processed/rf.pkl", "rb"))
cnn = BeatCNN(); cnn.load_state_dict(torch.load("data/processed/cnn.pt")); cnn.eval()

Zc = zscore(X)
tgt = {"rf": rf.predict(Zc), "cnn": predict_cnn(cnn, Zc)}
expl = {"rf": lambda Z, t: shap_attr(rf, Z, t), "cnn": lambda Z, t: gradcam(cnn, Z, t)}
pred = {"rf": rf.predict, "cnn": lambda Z: predict_cnn(cnn, Z)}
clean = {m: expl[m](Zc, tgt[m]) for m in expl}

rows = []
for kind in NOISE_TYPES:
    for snr in SNR_LEVELS[1:]:
        t0 = time.time()
        Z = zscore(add_noise(X, kind, snr, seed=2))
        for m in expl:
            A = expl[m](Z, tgt[m]); same = pred[m](Z) == tgt[m]
            metrics = {"spearman": st.spearman(clean[m], A), "top20_jaccard": st.topk_jaccard(clean[m], A),
                       "top20_dice": st.topk_dice(clean[m], A), "cosine": st.cosine(clean[m], A)}
            for name, v in metrics.items():
                rows.append(dict(model=m, xai="SHAP" if m == "rf" else "Grad-CAM", noise=kind, snr_db=snr, metric=name,
                                 mean_all=np.nanmean(v), mean_pred_unchanged=np.nanmean(v[same]) if same.any() else np.nan,
                                 frac_pred_unchanged=same.mean()))
        print(kind, snr, f"{time.time() - t0:.0f}s", flush=True)
df = pd.DataFrame(rows); df.to_csv("results/explanation_stability.csv", index=False)
print("chance-level top-20% Jaccard:", round(st.random_topk_jaccard(), 3))
