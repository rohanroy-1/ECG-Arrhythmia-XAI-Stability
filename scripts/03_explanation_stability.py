"""Explanation stability: compare the attribution of each clean beat with the attribution of the SAME beat after noise.
Both explain the class predicted on the clean beat, so differences come from the noise, not a changed target.
Variants: RF+SHAP (seed 0), CNN+Grad-CAM and CNN+IntegratedGradients (seeds 0-2), each clean-trained and noise-augmented."""
import sys, pickle, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch
from ecg_xai.data import zscore
from ecg_xai.noise import add_noise, NOISE_TYPES, SNR_LEVELS
from ecg_xai.models import BeatCNN, predict_cnn
from ecg_xai.xai import shap_attr, gradcam, integrated_gradients
from ecg_xai import stability as st

N_PER_CLASS = int(sys.argv[1]) if len(sys.argv) > 1 else 250
d = np.load("data/processed/beats.npz"); X, y = d["X_test"], d["y_test"]
rng = np.random.default_rng(0)
idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:N_PER_CLASS] for c in range(4)])
X = X[idx]; Zc = zscore(X)
noisy = {(k, s): zscore(add_noise(X, k, s, seed=2)) for k in NOISE_TYPES for s in SNR_LEVELS[1:]}

def load_cnn(tag, seed):
    m = BeatCNN(); m.load_state_dict(torch.load(f"data/processed/models/cnn_{tag}_s{seed}.pt")); return m.eval()

jobs = []   # (label, xai_name, training, seed, predict_fn, explain_fn)
for tag in ("clean", "aug"):
    rf = pickle.load(open(f"data/processed/models/rf_{tag}_s0.pkl", "rb"))
    jobs.append(("rf", "SHAP", tag, 0, rf.predict, lambda Z, t, rf=rf: shap_attr(rf, Z, t)))
    for seed in (0, 1, 2):
        c = load_cnn(tag, seed)
        pf = lambda Z, c=c: predict_cnn(c, Z)
        jobs.append(("cnn", "Grad-CAM", tag, seed, pf, lambda Z, t, c=c: gradcam(c, Z, t)))
        jobs.append(("cnn", "IntGrad", tag, seed, pf, lambda Z, t, c=c: integrated_gradients(c, Z, t)))

rows = []
for model, xai, tag, seed, predict, explain in jobs:
    t0 = time.time(); tgt = predict(Zc); clean = explain(Zc, tgt)
    for (kind, snr), Z in noisy.items():
        A = explain(Z, tgt); same = predict(Z) == tgt
        for name, v in {"spearman": st.spearman(clean, A), "top20_jaccard": st.topk_jaccard(clean, A),
                        "top20_dice": st.topk_dice(clean, A), "cosine": st.cosine(clean, A)}.items():
            rows.append(dict(model=model, xai=xai, training=tag, seed=seed, noise=kind, snr_db=snr, metric=name,
                             mean_all=np.nanmean(v), mean_pred_unchanged=np.nanmean(v[same]) if same.any() else np.nan,
                             frac_pred_unchanged=same.mean()))
    pd.DataFrame(rows).to_csv("results/explanation_stability.csv", index=False)
    print(model, xai, tag, seed, f"{time.time() - t0:.0f}s", flush=True)
print("chance-level top-20% Jaccard:", round(st.random_topk_jaccard(), 3))
