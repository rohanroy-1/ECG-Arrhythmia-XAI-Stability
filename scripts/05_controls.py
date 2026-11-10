"""Control experiments for the explanation-stability claim (Gaussian noise).

 A. Resolution control   - Grad-CAM is coarse (45 steps upsampled). Re-score SHAP/IG after the same pooling ("smoothed").
 B. Noise-draw floor     - explanation(noise draw 1) vs explanation(noise draw 2), same beat, same SNR.
 C. Seed floor           - explanation of the same clean beat from independently trained models (same recipe).
 D. Bootstrap 95% CIs over beats; per-beat arrays saved for paired tests (analysis in 06_analysis.py).
"""
import sys, pickle, time; sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from ecg_xai.data import zscore
from ecg_xai.noise import add_noise
from ecg_xai.models import BeatCNN, predict_cnn
from ecg_xai.xai import shap_attr, gradcam, integrated_gradients
from ecg_xai import stability as st

SNRS, K, N_PER = [24, 18, 12, 6, 0], 4, 250
d = np.load("data/processed/beats.npz"); X, y = d["X_test"], d["y_test"]
rng = np.random.default_rng(0)
idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:N_PER] for c in range(4)])
X, y = X[idx], y[idx]; cls = np.repeat(np.arange(4), N_PER)
Zc = zscore(X)
Z1 = {s: zscore(add_noise(X, "gaussian", s, seed=2)) for s in SNRS}   # draw 1 (same as script 03)
Z2 = {s: zscore(add_noise(X, "gaussian", s, seed=3)) for s in SNRS}   # draw 2


def smooth(A, k=K):
    t = F.avg_pool1d(torch.tensor(A, dtype=torch.float32).unsqueeze(1), k, ceil_mode=True)
    return F.interpolate(t, size=A.shape[1], mode="linear", align_corners=False).squeeze(1).numpy()


def per_beat(a, b):
    return {"spearman": st.spearman(a, b), "top20_jaccard": st.topk_jaccard(a, b)}


def boot(v, n=1000, seed=0):
    v = v[~np.isnan(v)]; r = np.random.default_rng(seed)
    m = v[r.integers(0, len(v), (n, len(v)))].mean(1)
    return v.mean(), *np.percentile(m, [2.5, 97.5])


store, rows = {}, []   # store[(method,training,series,smooth,snr,metric)] -> per-beat array averaged over seeds


def record(method, training, series, sm, snr, vals_by_seed):
    for metric in ("spearman", "top20_jaccard"):
        v = np.nanmean([vs[metric] for vs in vals_by_seed], 0)
        store[(method, training, series, sm, snr, metric)] = v
        mean, lo, hi = boot(v)
        rows.append(dict(method=method, training=training, series=series, smoothed=sm, snr_db=snr, metric=metric,
                         mean=mean, ci_lo=lo, ci_hi=hi, n_seeds=len(vals_by_seed)))


def run(method, training, models):
    """models: list of (predict_fn, explain_fn) over seeds; attributions explain the class predicted by model #0 on the clean beat."""
    t0 = time.time()
    tgt0 = models[0][0](Zc)
    tgts = [m[0](Zc) for m in models]
    clean = [m[1](Zc, t) for m, t in zip(models, tgts)]
    out = {s: {"d1": [], "d2": []} for s in SNRS}
    for s in SNRS:
        for m, t in zip(models, tgts):
            out[s]["d1"].append(m[1](Z1[s], t)); out[s]["d2"].append(m[1](Z2[s], t))
    for sm in (False, True):
        f = (lambda a: smooth(a)) if sm else (lambda a: a)
        for s in SNRS:
            record(method, training, "noisy_vs_clean", sm, s, [per_beat(f(c), f(a)) for c, a in zip(clean, out[s]["d1"])])
            record(method, training, "noisy_vs_noisy", sm, s, [per_beat(f(a), f(b)) for a, b in zip(out[s]["d1"], out[s]["d2"])])
        if len(models) > 1:   # seed floor: all models explain model-0's predicted class
            c0 = [m[1](Zc, tgt0) for m in models]
            pairs = [(0, 1), (0, 2), (1, 2)]
            record(method, training, "seed_floor", sm, np.nan, [per_beat(f(c0[i]), f(c0[j])) for i, j in pairs])
    print(method, training, f"{time.time() - t0:.0f}s", flush=True)


for tag in ("clean", "aug"):
    cnns = []
    for seed in (0, 1, 2):
        c = BeatCNN(); c.load_state_dict(torch.load(f"data/processed/models/cnn_{tag}_s{seed}.pt")); cnns.append(c.eval())
    pf = lambda c: (lambda Z: predict_cnn(c, Z))
    run("GradCAM", tag, [(pf(c), lambda Z, t, c=c: gradcam(c, Z, t)) for c in cnns])
    run("IntGrad", tag, [(pf(c), lambda Z, t, c=c: integrated_gradients(c, Z, t)) for c in cnns])
    rfs = [pickle.load(open(f"data/processed/models/rf_{tag}_s{s}.pkl", "rb")) for s in (0, 1, 2)]
    run("SHAP", tag, [(rf.predict, lambda Z, t, rf=rf: shap_attr(rf, Z, t)) for rf in rfs])
    pd.DataFrame(rows).to_csv("results/controls.csv", index=False)

np.savez_compressed("data/processed/per_beat_controls.npz", cls=cls,
                    **{"|".join(map(str, k)): v for k, v in store.items()})
