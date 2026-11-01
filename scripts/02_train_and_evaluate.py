"""Train {RF, CNN} x {clean, noise-augmented} x 3 seeds on DS1; evaluate on clean + noisy DS2."""
import sys, pickle, os; sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, f1_score
from ecg_xai.data import zscore
from ecg_xai.noise import add_noise, NOISE_TYPES, SNR_LEVELS
from ecg_xai.models import train_rf, train_cnn, predict_cnn

SEEDS = [0, 1, 2]
os.makedirs("data/processed/models", exist_ok=True)
d = np.load("data/processed/beats.npz")
Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
noisy = {(k, s): zscore(add_noise(Xte, k, s, seed=1)) for k in NOISE_TYPES for s in SNR_LEVELS}
rows = []
for aug in (False, True):
    for seed in SEEDS:
        tag = "aug" if aug else "clean"
        print(f"== seed {seed} {tag}", flush=True)
        rf = train_rf(Xtr, ytr, seed=seed, augment=aug)
        pickle.dump(rf, open(f"data/processed/models/rf_{tag}_s{seed}.pkl", "wb"))
        cnn = train_cnn(Xtr, ytr, seed=seed, augment=aug)
        torch.save(cnn.state_dict(), f"data/processed/models/cnn_{tag}_s{seed}.pt")
        pred = {"rf": rf.predict, "cnn": lambda Z: predict_cnn(cnn, Z)}
        for m, f in pred.items():
            cp = f(noisy[(NOISE_TYPES[0], None)])
            for (kind, snr), Z in noisy.items():
                p = f(Z)
                rows.append(dict(model=m, training=tag, seed=seed, noise=kind, snr_db="clean" if snr is None else snr,
                                 accuracy=accuracy_score(yte, p), macro_f1=f1_score(yte, p, average="macro"),
                                 pred_consistency=(p == cp).mean()))
        pd.DataFrame(rows).to_csv("results/classification_vs_noise.csv", index=False)
