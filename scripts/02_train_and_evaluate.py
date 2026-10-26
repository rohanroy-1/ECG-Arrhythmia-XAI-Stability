"""Train RF + 1D-CNN on clean DS1, evaluate on clean and noisy DS2 (accuracy / F1 / prediction consistency)."""
import sys, pickle; sys.path.insert(0, "src")
import numpy as np, pandas as pd, torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from ecg_xai.data import zscore, CLASSES
from ecg_xai.noise import add_noise, NOISE_TYPES, SNR_LEVELS
from ecg_xai.models import train_rf, train_cnn, predict_cnn

d = np.load("data/processed/beats.npz")
Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
rf = train_rf(zscore(Xtr), ytr)
print("training CNN"); cnn = train_cnn(zscore(Xtr), ytr)
pickle.dump(rf, open("data/processed/rf.pkl", "wb")); torch.save(cnn.state_dict(), "data/processed/cnn.pt")

pred = {"rf": lambda Z: rf.predict(Z), "cnn": lambda Z: predict_cnn(cnn, Z)}
clean_pred = {k: f(zscore(Xte)) for k, f in pred.items()}
rows = []
for kind in NOISE_TYPES:
    for snr in SNR_LEVELS:
        Z = zscore(add_noise(Xte, kind, snr, seed=1))
        for m, f in pred.items():
            p = f(Z)
            rows.append(dict(model=m, noise=kind, snr_db="clean" if snr is None else snr,
                             accuracy=accuracy_score(yte, p), macro_f1=f1_score(yte, p, average="macro"),
                             pred_consistency=(p == clean_pred[m]).mean()))
        print(kind, snr, flush=True)
df = pd.DataFrame(rows); df.to_csv("results/classification_vs_noise.csv", index=False)
for m, p in clean_pred.items():
    print(m, "clean DS2\n", pd.DataFrame(confusion_matrix(yte, p), index=CLASSES, columns=CLASSES))
    print("macro-F1", f1_score(yte, p, average="macro"), "acc", accuracy_score(yte, p))
