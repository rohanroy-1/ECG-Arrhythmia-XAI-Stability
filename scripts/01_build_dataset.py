import sys; sys.path.insert(0, "src")
import numpy as np
from ecg_xai.data import build_dataset, CLASSES

d = build_dataset("data/mit-bih-arrhythmia-database-1.0.0", "data/processed/beats.npz")
for s in ("train", "test"):
    print(s, d[f"X_{s}"].shape, dict(zip(CLASSES, np.bincount(d[f"y_{s}"], minlength=5))))
