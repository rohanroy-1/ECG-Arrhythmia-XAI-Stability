import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from .data import BEAT_LEN, CLASSES


def train_rf(Xz, y, seed=0, cap_per_class=6000):
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.permutation(np.where(y == c)[0])[:cap_per_class] for c in range(len(CLASSES))])
    rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=3, max_features="sqrt",
                                class_weight="balanced", n_jobs=-1, random_state=seed)
    return rf.fit(Xz[idx], y[idx])


class BeatCNN(nn.Module):
    def __init__(self, n_classes=len(CLASSES)):
        super().__init__()
        def blk(i, o, k): return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())
        self.b1, self.b2 = blk(1, 16, 7), blk(16, 32, 5)
        self.pool = nn.MaxPool1d(2)
        self.b3 = blk(32, 64, 5)          # last conv layer -> Grad-CAM target
        self.head = nn.Linear(64, n_classes)

    def features(self, x):
        return self.b3(self.pool(self.b2(self.pool(self.b1(x)))))

    def forward(self, x):
        return self.head(self.features(x).mean(-1))


def train_cnn(Xz, y, epochs=12, seed=0, bs=256, lr=2e-3):
    torch.manual_seed(seed)
    m = BeatCNN()
    w = torch.tensor(1.0 / np.sqrt(np.bincount(y, minlength=len(CLASSES)).clip(1)), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=w / w.mean())
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-3)
    X = torch.tensor(Xz).unsqueeze(1); Y = torch.tensor(y)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(X)); tot = 0
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            opt.zero_grad(); loss = lossf(m(X[b]), Y[b]); loss.backward(); opt.step(); tot += loss.item() * len(b)
        print(f"  epoch {ep + 1}/{epochs} loss {tot / len(X):.4f}", flush=True)
    return m.eval()


@torch.no_grad()
def predict_cnn(m, Xz, bs=2048):
    m.eval()
    return np.concatenate([m(torch.tensor(Xz[i:i + bs]).unsqueeze(1)).argmax(1).numpy() for i in range(0, len(Xz), bs)])
