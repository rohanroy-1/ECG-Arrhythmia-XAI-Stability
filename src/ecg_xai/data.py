"""MIT-BIH loading, filtering, beat segmentation, AAMI labelling, inter-patient split."""
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

FS = 360
PRE, POST = 90, 90            # 0.25 s before / after the R peak -> 180 samples
BEAT_LEN = PRE + POST
CLASSES = ["N", "S", "V", "F"]   # Q (paced/unknown) has <10 beats in non-paced records -> dropped

AAMI = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
}

# Inter-patient split of de Chazal et al. (2004). Paced records 102,104,107,217 are excluded (AAMI).
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
       213, 214, 219, 221, 222, 228, 231, 232, 233, 234]


def bandpass(x, fs=FS, lo=0.5, hi=40.0, order=3):
    """Zero-phase band-pass: removes baseline wander and high-frequency noise of the *source* signal."""
    sos = butter(order, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, x)


def load_record(data_dir, rec):
    """Return (beats[n,180] in mV, labels[n] int, rpeaks[n]) for the MLII lead of one record."""
    path = str(Path(data_dir) / str(rec))
    r = wfdb.rdrecord(path)
    ch = r.sig_name.index("MLII")
    sig = bandpass(r.p_signal[:, ch])
    ann = wfdb.rdann(path, "atr")
    beats, labels, peaks = [], [], []
    for s, sym in zip(ann.sample, ann.symbol):
        if sym not in AAMI or s - PRE < 0 or s + POST > len(sig):
            continue
        beats.append(sig[s - PRE:s + POST])
        labels.append(CLASSES.index(AAMI[sym]))
        peaks.append(s)
    return np.asarray(beats, np.float32), np.asarray(labels, np.int64), np.asarray(peaks)


def build_dataset(data_dir, out_path):
    out = {}
    for name, recs in (("train", DS1), ("test", DS2)):
        X, y, g = [], [], []
        for rec in recs:
            b, l, _ = load_record(data_dir, rec)
            X.append(b); y.append(l); g.append(np.full(len(l), rec))
        out[f"X_{name}"] = np.concatenate(X)
        out[f"y_{name}"] = np.concatenate(y)
        out[f"rec_{name}"] = np.concatenate(g)
    np.savez_compressed(out_path, **out)
    return out


def zscore(X):
    """Per-beat standardisation (applied AFTER noise injection so models see what a deployed system would)."""
    X = np.asarray(X, np.float32)
    return (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-8)
