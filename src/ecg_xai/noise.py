"""Controlled ECG noise at a target SNR (dB), computed per beat against the clean beat power."""
import numpy as np
from scipy.signal import butter, sosfiltfilt

from .data import BEAT_LEN, FS

NOISE_TYPES = ["gaussian", "baseline_wander", "muscle"]
SNR_LEVELS = [None, 24, 18, 12, 6, 0]   # None = clean


def _unit_noise(kind, n, rng):
    if kind == "gaussian":
        z = rng.standard_normal((n, BEAT_LEN))
    elif kind == "muscle":   # EMG-like: white noise high-passed at 20 Hz
        sos = butter(4, 20, btype="high", fs=FS, output="sos")
        z = sosfiltfilt(sos, rng.standard_normal((n, BEAT_LEN)), axis=1)
    elif kind == "baseline_wander":  # slow sinusoids 0.15-0.5 Hz with random phase
        t = np.arange(BEAT_LEN) / FS
        f = rng.uniform(0.15, 0.5, (n, 1))
        ph = rng.uniform(0, 2 * np.pi, (n, 1))
        z = np.sin(2 * np.pi * f * t + ph)
    else:
        raise ValueError(kind)
    return z / (z.std(1, keepdims=True) + 1e-12)


def add_noise(X, kind, snr_db, seed=0):
    """X: beats in mV [n,180]. Noise power is scaled so that 10*log10(P_signal/P_noise) = snr_db."""
    if snr_db is None:
        return X.copy()
    rng = np.random.default_rng(seed)
    p_sig = (X - X.mean(1, keepdims=True)).var(1, keepdims=True)
    snr = np.asarray(snr_db, float).reshape(-1, 1) if np.ndim(snr_db) else snr_db   # scalar or per-beat
    scale = np.sqrt(p_sig / (10 ** (snr / 10)))
    return (X + scale * _unit_noise(kind, len(X), rng)).astype(np.float32)


def add_noise_mixed(X, rng, p_clean=0.2, snr_range=(0, 24)):
    """Training-time augmentation: each beat gets a random noise type and SNR (or stays clean with prob p_clean)."""
    out = X.copy()
    kinds = rng.integers(0, len(NOISE_TYPES), len(X))
    snr = rng.uniform(*snr_range, len(X))
    noisy = rng.random(len(X)) >= p_clean
    for k, kind in enumerate(NOISE_TYPES):
        m = noisy & (kinds == k)
        if m.any():
            out[m] = add_noise(X[m], kind, snr[m], seed=int(rng.integers(1 << 31)))
    return out


def add_real_noise(X, kind, snr_db, noise_dir="data/nstdb", seed=0):
    """Real noise from the MIT-BIH Noise Stress Test Database (kind in {'bw','em','ma'}; files bw/em/ma.{hea,dat}).
    Random 180-sample windows from channel 0 are scaled to the target SNR. Download:
    https://physionet.org/content/nstdb/1.0.0/  (not reachable from the sandbox used for development)."""
    import wfdb
    rec = wfdb.rdrecord(f"{noise_dir}/{kind}")
    n = rec.p_signal[:, 0].astype(np.float64)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(n) - BEAT_LEN, len(X))
    w = np.stack([n[s:s + BEAT_LEN] for s in starts])
    w = (w - w.mean(1, keepdims=True)) / (w.std(1, keepdims=True) + 1e-12)
    p_sig = (X - X.mean(1, keepdims=True)).var(1, keepdims=True)
    return (X + np.sqrt(p_sig / (10 ** (snr_db / 10))) * w).astype(np.float32)
