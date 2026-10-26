"""Similarity between the explanation of a clean beat and of the same beat after noise."""
import numpy as np
from scipy.stats import spearmanr


def spearman(a, b):
    out = []
    for x, y in zip(a, b):
        out.append(np.nan if x.std() == 0 or y.std() == 0 else spearmanr(x, y)[0])
    return np.array(out)


def _topk_mask(a, k):
    m = np.zeros_like(a, bool)
    idx = np.argsort(-a, 1)[:, :k]
    np.put_along_axis(m, idx, True, 1)
    return m


def topk_jaccard(a, b, k=36):          # k = 20 % of the 180 samples
    A, B = _topk_mask(a, k), _topk_mask(b, k)
    return (A & B).sum(1) / (A | B).sum(1)


def topk_dice(a, b, k=36):
    A, B = _topk_mask(a, k), _topk_mask(b, k)
    return 2 * (A & B).sum(1) / (A.sum(1) + B.sum(1))


def cosine(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def random_topk_jaccard(n=180, k=36):
    """Expected Jaccard for two independent random top-k sets (chance level)."""
    i = k * k / n
    return i / (2 * k - i)
