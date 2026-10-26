import numpy as np
import shap
import torch
import torch.nn.functional as F

from .data import BEAT_LEN


def shap_attr(rf, Xz, classes):
    """|SHAP| per time-sample for the given class of each beat -> [n,180]."""
    sv = shap.TreeExplainer(rf).shap_values(Xz, check_additivity=False)
    sv = np.stack(sv, -1) if isinstance(sv, list) else sv          # -> [n, feat, class]
    return np.abs(sv[np.arange(len(Xz)), :, classes])


def gradcam(model, Xz, classes):
    """1D Grad-CAM on the last conv block, upsampled to 180 samples, ReLU'd -> [n,180]."""
    model.eval()
    x = torch.tensor(Xz).unsqueeze(1)
    A = model.features(x).detach().requires_grad_(True)
    logits = model.head(A.mean(-1))
    logits.gather(1, torch.tensor(classes).unsqueeze(1)).sum().backward()
    w = A.grad.mean(-1, keepdim=True)
    cam = F.relu((w * A).sum(1, keepdim=True))
    return F.interpolate(cam, size=BEAT_LEN, mode="linear", align_corners=False).squeeze(1).detach().numpy()
