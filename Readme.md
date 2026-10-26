# ECG Arrhythmia Classification with Robust Explainability

## Overview

This project investigates the robustness of explainable AI
methods for ECG arrhythmia classification under controlled
signal degradation.

The study evaluates whether model explanations remain stable
when the same ECG beat is exposed to increasing levels of noise.

## Research Question

Does explanation stability degrade as ECG signal noise increases,
even when predictive performance remains relatively stable?

## Pipeline

ECG Dataset
→ Preprocessing
→ Beat Segmentation
→ Clean / Noisy ECG
→ Model Training
→ Classification
→ XAI
→ Explanation Stability
→ Robustness Evaluation

## Dataset

MIT-BIH Arrhythmia Database

## Models

Models will be evaluated progressively, including classical
machine-learning and deep-learning approaches.

## Explainability

- SHAP for feature-level explanations
- Grad-CAM / suitable XAI methods for deep models

## Evaluation

### Classification
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### Explanation Stability
- SHAP rank correlation / feature stability
- Grad-CAM IoU / Dice / heatmap similarity
- Prediction consistency

## How to run

```bash
pip install -r requirements.txt
python scripts/01_build_dataset.py        # segment beats, AAMI labels, inter-patient split -> data/processed/beats.npz
python scripts/02_train_and_evaluate.py   # RF + 1D-CNN on clean DS1; accuracy/F1 on noisy DS2
python scripts/03_explanation_stability.py 250   # SHAP / Grad-CAM clean-vs-noisy comparison (250 beats per class)
python scripts/04_plots.py                # results/stability_vs_noise.png
```

## Setup actually used

- MLII lead, 0.5-40 Hz zero-phase band-pass, 180-sample beats (90 before / 90 after the R peak), per-beat z-score applied *after* noise injection.
- AAMI classes N / S / V / F. Class Q is dropped (7-8 beats in the non-paced records). Paced records 102, 104, 107, 217 excluded.
- **Inter-patient split** (de Chazal DS1 train / DS2 test) so no patient appears in both. 51,002 train / 49,691 test beats.
- Noise: Gaussian, baseline wander (0.15-0.5 Hz), muscle/EMG-like (>20 Hz), at SNR 24, 18, 12, 6, 0 dB relative to each beat's power.
- Models: Random Forest (class-balanced, waveform only) and a small 1D-CNN. Both trained on clean data only.
- Explanations: SHAP TreeExplainer (RF), 1D Grad-CAM on the last conv block (CNN). The clean and noisy beat are both explained for the class predicted on the clean beat, so only the noise differs.
- Stability metrics: Spearman rank correlation, top-20% Jaccard and Dice, cosine similarity, prediction consistency. Chance-level top-20% Jaccard is 0.11.

## Results so far (`results/`)

| | RF + SHAP | CNN + Grad-CAM |
|---|---|---|
| Clean DS2 accuracy / macro-F1 | 0.68 / 0.39 | 0.86 / 0.45 |

Explanation Spearman vs. clean (all beats):

| Noise | Model | 24 dB | 18 dB | 12 dB | 6 dB | 0 dB |
|---|---|---|---|---|---|---|
| Gaussian | RF / SHAP | 0.80 | 0.69 | 0.57 | 0.44 | 0.34 |
| Gaussian | CNN / Grad-CAM | 0.76 | 0.50 | 0.26 | 0.13 | 0.08 |
| Baseline wander | RF / SHAP | 0.78 | 0.66 | 0.54 | 0.41 | 0.32 |
| Baseline wander | CNN / Grad-CAM | 0.95 | 0.86 | 0.67 | 0.44 | 0.20 |

- Explanation stability falls monotonically with noise for every noise type and both models.
- At 18 dB Gaussian/muscle noise the CNN keeps only ~58% of its predictions and ~0.50 Spearman, so on this setup "stable accuracy but unstable explanations" is **not** demonstrated for the CNN. Below 12 dB it collapses (accuracy ~7%).
- The RF's predictions drift less than the CNN's under Gaussian/muscle noise, yet its explanations are only moderately stable (0.57 Spearman at 12 dB).
- The CNN is much more robust to baseline wander than to broadband noise (0.86 Spearman at 18 dB).

## Known limitations / next steps

- Weak S and F recall (typical of inter-patient evaluation). Adding RR-interval features and/or noise-augmented training are the obvious next steps, and noise-augmented training is the fair test of the research question (stable accuracy, unstable explanations).
- Single seed, single split, no confidence intervals yet.
- Grad-CAM from one small CNN; add a deeper model and Integrated Gradients / saliency as a second XAI method.
- Real noise (MIT-BIH NSTDB) instead of synthetic.

