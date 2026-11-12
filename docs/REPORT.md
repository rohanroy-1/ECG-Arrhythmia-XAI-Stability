# Do ECG explanations survive noise that predictions survive? A controlled study on MIT-BIH

*Working draft. All numbers are from `results/` and reproducible with `scripts/01`-`06`.*

## 1. Question
Does explanation stability degrade as ECG noise increases, even when predictive performance stays stable?

## 2. Setup
- **Data:** MIT-BIH, MLII lead, 0.5-40 Hz band-pass, 180-sample beats, AAMI classes N/S/V/F (Q dropped, <10 beats). De Chazal inter-patient split: DS1 train (51,002 beats), DS2 test (49,691). Paced records excluded.
- **Noise:** synthetic Gaussian, baseline wander, EMG-like; SNR 24/18/12/6/0 dB per beat. Noise is added before per-beat z-scoring.
- **Models:** Random Forest and 1D-CNN, each trained *clean* or *noise-augmented* (random type, SNR 0-24 dB); 3 seeds.
- **Explanations:** TreeSHAP (RF), Grad-CAM and Integrated Gradients (CNN). Clean and noisy versions of the same beat are both explained for the class predicted on the clean beat.
- **Stability:** Spearman and top-20% Jaccard between clean and noisy attribution maps (250 beats/class from DS2).
- **Controls:** (a) *resolution control* - SHAP/IG pooled to Grad-CAM's effective resolution (4-sample blocks); (b) *noise-draw floor* - two independent noise draws, same beat, same SNR; (c) *seed floor* - same clean beat explained by independently trained models; (d) bootstrap CIs over beats and paired Wilcoxon tests.

## 3. Results

**3.1 Predictions.** Clean-trained CNN accuracy on DS2 is seed-unstable (0.70, std 0.14) and collapses under broadband noise (Gaussian: 0.78 / 0.17 / 0.07 at 24 / 12 / 0 dB). Noise-augmented CNN: 0.87 clean, 0.86 at 12 dB, 0.85 at 0 dB. RF: 0.68 -> 0.82 with augmentation. Macro-F1 is ~0.4 for all models (weak S and F recall).

**3.2 Explanations degrade monotonically with noise** for every method, noise type and training regime (`results/explanation_stability.csv`, `fig stability_vs_noise.png`), including the noise-augmented CNN whose predictions barely move.

**3.3 Resolution matters a lot.** Raw Grad-CAM looks far more stable than IG or SHAP (Spearman at 12 dB, noise-augmented: 0.85 vs 0.51 vs 0.57). After pooling IG and SHAP to Grad-CAM's resolution the gap nearly disappears: 0.78 (IG) and 0.81 (SHAP) vs 0.85. The remaining differences are small but significant (paired Wilcoxon, p < 1e-20; median per-beat difference -0.06 IG-GradCAM, -0.02 SHAP-GradCAM). Most of the "Grad-CAM is more stable" effect is spatial smoothing, not a better explanation.

**3.4 Floors put the effect in context (noise-augmented, 12 dB, Spearman):**

| Method | noisy vs clean | noisy vs noisy | seed floor |
|---|---|---|---|
| Grad-CAM | 0.85 | 0.80 | 0.72 |
| IntGrad | 0.51 | 0.42 | 0.64 |
| SHAP | 0.57 | 0.52 | 0.53 |

For SHAP, 12 dB noise changes the explanation about as much as retraining the model from a different seed. For IG the noise effect exceeds the retraining effect. For Grad-CAM it is smaller. So "explanations are unstable under noise" needs a reference: part of the instability is ordinary model-to-model explanation variability.

**3.5 Per class (12 dB).** No class is an outlier for Grad-CAM (0.83-0.88). SHAP is highest for V (0.70) and lowest for F (0.50); IG 0.46-0.54.

## 4. Conclusion (as supported so far)
1. Explanations degrade steadily with noise even when accuracy is held roughly constant by augmentation (CNN: accuracy 0.86 vs 0.87 clean at 12 dB, yet explanation Spearman 0.51-0.85 depending on method/resolution).
2. The size of that degradation is strongly method- and resolution-dependent; comparisons between XAI methods must control resolution.
3. Seed-to-seed explanation variability is a necessary reference. At 12 dB, SHAP noise-induced drift is comparable to it.

## 5. Threats to validity
- **Synthetic noise only.** Real NSTDB noise is not reachable from the development sandbox; `ecg_xai.noise.add_real_noise` is implemented and tested on stand-in files, awaiting `data/nstdb/{bw,em,ma}`.
- **Bootstrap CIs are over beats**, which are not independent (same patients, same models), so they are far too narrow. Seed std (up to 0.2 for clean-trained CNNs, ~0.05 for augmented) is the more honest uncertainty.
- Spearman on |attribution| ignores sign; top-20% cut-off is arbitrary (Jaccard shows the same ordering).
- Stability is not faithfulness: a stable map may still be wrong. No deletion/insertion fidelity test yet.
- Weak S/F detection; no RR features. Small CNN; one dataset; one lead.
- Seed floors use only three seeds per recipe.

## 6. Next steps
NSTDB real noise; deletion/insertion fidelity curves; RR-interval features and a stronger model; a second dataset (e.g. INCART or PTB-XL) for external validity; per-patient (cluster) bootstrap.
