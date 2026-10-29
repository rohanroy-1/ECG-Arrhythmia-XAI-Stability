# ECG Arrhythmia Classification with Robust Explainability

**Investigating the Stability of Explainable AI Under Controlled ECG Signal Degradation**

## 1. Introduction

Artificial Intelligence and Machine Learning have demonstrated considerable potential in automated electrocardiogram (ECG) analysis and cardiac arrhythmia detection. However, the reliability of these systems depends not only on their predictive performance but also on the consistency and trustworthiness of their explanations.

In real-world clinical environments, ECG signals are frequently affected by noise and other forms of signal degradation. Although a classification model may continue producing predictions under such conditions, its underlying decision-making process and feature attributions may change significantly.

This project investigates the robustness of Explainable Artificial Intelligence (XAI) methods for ECG arrhythmia classification under controlled signal degradation.

The primary objective is to evaluate how increasing levels of ECG noise affect:
- Classification performance and prediction consistency.
- Feature attribution and explanation stability.
- The relationship between predictive robustness and explanation reliability.

The study compares classical machine learning and deep learning approaches using the MIT-BIH Arrhythmia Database.

## 2. Research Question

**Does explanation stability degrade as ECG signal noise increases, even when predictive performance remains relatively stable?**

The investigation examines whether model explanations remain consistent when the same ECG heartbeat is exposed to progressively increasing levels of controlled noise.

Rather than evaluating classification accuracy alone, the project treats explanation stability as an additional dimension of model reliability.

## 3. Research Objectives

1. Develop a reproducible ECG preprocessing and arrhythmia classification pipeline.
2. Evaluate classical machine learning and deep learning models using an inter-patient evaluation strategy.
3. Introduce controlled signal degradation at multiple signal-to-noise ratio (SNR) levels.
4. Apply explainability techniques to understand model predictions.
5. Quantitatively measure changes in explanations under increasing noise.
6. Compare predictive consistency with explanation stability.
7. Identify the limitations of existing explainability approaches under degraded signal conditions.

## 4. Research Pipeline

The project follows a modular experimental pipeline:

```text
MIT-BIH Arrhythmia Database
           |
           v
    ECG Preprocessing
           |
           v
     Beat Segmentation
           |
           v
    AAMI Classification
           |
           v
   Inter-Patient Split
      (DS1 / DS2)
           |
           v
    Model Training
       /       \
      v         v
 Random Forest  1D-CNN
      \         /
       v       v
   Clean / Noisy Evaluation
           |
           v
    Explainability Analysis
       /             \
      v               v
   SHAP            Grad-CAM
      \               /
       v             v
    Explanation Stability
           |
           v
   Robustness Evaluation
```

Each stage is implemented independently to support reproducibility, experimentation, and future extensions.

## 5. Dataset and Preprocessing

### Dataset

**MIT-BIH Arrhythmia Database**

The experiments use the MIT-BIH Arrhythmia Database, a widely used benchmark for ECG arrhythmia classification.

- Signal lead: MLII
- Classification scheme: AAMI
- Target classes: N, S, V and F
- Paced records excluded: 102, 104, 107 and 217
- Q-class beats excluded due to their limited representation in the selected non-paced records.

### Signal Preprocessing

The ECG preprocessing pipeline consists of:

- Zero-phase Butterworth band-pass filtering (0.5–40 Hz).
- R-peak-based heartbeat segmentation.
- Fixed-length beat extraction using 90 samples before and 90 samples after the R peak.
- Per-beat z-score normalization.

Each heartbeat is represented as a 180-sample waveform.

Normalization is performed after noise injection during noisy evaluation so that the effects of signal degradation can be studied under a consistent preprocessing procedure.

### Inter-Patient Evaluation

To avoid patient-level data leakage, the project uses the de Chazal DS1/DS2 inter-patient partition.

| Dataset | Purpose | Number of beats |
|---|---|---:|
| DS1 | Training | 51,002 |
| DS2 | Testing | 49,691 |

No patient appears in both training and testing partitions.

This evaluation strategy provides a more realistic assessment of generalization to previously unseen patients.

## 6. Controlled Noise Injection

To investigate the effects of signal degradation, three types of noise are introduced independently into ECG beats.

| Noise type | Description |
|---|---|
| Gaussian noise | Random broadband additive noise |
| Baseline wander | Low-frequency drift (0.15–0.5 Hz) |
| Muscle noise | EMG-like high-frequency interference (>20 Hz) |

### Signal-to-Noise Ratio Levels

Each noise type is evaluated at five SNR levels:

**24 dB, 18 dB, 12 dB, 6 dB and 0 dB**

Noise intensity is calculated relative to the power of each individual ECG beat.

Both models are trained exclusively on clean training data. Their performance is subsequently evaluated on clean and progressively degraded test signals.

This allows the study to investigate model robustness without introducing noise-augmented training into the initial experiments.

## 7. Classification Models

Two classification approaches are currently implemented.

### 7.1 Random Forest

A class-balanced Random Forest classifier serves as the classical machine learning baseline.

- Input: 180-sample ECG waveform.
- Class balancing enabled.
- Training performed using clean DS1 data.
- Evaluation performed using clean and noisy DS2 data.

### 7.2 One-Dimensional Convolutional Neural Network

A lightweight 1D-CNN is used to investigate deep learning-based ECG classification.

The architecture processes the temporal structure of ECG waveforms through convolutional layers and learns representations useful for arrhythmia classification.

- Input: 180-sample ECG waveform.
- Training performed using clean DS1 data.
- Evaluation performed using clean and noisy DS2 data.
- Grad-CAM is extracted from the final convolutional block.

## 8. Explainable Artificial Intelligence

The project evaluates different explanation methods depending on the underlying model architecture.

### 8.1 SHAP — Random Forest

SHAP (SHapley Additive exPlanations) is used to estimate the contribution of individual waveform features to model predictions.

TreeExplainer is employed to generate feature-level explanations for the Random Forest classifier.

The resulting attribution values are used to examine which ECG samples contribute most strongly to a prediction and whether those contributions remain consistent under noise.

### 8.2 Grad-CAM — 1D-CNN

Gradient-weighted Class Activation Mapping (Grad-CAM) is adapted to the temporal structure of one-dimensional ECG signals.

Grad-CAM generates temporal attribution maps that highlight portions of the input waveform contributing to a selected model prediction.

The final convolutional block is used to extract activation maps for the deep learning model.

### Explanation Comparison Protocol

For each selected heartbeat:

1. Generate an explanation for the clean ECG signal.
2. Generate a noisy version of the same heartbeat.
3. Obtain the explanation for the same reference class: the class predicted on the clean beat.
4. Compare clean and noisy attribution patterns.

This protocol isolates changes associated with signal degradation rather than changing the explanation target between the two inputs.

## 9. Evaluation Metrics

### 9.1 Classification Performance

The following metrics are used to evaluate classification quality:

- Accuracy
- Precision
- Recall
- Macro F1-score
- Confusion Matrix
- Prediction consistency under noise

Macro-averaged metrics are particularly important because the AAMI classes have an imbalanced distribution.

### 9.2 Explanation Stability

Explanation stability is evaluated by comparing attribution patterns generated from clean and noisy versions of the same ECG beat.

| Metric | Purpose |
|---|---|
| Spearman Rank Correlation | Measures similarity in feature attribution rankings |
| Top-20% Jaccard Similarity | Measures overlap between the most important features |
| Dice Coefficient | Measures similarity between selected important-feature sets |
| Cosine Similarity | Measures similarity between attribution vectors |
| Prediction Consistency | Measures whether predictions remain unchanged under noise |

For the top-20% feature selection metric, the chance-level Jaccard similarity under an independent random selection model is approximately 0.11.

Higher stability scores indicate greater similarity between clean and noisy explanations.

## 10. Experimental Results

### 10.1 Clean Test Performance

The initial models were evaluated on the clean DS2 test partition.

| Metric | Random Forest + SHAP | 1D-CNN + Grad-CAM |
|---|---:|---:|
| Accuracy | 0.68 | 0.86 |
| Macro F1-score | 0.39 | 0.45 |

The CNN achieves higher overall accuracy than the Random Forest model. However, the macro F1-scores indicate that performance across the four arrhythmia classes remains uneven.

In particular, S and F class recall remain areas requiring improvement.

### 10.2 Explanation Stability Under Gaussian Noise

Spearman rank correlation between clean and noisy explanations:

| SNR | RF + SHAP | CNN + Grad-CAM |
|---|---:|---:|
| 24 dB | 0.80 | 0.76 |
| 18 dB | 0.69 | 0.50 |
| 12 dB | 0.57 | 0.26 |
| 6 dB | 0.44 | 0.13 |
| 0 dB | 0.34 | 0.08 |

### 10.3 Explanation Stability Under Baseline Wander

| SNR | RF + SHAP | CNN + Grad-CAM |
|---|---:|---:|
| 24 dB | 0.78 | 0.95 |
| 18 dB | 0.66 | 0.86 |
| 12 dB | 0.54 | 0.67 |
| 6 dB | 0.41 | 0.44 |
| 0 dB | 0.32 | 0.20 |

### 10.4 Preliminary Observations

The initial experiments reveal several interesting patterns:

**1. Explanation stability decreases with increasing noise.**

Both models exhibit declining Spearman correlation as signal quality deteriorates across the evaluated noise conditions.

**2. Different models exhibit different sensitivity to noise.**

The Random Forest model shows relatively gradual degradation in explanation stability under Gaussian noise, whereas the CNN exhibits a much sharper decline.

**3. Noise characteristics matter.**

The CNN demonstrates greater explanation stability under baseline wander than under broadband Gaussian noise. At 18 dB, its baseline-wander explanation correlation is 0.86, compared with 0.50 under Gaussian noise.

**4. Predictive and explanatory robustness are not interchangeable.**

The initial experiments do not yet establish the central hypothesis of stable predictive performance alongside unstable explanations for the CNN.

At 18 dB Gaussian and muscle noise, the CNN retains approximately 58% prediction consistency and exhibits approximately 0.50 explanation correlation. At lower SNR levels, its classification performance deteriorates substantially.

The Random Forest exhibits comparatively less prediction drift under Gaussian and muscle noise, although its explanations also become progressively less stable.

These observations motivate further experiments investigating the relationship between prediction robustness and explanation reliability.

## 11. Current Limitations

The current implementation provides a foundation for investigating explanation robustness, but several limitations remain.

- **Class imbalance:** S and F classes have relatively weak recall under inter-patient evaluation.
- **Limited experimental repetitions:** Current results use a single random seed and one patient partition.
- **Statistical uncertainty:** Confidence intervals and statistical significance tests have not yet been established.
- **Limited model diversity:** The deep learning experiments currently use a lightweight 1D-CNN.
- **Limited XAI diversity:** Grad-CAM is currently the primary explanation method for the CNN.
- **Synthetic noise:** Initial experiments use controlled synthetic noise rather than real-world recorded noise.
- **Feature representation:** The initial models use waveform samples without additional RR-interval features.

These limitations are treated as directions for subsequent investigation rather than conclusions about the general reliability of XAI methods.

## 12. Future Work

The next research phase will extend the current experimental framework in the following directions.

### 12.1 Improving Classification Robustness

- Investigate RR-interval and additional morphological features.
- Evaluate noise-augmented training.
- Compare clean-only and noise-augmented training strategies.
- Investigate class imbalance mitigation.
- Evaluate additional deep learning architectures.

### 12.2 Expanding Explainability Methods

- Implement Integrated Gradients.
- Evaluate gradient-based saliency methods.
- Compare explanation stability across different XAI techniques.
- Investigate the sensitivity of explanations to model architecture.

### 12.3 Strengthening Experimental Validation

- Repeat experiments across multiple random seeds.
- Calculate confidence intervals.
- Perform statistical comparisons across noise conditions.
- Investigate the relationship between classification consistency and explanation stability.
- Evaluate additional patient partitions where feasible.

### 12.4 Real-World Noise Evaluation

Introduce real-world ECG noise using the MIT-BIH Noise Stress Test Database (NSTDB).

This will help determine whether observations made under synthetic noise conditions generalize to more realistic ECG degradation.

### 12.5 Interactive Research Dashboard

Develop an interactive interface to explore:

- Clean and noisy ECG waveforms.
- SNR-dependent classification performance.
- Model predictions.
- SHAP feature attributions.
- Grad-CAM temporal explanations.
- Explanation stability across noise conditions.

## 13. Repository Structure

```text
ECG-Arrhythmia-XAI-Stability/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│
├── scripts/
│   ├── 01_build_dataset.py
│   ├── 02_train_and_evaluate.py
│   ├── 03_explanation_stability.py
│   └── 04_plots.py
│
├── results/
│   └── stability_vs_noise.png
│
├── tests/
│
├── requirements.txt
├── test_ecg.py
└── README.md
```

The structure represents the intended organization of the experimental pipeline. Individual directories and generated outputs may depend on the current implementation state.

## 14. Installation and Execution

### Prerequisites

- Python 3.10 or later
- MIT-BIH Arrhythmia Database
- Sufficient memory for ECG preprocessing and model training

### Installation

Clone the repository:

```bash
git clone https://github.com/rohanroy-1/ECG-Arrhythmia-XAI-Stability.git

cd ECG-Arrhythmia-XAI-Stability
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Dataset Preparation

Place the MIT-BIH database files in the configured raw-data directory.

Build the processed dataset:

```bash
python scripts/01_build_dataset.py
```

This stage performs ECG preprocessing, beat segmentation, AAMI label mapping and inter-patient partitioning.

Expected output:

```text
data/processed/beats.npz
```

### Model Training and Evaluation

```bash
python scripts/02_train_and_evaluate.py
```

This stage trains the Random Forest and 1D-CNN models on clean DS1 data and evaluates their classification performance on clean and noisy DS2 signals.

### Explanation Stability Analysis

```bash
python scripts/03_explanation_stability.py 250
```

The argument specifies the number of beats per class selected for explanation analysis.

The script compares clean and noisy SHAP and Grad-CAM explanations.

### Result Visualization

```bash
python scripts/04_plots.py
```

This generates the stability-versus-noise visualization:

```text
results/stability_vs_noise.png
```

*Note: Execution commands and output paths assume the corresponding scripts and directories are present in the checked-out implementation.*

## 15. Research Contribution

This project explores an important aspect of trustworthy AI in biomedical signal processing: the reliability of model explanations under imperfect input conditions.

Its central contribution is an experimental framework that evaluates predictive performance and explanation stability together rather than treating classification accuracy as the sole indicator of model reliability.

By comparing different model architectures, controlled noise conditions and explainability techniques, the study aims to identify how signal degradation influences the interpretability of automated ECG arrhythmia classification systems.

The longer-term objective is to develop a reproducible evaluation framework for understanding the limitations of explainable AI in noise-sensitive biomedical applications.

---

**Project:** ECG Arrhythmia Classification with Robust Explainability  
**Dataset:** MIT-BIH Arrhythmia Database  
**Primary Research Area:** Explainable AI, Biomedical Signal Processing, Robust Machine Learning  
**Repository:** https://github.com/rohanroy-1/ECG-Arrhythmia-XAI-Stability