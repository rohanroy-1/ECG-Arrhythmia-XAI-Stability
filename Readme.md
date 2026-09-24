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