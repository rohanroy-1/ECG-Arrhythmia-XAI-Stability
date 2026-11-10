### Spearman (clean vs noisy), noise-augmented models, 95% bootstrap CI over beats

| Method | resolution | 24 dB | 12 dB | 0 dB |
|---|---|---|---|---|
| GradCAM | native | 0.97 [0.97, 0.97] | 0.85 [0.84, 0.85] | 0.54 [0.53, 0.55] |
| IntGrad | native | 0.72 [0.72, 0.72] | 0.51 [0.51, 0.51] | 0.24 [0.24, 0.24] |
| IntGrad | pooled | 0.91 [0.91, 0.92] | 0.78 [0.78, 0.79] | 0.49 [0.48, 0.50] |
| SHAP | native | 0.79 [0.78, 0.79] | 0.57 [0.57, 0.58] | 0.41 [0.40, 0.41] |
| SHAP | pooled | 0.92 [0.92, 0.92] | 0.81 [0.81, 0.82] | 0.68 [0.67, 0.69] |

### Floors (Spearman, native resolution, noise-augmented models)

| Method | noisy vs clean @12 dB | noisy vs noisy @12 dB | seed floor (clean beats) |
|---|---|---|---|
| GradCAM | 0.85 [0.84, 0.85] | 0.80 [0.80, 0.81] | 0.72 [0.71, 0.72] |
| IntGrad | 0.51 [0.51, 0.51] | 0.42 [0.42, 0.43] | 0.64 [0.64, 0.65] |
| SHAP | 0.57 [0.57, 0.58] | 0.52 [0.51, 0.53] | 0.53 [0.52, 0.53] |

### Paired Wilcoxon on per-beat Spearman @12 dB, noise-augmented, pooled resolution

| Comparison | median diff | p |
|---|---|---|
| IntGrad - GradCAM | -0.06 | 1.3e-111 |
| SHAP - GradCAM | -0.02 | 1.3e-23 |
| SHAP - IntGrad | +0.04 | 1.0e-21 |

### Per-class Spearman @12 dB (noise-augmented, native)

| Method | N | S | V | F |
|---|---|---|---|---|
| GradCAM | 0.83 | 0.83 | 0.88 | 0.85 |
| IntGrad | 0.46 | 0.52 | 0.54 | 0.53 |
| SHAP | 0.55 | 0.55 | 0.70 | 0.50 |

Fraction of beats with undefined Spearman (constant attribution): {'GradCAM': np.float64(0.0), 'IntGrad': np.float64(0.0), 'SHAP': np.float64(0.0)}