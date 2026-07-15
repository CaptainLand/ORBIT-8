# ORBIT-8 v2 16M Report

## Model

| Item | Value |
| --- | ---: |
| Parameters | 15,910,098 |
| Increase over v2 | 11,764,992 |
| Relative size | 3.84x |
| Sequence width | 288 |
| Sequence layers | 12 |
| Feedforward width | 1,152 |
| Attention heads / KV heads | 8 / 2 |
| Inference checkpoint | 60.8 MiB |

The existing 160-dim transient U-Net is retained. Identity-initialized
160-to-288 and 288-to-160 adapters surround the larger hybrid sequence core.
The model inherits 148 compatible tensors from v2 and is distilled from the
4M v2 teacher on onset, event, and count predictions.

Training uses dynamic prepared_v3 crops. The first two epochs freeze inherited
modules, followed by full fine-tuning and one low-learning-rate calibration
epoch on fixed windows. Peak training memory was 1,617 MiB.

## Validation

| Metric | v2 4M | Dynamic 4M | 16M calibrated |
| --- | ---: | ---: | ---: |
| Onset F1 | **0.7441** | 0.7457 | 0.7405 |
| Event F1 | 0.3292 | 0.3282 | **0.3456** |
| Subdivision accuracy | 0.6378 | **0.6425** | 0.6375 |
| Dense-rhythm F1 | 0.6683 | 0.6703 | **0.6811** |
| Count accuracy | **0.9730** | 0.9722 | 0.9718 |

The larger model is materially better at note typing and dense rhythmic
structure. Pure onset timing remains slightly weaker than the best 4M model.

## Held-Out Test Split

| Metric | v2 4M | 16M calibrated | Delta |
| --- | ---: | ---: | ---: |
| Onset F1 | **0.7218** | 0.7184 | -0.0034 |
| Event F1 | 0.3148 | **0.3260** | +0.0112 |
| Subdivision accuracy | **0.6235** | 0.6233 | -0.0002 |
| Dense-rhythm F1 | 0.6431 | **0.6558** | +0.0128 |
| Count accuracy | **0.9732** | 0.9724 | -0.0008 |

## B.M.S. Same-Seed Result

| Metric | v2 4M | Dynamic 4M | 16M calibrated |
| --- | ---: | ---: | ---: |
| Exact official-tick precision | 86.20% | 86.44% | **86.45%** |
| 16th-note gaps | 25 | 26 | 26 |
| Interaction segments | 2 | 2 | 1 |
| Sweep segments | 0 | 0 | **1** |
| Safety conflicts | 0 | 0 | 0 |

## Decision

Release as an optional high-capacity model, not as the default timing model.
It produces richer dense structures and better event types while preserving
playability, but the lightweight model still has a small cross-song onset
advantage.
