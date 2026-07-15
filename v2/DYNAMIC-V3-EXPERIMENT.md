# ORBIT-8 Dynamic prepared_v3 Experiment

## Dataset

prepared_v3 is an index-only dynamic dataset layered on prepared_v2. It does
not duplicate chart tensors or audio and adds only 1.93 MiB of metadata.

| Item | Value |
| --- | ---: |
| Train charts | 356 |
| Regular anchors | 170,310 |
| Dense anchors | 61,197 |
| Long-object anchors | 56,110 |
| Double-note anchors | 63,952 |
| Silence anchors | 2,125 |

Each epoch draws 4,052 aligned crops. Crop lengths are 8, 12, or 16 measures,
with starts locked to measure boundaries. Sampling is stratified across
regular, dense, long-object, double-note, and silence regions. Frequency
masking, time masking, level variation, spectral tilt, and light noise are
applied online. Padded regions are masked in both targets and audio to prevent
future-context leakage.

QA sampled 512 crops across 225 songs. All chart/audio shapes, valid lengths,
and finite-value checks passed.

## Fair Three-Epoch Comparison

Both runs use the same Trans-02 warm start, random seed, batch size, learning
rate, epoch size, scheduler, and fixed prepared_v2 validation set.

| Validation metric at epoch 3 | Fixed v2 data | Dynamic v3 | Delta |
| --- | ---: | ---: | ---: |
| Selection score | 0.9370 | **0.9420** | +0.0050 |
| Onset F1 | 0.7399 | **0.7457** | +0.0059 |
| Subdivision accuracy | 0.6381 | **0.6425** | +0.0044 |
| Dense-rhythm F1 | **0.6750** | 0.6703 | -0.0047 |
| Event F1 | **0.3388** | 0.3282 | -0.0106 |
| Count accuracy | 0.9716 | **0.9722** | +0.0006 |

Dynamic slicing improves onset and subdivision learning speed, but the current
sampling/loss balance weakens event-type classification.

## Held-Out Test Split

The three-epoch dynamic checkpoint is compared with the nine-epoch fixed v2
checkpoint, so this table measures practical status rather than equal compute.

| Test metric | Fixed v2 | Dynamic v3 trial |
| --- | ---: | ---: |
| Onset F1 | **0.7218** | 0.7213 |
| Subdivision accuracy | 0.6235 | **0.6288** |
| Dense-rhythm F1 | 0.6431 | **0.6456** |
| Event F1 | 0.3148 | **0.3151** |
| Count accuracy | **0.9732** | 0.9720 |

Structural generalization improves slightly. Onset performance is effectively
flat after only three dynamic epochs.

## B.M.S. Same-Seed Result

| Metric | Fixed v2 | Dynamic v3 trial |
| --- | ---: | ---: |
| Exact official-tick precision | 86.20% | **86.44%** |
| 16th-note gaps | 25 | **26** |
| Interaction notes | **16** | 14 |
| Safety conflicts | 0 | 0 |

## Decision

Keep prepared_v3. It provides real but modest gains and costs almost no disk
space. Do not replace the current v2 checkpoint yet. Before a full ten-epoch
run, increase event-head protection and separate dense sampling by 16th,
24th, and 32nd subdivision so the model does not trade note typing for onset
quality.
