# ORBIT-8 Trans-1

Trans-1 is an isolated arranger experiment. It reuses the ORBIT-8 v1.7.1
timing model and safety pipeline, but it does not replace or modify the current
v1.7.1 arranger or checkpoint.

## Architecture

- 2,958,842 parameters
- RMSNorm pre-normalization
- rotary position encoding (RoPE)
- grouped-query attention (8 query heads, 2 key/value heads)
- PyTorch scaled-dot-product attention
- SwiGLU feed-forward layers
- alternating global-attention and gated linear-cost sequence-mixer blocks
- three-pass parallel lane refinement at inference

The gated mixer is Mamba-inspired but is not presented as a complete Mamba-2
implementation. MoE and MLA were intentionally excluded because the training
set and 384-event context do not justify their routing and capacity overhead.

## Training

- Source charts: the same 451 prepared official charts as v1.7.1
- Train windows after mirror augmentation: 16,208
- Validation windows: 593
- Warm-started compatible input/output tensors: 25
- Best checkpoint: epoch 1
- Early stop: epoch 5
- Peak training memory: 1,099.6 MiB

| Validation metric | v1.7.1 | Trans-1 |
| --- | ---: | ---: |
| Loss | 3.0139 | 3.0370 |
| Lane accuracy | 45.16% | 40.76% |
| Slide operator accuracy | 48.57% | 49.17% |
| Pattern accuracy | 89.26% | 91.12% |
| Interaction recall | 79.58% | 80.44% |
| Sweep recall | 68.20% | 79.54% |

## B.M.S. same-seed comparison

Both charts use BPM 200, level 12.6, seed 20260702, the same timing model, and
the same post-generation safety rules.

| Generated metric | v1.7.1 | Trans-1 |
| --- | ---: | ---: |
| Events | 448 | 447 |
| Slides | 53 | 54 |
| Interaction segments | 2 | 3 |
| Sweep segments | 2 | 0 |
| Eighth-note orbit repairs | 20 | 2 |
| Maximum hand demand | 2 | 2 |
| Safety conflicts | 0 | 0 |

## Initial verdict

Trans-1 is not an across-the-board upgrade yet. It produces much less repeated
one-direction eighth-note orbiting and slightly better operator/pattern metrics,
but its raw lane accuracy is lower and its sampled B.M.S. chart underuses sweep
patterns. The architecture is promising enough to keep as a separate branch;
the next experiment should use a lower learning rate and select checkpoints
with a lane/operator-balanced validation score instead of total loss alone.
