# ORBIT-8 v2 Hierarchical Rhythm Report

## What Changed

ORBIT-8 v2 is an independent hierarchical generation path. It keeps the
proven Trans-1 arranger and safety pipeline, while replacing flat onset-only
timing with a multi-task rhythm planner.

The v2 model predicts five synchronized views of every chart tick:

- onset confidence;
- note/event type;
- simultaneous note count;
- metrical subdivision: slow, eighth, triplet, sixteenth, twenty-fourth,
  thirty-second, or irregular;
- accent strength derived from official doubles, breaks, holds, and slides.

The decoder uses these predictions as rhythm priors. Sixteenth-note evidence
receives the strongest dense-pattern weight; twenty-fourth and thirty-second
evidence receives a smaller weight to avoid unexplained fast notes.

## Training

| Item | Value |
| --- | ---: |
| Training windows | 4,052 |
| Validation windows | 593 |
| Epochs | 10 |
| Best epoch | 9 |
| Parameters | 4,145,106 |
| Warm-started tensors | 205 |
| Peak training memory | 1,322.5 MiB |

Checkpoint selection balances onset F1, subdivision accuracy, dense-rhythm
F1, and event F1. This prevents a checkpoint with slightly better onset
detection from winning after its dense-pattern understanding has degraded.

## Validation

| Metric | v1.7.1 timing | Trans-02 | ORBIT-8 v2 |
| --- | ---: | ---: | ---: |
| Onset F1 | 0.7411 | **0.7451** | 0.7441 |
| Event F1 | **0.3945** | 0.3502 | 0.3292 |
| Count accuracy | 0.9704 | 0.9715 | **0.9730** |
| Subdivision accuracy | - | - | **0.6378** |
| Dense-rhythm F1 | - | - | **0.6683** |

v2 nearly matches Trans-02 onset quality while adding explicit metrical
understanding. Event classification remains the clearest weak point.

## Held-Out Test Split

The final checkpoint was also evaluated on all 496 test windows:

| Metric | Test result |
| --- | ---: |
| Onset F1 | 0.7218 |
| Event F1 | 0.3148 |
| Count accuracy | 0.9732 |
| Subdivision accuracy | 0.6235 |
| Dense-rhythm F1 | 0.6431 |

Subdivision and dense-rhythm understanding generalize to unseen songs, but
the lower onset F1 confirms that cross-song audio timing is still the main v2
bottleneck.

## B.M.S. Same-Seed Result

All charts use BPM 200, offset 0, level 12.6, and seed 20260702.

| Metric | v1.7.1 | Trans-02 | ORBIT-8 v2 |
| --- | ---: | ---: | ---: |
| Events | 448 | 455 | 461 |
| Exact official-tick precision | **88.04%** | 86.17% | 86.20% |
| 16th-note gaps | 51 | 11 | 25 |
| Interaction segments | 2 | 0 | 2 |
| Interaction notes | - | 0 | 16 |
| Safety conflicts | 0 | 0 | 0 |
| Maximum hand demand | 2 | 2 | 2 |

The structured decoder more than doubles the number of sixteenth-note gaps
relative to Trans-02 and restores two interaction phrases. It does not inject
unexplained twenty-fourth or thirty-second runs. Exact timing alignment on
this song is still below v1.7.1, so v2 is not yet a universal replacement.

## Release Status

This checkpoint is suitable as a v2 preview model. The next training revision
should improve event-type supervision and evaluate per-song timing across the
entire test split instead of optimizing from one B.M.S. example.
