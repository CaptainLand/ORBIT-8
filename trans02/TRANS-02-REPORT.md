# ORBIT-8 Trans-02 Report

## Scope

Trans-02 is an independent experiment. It leaves the v1.7.1 pipeline and
Trans-1 checkpoint unchanged, and combines:

- a new hybrid Transformer rhythm model for onset and event extraction;
- the existing Trans-1 hybrid Transformer arranger;
- the existing ORBIT-8 safety and chart serialization pipeline.

The rhythm model keeps the proven transient convolutional encoder/decoder and
replaces its bottleneck with the same modern hybrid sequence stack used by
Trans-1: RMSNorm, RoPE, grouped-query attention, scaled dot-product attention,
SwiGLU, and gated sequence-mixer layers.

## Training

| Item | Value |
| --- | ---: |
| Training windows | 4,052 |
| Validation windows | 593 |
| Epochs | 12 |
| Best epoch | 10 |
| Parameters | 4,143,369 |
| Warm-started tensors | 144 |

The convolutional frontend, decoder, and prediction heads were warm-started
from the v1.7.1 rhythm checkpoint. The sequence core was trained as a new
module.

## Rhythm Validation

| Metric | v1.7.1 rhythm | Trans-02 | Delta |
| --- | ---: | ---: | ---: |
| Onset F1 | 0.7411 | **0.7451** | +0.0040 |
| Event F1 | **0.3945** | 0.3502 | -0.0443 |
| Count accuracy | 0.9704 | **0.9715** | +0.0011 |
| Peak training memory | **839 MiB** | 1,553 MiB | +714 MiB |

Trans-02 slightly improves the main onset detection score and count accuracy,
but classifies note/event types less accurately and uses substantially more
memory. It should therefore remain an experimental model rather than replacing
the current rhythm model.

## B.M.S. Same-Seed Comparison

All three charts used the same source audio, BPM 200, offset 0, difficulty
12.6, and seed 20260702.

| Metric | v1.7.1 | Trans-1 | Trans-02 |
| --- | ---: | ---: | ---: |
| Events | 448 | 447 | 455 |
| Slides | 53 | 54 | 67 |
| Interaction segments | 2 | 3 | 0 |
| Sweep segments | 2 | 0 | 0 |
| Orbit repairs | 20 | 2 | 2 |
| Safety conflicts | 0 | 0 | 0 |
| Exact official-tick match | **87.71%** | 87.62% | 86.17% |
| 16th-note gaps (12 ticks) | 51 | 51 | 11 |
| 8th-note gaps (24 ticks) | 187 | 185 | 239 |

The generated Trans-02 chart is valid and hand-capacity safe, with fewer orbit
repairs than v1.7.1. Its timing plan is strongly biased toward eighth-note
spacing. That explains both the increased eighth-note slide count and the lack
of interaction/sweep phrases: under the current definitions, interaction uses
alternating 16ths, sweeps use dense 16th/24th/32nd runs, and isolated eighths
do not count as a configuration.

On this song, exact alignment with the prepared official chart is 1.54
percentage points below v1.7.1. The global validation gain therefore does not
yet translate into a per-song improvement for B.M.S.

## Verdict

Trans-02 proves that the hybrid Transformer can perform the complete
audio-to-rhythm-to-layout pipeline, and its global onset F1 is slightly better.
It is not yet a strict upgrade: event typing is weaker, memory use is higher,
and the B.M.S. result suppresses too many 16th-note phrases.

The next useful iteration is a metrical multi-task rhythm head that separately
predicts onset confidence, beat subdivision, and accent strength, with
checkpoint selection balancing onset F1 and event F1. This should retain the
better long-range onset judgment without collapsing dense musical passages
into eighth notes.
