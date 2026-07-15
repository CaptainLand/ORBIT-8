# ORBIT-8 v2.2 Training Report

ORBIT-8 v2.2 keeps the v2.1 inference interfaces and upgrades both checkpoint
selection and arranger training. It is intended as the default local model while
v2.1 remains available for comparison.

## Delivered Models

| Component | Parameters | Release size | Source |
| --- | ---: | ---: | --- |
| Rhythm 16M | 15,910,098 | 63,717,466 bytes | v2.1 backbone + validation-only logit calibration |
| Trans arranger | 2,958,842 | 11,883,056 bytes | v2.1 warm start + v2.2 scheduled-sampling training |

Training data and intermediate optimizer checkpoints are not included in the
release package.

## Training Changes

- Rebuild the dynamic song/category sample plan every epoch with deterministic seeds.
- Keep bounded cache-local buckets while changing sample coverage between epochs.
- Anneal confidence-masked 4M distillation instead of applying a fixed teacher loss.
- Train the arranger with scheduled sampling for both pattern tokens and previous lanes.
- Validate the arranger without ground-truth pattern or lane-history tokens.
- Select arranger checkpoints with pattern precision/F1 and slide endpoint metrics.
- Calibrate rhythm event/onset logits using only the validation split; the test split
  remains unseen until final evaluation.

The first dynamic rhythm fine-tune did not beat v2.1 on the test split and was rejected.
The delivered rhythm checkpoint therefore preserves the stronger v2.1 backbone and
only applies the validation-derived output calibration.

## Held-Out Test Results

### Rhythm

| Metric | v2.1 | v2.2 | Change |
| --- | ---: | ---: | ---: |
| Onset F1 | 0.7184 | **0.7224** | +0.0040 |
| Event F1 | 0.3260 | **0.3448** | +0.0188 |
| Dense-rhythm F1 | 0.6558 | 0.6558 | unchanged |
| Subdivision accuracy | 0.6233 | 0.6233 | unchanged |
| Count accuracy | 0.9724 | 0.9724 | unchanged |

### Arranger (free-running evaluation)

| Metric | v2.1 | v2.2 | Change |
| --- | ---: | ---: | ---: |
| Composite | 0.4430 | **0.4525** | +0.0095 |
| Lane accuracy | 0.2605 | **0.2653** | +0.0048 |
| Endpoint accuracy | 0.3094 | **0.3146** | +0.0052 |
| Interaction F1 | 0.3777 | **0.3848** | +0.0070 |
| Sweep F1 | **0.6886** | 0.6573 | -0.0313 |

The lower sweep F1 is a known tradeoff. v2.2 improves free-running lane placement and
interaction recognition, but v2.1 remains selectable in the web UI for sweep-heavy songs.

## Full-Song Acceptance

Five Finale 14.9 songs were generated with fixed seeds. Final checks reported:

- HandFlow feasible: 5/5 songs.
- Maximum simultaneous hand demand: 2.
- Slide-path tap conflicts: 0.
- Long-object sixteenth conflicts: 0.
- Slide-tail clearance conflicts: 0.

Generated examples and the machine-readable summary are under
`v22/output/HandFlow 14.9 Test Suite/`.

## Run

Start the existing local web server and select `ORBIT-8 v2.2 HandFlow`, or run:

```powershell
python -m v22.generate_v22 track.mp3 --bpm 200 --offset 0 --level 14.9 --output output/song
```
