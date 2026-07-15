# ORBIT-8 v1.5 Training Report

Created by SeaLandX.

## Dataset

- Difficulty range: 12.0-15.0
- Parsed charts: 451
- Prepared windows: 5,141
- Rhythm train/validation windows: 4,052 / 593
- Arranger train/validation windows: 3,497 / 451
- TOUCH and TOUCH HOLD: removed from rhythm training
- Charts originally containing TOUCH: excluded from arranger training
- Incompatible high-resolution or special DX charts: skipped

## Models

- Rhythm: `maimai_rhythm/runs/orbit_v15_rhythm_grouped/best.pt`
- Arranger: `maimai_arranger/runs/orbit_v15_arranger/best.pt`
- Prepared charts: `maimai_finale_dataset/prepared_v2`
- Audio features: `maimai_finale_dataset/prepared_audio_orbit_v15`

## Evaluation

- Rhythm best epoch: 6
- Validation event F1: 0.4319
- Held-out test exact F1: 0.3972
- Held-out test F1 at +/-4 ticks: 0.4038
- v1 held-out test exact F1: approximately 0.3280
- Arranger best epoch: 12
- Arranger validation loss: 2.8129
- Arranger validation lane accuracy: 0.4397
- Arranger validation operator accuracy: 0.4835

## Acceptance Chart

`output/B.M.S. ORBIT-8 v1.5`

- Events: 441
- Maximum simultaneous heads: 2
- Maximum active hands: 2
- Peak one-second note starts: 15
- Slide-path note collisions: 0
- Slide-tail collisions: 0
- Same-lane notes inside HOLD: 0
