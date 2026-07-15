# ORBIT-8 v1.6 Training Report

> Historical note: configuration counts in this report used the old broad
> equal-spacing detector. Use the strict v1.7.1 definitions for current QA.

## Scope

v1.6 keeps the v1.5 rhythm and automatic timing models and upgrades the lane
arranger. Slide templates use the compatibility-tested v1 catalog.

## Training data

- Source charts: 451 charts at levels 12.0-15.9
- Train/validation/test charts: 356 / 51 / 44
- Original training windows: 4,052
- Effective mirrored training windows: 16,208
- Mirror modes: normal, horizontal, vertical, half turn
- Touch policy: remove Touch notes while retaining the rest of each chart

Detected official-chart material:

- Interaction: 1,539 segments / 9,131 events
- Sweep: 1,957 segments / 14,961 events
- Jack: 1,111 segments / 5,452 events
- Mirror consistency failures: 0

## Model

- Base checkpoint: `maimai_arranger/runs/orbit_v15_arranger/best.pt`
- v1.6 checkpoint: `maimai_arranger/runs/orbit_v16_arranger/best.pt`
- Best epoch: 4
- Best validation loss: 3.107360
- Interaction recall: 64.86%
- Sweep recall: 63.73%
- Jack recall: 25.43%

The decoder treats predicted patterns as complete rhythmic phrases. Interaction
phrases alternate A-B-A-B, sweep phrases preserve direction, and jack phrases
remain on one lane. Pattern probabilities are calibrated back from the weighted
training loss before seeded sampling.

## Generation QA

Sample: `output/B.M.S. ORBIT-8 v1.6`

- Events: 438
- Final detected interaction events: 29
- Final configuration segments: 15
- Maximum simultaneous actions: 2
- Peak non-sweep heads per second: 15
- Invalid slides: 0
- Notes inside a same-lane hold: 0
