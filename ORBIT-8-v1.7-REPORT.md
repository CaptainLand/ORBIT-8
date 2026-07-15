# ORBIT-8 v1.7 Training Report

> Historical note: configuration counts in this report predate the strict
> v1.7.1 definitions. Current QA excludes all 8th-note patterns.

## Scope

v1.7 upgrades rhythm acquisition while retaining the v1.6 mirrored pattern
arranger and the v1-compatible slide catalog.

## Consensus onset material

- Songs: 348
- Official charts: 451
- Onset clusters: 189,479
- Clusters confirmed by multiple difficulties: 37,863
- Clustering tolerance: 18 ms
- Training target smoothing: +/-2 chart ticks

## Rhythm model

- Base: `maimai_rhythm/runs/orbit_v15_rhythm_grouped/best.pt`
- v1.7: `maimai_rhythm/runs/orbit_v17_rhythm_consensus/best.pt`
- Best epoch: 11
- Validation onset precision (+/-2 ticks): 66.23%
- Validation onset recall (+/-2 ticks): 84.12%
- Validation onset F1 (+/-2 ticks): 74.11%

The onset head is initialized from the previous Tap, Hold, and Slide heads. It
learns a difficulty-consensus acoustic onset target while the existing heads
continue to predict chart-specific note types and counts.

## Rhythm cleanup

- Candidate notes are ranked by local onset peaks.
- Candidate timing is restricted to 16th, 24th, and 32nd-compatible grids.
- Notes closer than a 16th are rejected during selection unless they form a
  four-note high-confidence 24th/32nd run.
- Accepted high-speed runs are forced to sweep patterns.
- A final pass removes any remaining isolated fast interval.

## Generation QA

Official comparison: `output/B.M.S. ORBIT-8 v1.7 final`

- v1.6 onset F1 at +/-30 ms: 62.02%
- v1.7 onset F1 at +/-30 ms: 63.78%
- v1.7 precision / recall: 56.46% / 73.29%
- Isolated sub-16th gaps: 0
- Maximum simultaneous actions: 2
- Invalid slides: 0

High-BPM stress test: `output/DayDream ORBIT-8 v1.7 grid`

- Events: 1,249 (previous output: 1,271)
- Previous sub-16th gaps: 49
- v1.7 sub-16th gaps: 0
- Notes removed by final emergency cleanup: 0
