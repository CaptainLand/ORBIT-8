# ORBIT-8 v1.7.1 Fine-tuning Report

## Configuration definitions

- Interaction: alternating lanes at 16th-note spacing (12 ticks)
- Jack: one lane at 16th-note spacing (12 ticks)
- Sweep: continuous lane movement at 16th, 24th, or 32nd spacing (12/8/6 ticks)
- 8th notes (24 ticks): never classified as a configuration
- Other equal spacing, including 16-tick spacing: not classified

## Corrected training material

- Source charts: 451
- Mirrored training windows: 16,208
- Interaction: 1,183 segments / 7,129 events
- Sweep: 991 segments / 10,138 events
- Jack: 109 segments / 1,097 events
- Mirror consistency failures: 0

The corrected material contains about 4.8% jack segments. Training lowers the
jack loss weight, generation caps sampled jack probability at 2%, and final
chart QA breaks any accidental jack segments beyond the same 2% limit.

## Fine-tuning

- Base arranger: `maimai_arranger/runs/orbit_v16_arranger/best.pt`
- Fine-tuned arranger: `maimai_arranger/runs/orbit_v171_arranger/best.pt`
- Best epoch: 2
- Validation interaction recall: 79.58%
- Validation sweep recall: 68.20%
- Predicted jack share: 0%
- Validation lane accuracy: 45.16%

## Generation QA

Hand occupation rules:

- Holds occupy one hand from head to tail.
- Slides occupy one hand from movement start to movement end; the pre-slide
  wait does not count as continuous occupation.
- A long-object tail keeps its hand occupied until one 16th note after the
  nominal end, preventing a third simultaneous action on the tail beat.
- Slide tails sharing an endpoint require at least 32 ticks (a sixth-note
  subdivision); an 8th-note gap of 24 ticks is rejected.
- Holds and slides cannot overlap a continuous 16th-note stream.
- Taps cannot appear on any lane traversed by a moving slide, including edge
  ring slides. The path is released only after the slide tail.
- One-hand 8th notes remain legal while one hold or slide occupies the other hand.

Slide speed normalization computes geometric path length separately for
straight, center, corner, fan, arc, edge-ring, S, and Z shapes. Per-shape speed
limits are learned from the v1 official slide catalog at the 75th percentile.
Long paths receive an additional slowdown factor. Existing durations are only
extended and are never shortened.

### B.M.S. 12.6 final-safe

- Events: 453
- Interaction: 5 segments / 45 events
- Sweep: 0
- Jack: 0
- Final jack share: 0%
- Jack runs repositioned before safety QA: 9
- Notes removed for jack limiting: 0
- Slide-path Tap conflicts: 0
- Hold/Slide versus 16th-stream conflicts: 0
- Slide-tail clearance conflicts: 0
- Maximum hand demand: 2
- Slides slowed from learned geometry limits: 24 / 47
- Maximum normalized slide speed: 0.07699
- Maximum simultaneous actions: 2
- Invalid slides: 0
- Same-lane notes inside holds: 0

### DayDream 15 hand-safe

- Events: 1,272
- Configuration segments: 0
- Jack share: 0%
- Rhythm consists mainly of 8th and 16-tick spacing; these are correctly left
  unclassified under the strict definitions.
- Maximum simultaneous actions: 2
- Slide-path Tap conflicts: 0
- Hold/Slide versus 16th-stream conflicts: 0
- Invalid slides: 0
- Same-lane notes inside holds: 0
