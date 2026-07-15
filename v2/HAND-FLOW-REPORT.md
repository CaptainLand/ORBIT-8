# ORBIT-8 HandFlow Prototype

## Algorithm

HandFlow performs beam search over possible left/right-hand assignments while
the chart is being finalized. Each state tracks:

- each hand's current lane and most recent action time;
- future hold and slide reservations;
- continuous slide start/end positions and direction;
- crossed-arm posture and rapid posture reversals;
- backhand-side actions and normalized movement speed;
- optional local lane repairs.

The beam width is 256. Hold and slide occupation, same-corridor conflicts, and
more than two required hands are hard constraints. Fast movement, backhand
positions, arm crossing, and returning to a future slide start are soft costs.
This permits intentional cross-hand patterns while strongly discouraging
rapid cross/uncross oscillation.

Only ordinary taps and holds may move. Pattern notes and all slide shapes are
locked. Tap repairs may search all eight lanes; hold repairs use a larger
penalty. Timing is never changed.

## B.M.S. Example

The same calibrated 16M checkpoint, BPM, offset, difficulty, and seed were
used for both charts.

| Metric | Before HandFlow | After HandFlow |
| --- | ---: | ---: |
| Full-chart feasible hand route | No | **Yes** |
| First failure tick | 3,480 | None |
| Repositioned taps | 0 | 6 |
| Repositioned holds | 0 | 1 |
| Interaction segments | 1 | 1 |
| Sweep segments | 1 | 1 |
| Rapid posture changes | - | 1 |
| Crossed-hand entries | - | 29 |
| Final hand-route cost | - | 321.931 |
| Safety conflicts | 0 | 0 |

The six tap changes move by one lane. The single hold repair moves from lane 1
to lane 3. Event count, note timing, density, slide templates, and exact
official-tick precision remain unchanged.

## Limitations

This is a geometric prototype rather than a biomechanical simulator. Shoulder
positions are approximated, curved slide motion is reduced to directional
interpolation, and the old-chart dataset contains no touch notes. Cost weights
still need calibration against official charts and real player hand traces.
