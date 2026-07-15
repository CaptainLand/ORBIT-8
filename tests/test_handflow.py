import unittest

from generate_maimai import GeneratedEvent
from maimai_ai.patterns import PATTERN_SWEEP
from v2.handflow import HandState, _candidate_lanes, _move_hand, optimize_handflow, search_handflow


class HandFlowTests(unittest.TestCase):
    def test_simple_alternating_stream_has_feasible_assignment(self):
        events = [
            GeneratedEvent(tick=index * 24, lane=3 + index % 2, kind="tap", score=1.0)
            for index in range(8)
        ]
        state, failure = search_handflow(events, allow_repair=False)
        self.assertIsNotNone(state)
        self.assertIsNone(failure)
        self.assertLessEqual(state.max_speed, 3.5)

    def test_tap_on_occupied_hold_lane_is_repositioned(self):
        events = [
            GeneratedEvent(tick=0, lane=4, kind="hold", score=1.0, duration=96),
            GeneratedEvent(tick=24, lane=4, kind="tap", score=1.0),
        ]
        baseline, failure = search_handflow(events, allow_repair=False)
        self.assertIsNone(baseline)
        self.assertEqual(failure, 24)
        optimized, report = optimize_handflow(events)
        self.assertTrue(report["optimized"]["feasible"])
        self.assertTrue(report["applied"])
        self.assertNotEqual(optimized[1].lane, 4)

    def test_only_non_pattern_slide_can_rotate_as_last_resort(self):
        slide = GeneratedEvent(tick=0, lane=2, kind="slide", score=1.0)
        self.assertEqual(len(_candidate_lanes(slide, allow_repair=True)), 8)
        slide.pattern_type = 1
        self.assertEqual(_candidate_lanes(slide, allow_repair=True), (2,))

    def test_single_hand_cannot_jump_four_lanes_on_a_sixteenth(self):
        hand = HandState(lane=1, last_tick=0)
        self.assertIsNone(_move_hand(hand, lane=5, tick=12, side="L", reservation=None))

    def test_regular_sweep_stays_on_one_hand(self):
        events = [
            GeneratedEvent(
                tick=index * 12,
                lane=index + 1,
                kind="tap",
                score=1.0,
                pattern_type=PATTERN_SWEEP,
            )
            for index in range(4)
        ]
        state, failure = search_handflow(events, allow_repair=False)
        self.assertIsNone(failure)
        self.assertIsNotNone(state)
        self.assertEqual(len({side for _, side in state.assignments}), 1)

    def test_unrepairable_plain_tap_is_dropped(self):
        events = [
            GeneratedEvent(tick=0, lane=4, kind="hold", score=1.0, duration=96),
            GeneratedEvent(tick=24, lane=5, kind="tap", score=0.1, pattern_type=1),
            GeneratedEvent(tick=24, lane=5, kind="tap", score=0.2),
        ]
        optimized, report = optimize_handflow(events)
        self.assertTrue(report["final_assignment"]["feasible"])
        self.assertEqual(len(report["dropped"]), 1)
        self.assertEqual(len(optimized), 2)


if __name__ == "__main__":
    unittest.main()
