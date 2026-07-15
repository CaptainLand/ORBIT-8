from __future__ import annotations

import unittest

import torch

from generate_maimai import (
    GeneratedEvent,
    break_eighth_note_orbits,
    break_long_eighth_jacks,
    break_irregular_sixteenth_runs,
    clean_irregular_fast_notes,
    enforce_hand_capacity,
    generated_pattern_summary,
    limit_jack_patterns,
    long_eighth_jack_excess,
    max_jack_share_for_bpm,
    playability_conflicts,
    slide_branch_length,
    slide_corridor_lanes,
    slide_motion_interval,
    slide_tail_lanes,
)
from maimai_ai.arranger import OPERATORS, OfficialPatternArranger
from maimai_ai.patterns import PATTERN_SWEEP


class FastRhythmTests(unittest.TestCase):
    def test_eighth_note_clockwise_orbit_is_broken(self) -> None:
        events = [
            GeneratedEvent(tick=tick, lane=lane, kind="slide", score=0.8)
            for tick, lane in zip((0, 24, 48, 72), (1, 2, 3, 4))
        ]
        adjusted, changes = break_eighth_note_orbits(events)
        self.assertEqual(changes, 1)
        self.assertEqual([event.lane for event in adjusted], [1, 2, 3, 6])

    def test_non_orbiting_eighth_notes_are_unchanged(self) -> None:
        events = [
            GeneratedEvent(tick=tick, lane=lane, kind="tap", score=0.8)
            for tick, lane in zip((0, 24, 48, 72), (1, 3, 2, 5))
        ]
        adjusted, changes = break_eighth_note_orbits(events)
        self.assertEqual(changes, 0)
        self.assertEqual([event.lane for event in adjusted], [1, 3, 2, 5])

    def test_isolated_fast_gap_removes_weaker_tick(self) -> None:
        events = [
            GeneratedEvent(tick=0, lane=1, kind="tap", score=0.9),
            GeneratedEvent(tick=6, lane=1, kind="tap", score=0.2),
            GeneratedEvent(tick=24, lane=1, kind="tap", score=0.8),
        ]
        self.assertEqual([event.tick for event in clean_irregular_fast_notes(events)], [0, 24])

    def test_four_note_32nd_run_is_preserved(self) -> None:
        events = [GeneratedEvent(tick=tick, lane=1, kind="tap", score=0.8) for tick in (0, 6, 12, 18)]
        self.assertEqual([event.tick for event in clean_irregular_fast_notes(events)], [0, 6, 12, 18])

    def test_fast_run_is_forced_to_sweep(self) -> None:
        logits = torch.zeros((1, 4, 4))
        batch = {
            "tick": torch.tensor([[0, 6, 12, 18]]),
            "event_type": torch.zeros((1, 4), dtype=torch.long),
            "simultaneous": torch.zeros((1, 4), dtype=torch.long),
            "valid_length": torch.tensor([4]),
        }
        tokens = OfficialPatternArranger._segment_pattern_tokens(batch, logits)
        self.assertEqual(tokens[0].tolist(), [PATTERN_SWEEP] * 4)

    def test_zero_pattern_heat_disables_forced_sweep(self) -> None:
        logits = torch.zeros((1, 4, 4))
        batch = {
            "tick": torch.tensor([[0, 6, 12, 18]]),
            "event_type": torch.zeros((1, 4), dtype=torch.long),
            "simultaneous": torch.zeros((1, 4), dtype=torch.long),
            "valid_length": torch.tensor([4]),
        }
        tokens = OfficialPatternArranger._segment_pattern_tokens(
            batch, logits, pattern_heat=(0.0, 0.0, 0.0)
        )
        self.assertEqual(tokens[0].tolist(), [0, 0, 0, 0])

    def test_default_heat_matches_explicit_one_hundred_percent(self) -> None:
        logits = torch.zeros((1, 4, 4))
        batch = {
            "tick": torch.tensor([[0, 12, 24, 36]]),
            "event_type": torch.zeros((1, 4), dtype=torch.long),
            "simultaneous": torch.zeros((1, 4), dtype=torch.long),
            "valid_length": torch.tensor([4]),
        }
        torch.manual_seed(17)
        default = OfficialPatternArranger._segment_pattern_tokens(batch, logits)
        torch.manual_seed(17)
        explicit = OfficialPatternArranger._segment_pattern_tokens(
            batch, logits, pattern_heat=(1.0, 1.0, 1.0)
        )
        self.assertTrue(torch.equal(default, explicit))

    def test_final_jack_share_is_hard_limited(self) -> None:
        events = [
            *[
                GeneratedEvent(tick=tick, lane=lane, kind="tap", score=0.9)
                for tick, lane in zip((0, 12, 24, 36), (3, 4, 3, 4))
            ],
            *[
                GeneratedEvent(tick=tick, lane=5, kind="tap", score=0.5)
                for tick in (72, 84, 96)
            ],
        ]
        limited, changes = limit_jack_patterns(events, remove_excess=False)
        _, segments, share = generated_pattern_summary(limited)
        self.assertGreater(changes, 0)
        self.assertEqual(segments.get("jack", 0), 0)
        self.assertLessEqual(share, 0.02)

    def test_jacks_are_disabled_above_200_bpm(self) -> None:
        self.assertEqual(max_jack_share_for_bpm(200.01, 2.0), 0.0)
        self.assertGreater(max_jack_share_for_bpm(200.0, 1.0), 0.0)

    def test_seventh_consecutive_eighth_moves_to_a_neighbor(self) -> None:
        events = [
            GeneratedEvent(tick=index * 24, lane=4, kind="tap", score=1.0)
            for index in range(10)
        ]
        arranged, changes = break_long_eighth_jacks(events)
        self.assertGreater(changes, 0)
        self.assertEqual(long_eighth_jack_excess(arranged), 0)
        self.assertNotEqual(next(event for event in arranged if event.tick == 6 * 24).lane, 4)

    def test_regular_sixteenth_sweep_is_preserved(self) -> None:
        events = [
            GeneratedEvent(tick=index * 12, lane=index + 1, kind="tap", score=1.0)
            for index in range(4)
        ]
        arranged, removed = break_irregular_sixteenth_runs(events, bpm=220.0)
        self.assertEqual(removed, 0)
        self.assertEqual(len(arranged), 4)

    def test_irregular_sixteenth_flying_hands_are_split(self) -> None:
        events = [
            GeneratedEvent(tick=tick, lane=lane, kind="tap", score=score)
            for tick, lane, score in zip((0, 12, 24, 36), (1, 5, 2, 7), (0.9, 0.2, 0.8, 0.7))
        ]
        arranged, removed = break_irregular_sixteenth_runs(events, bpm=220.0)
        self.assertGreater(removed, 0)
        self.assertLess(len(arranged), 4)

class OccupationTests(unittest.TestCase):
    @staticmethod
    def ring_catalog() -> list[dict]:
        return [{
            "branch_count": 1,
            "operators": [">"],
            "relative_paths": [[4]],
            "durations": [{"raw": "4:1", "duration_ticks": 48, "wait_ticks": None}],
            "levels": {},
            "count_12_15": 1,
        }]

    def test_edge_slide_blocks_taps_on_its_path_until_tail(self) -> None:
        slide = GeneratedEvent(
            tick=0,
            lane=1,
            kind="slide",
            score=1.0,
            operator_id=OPERATORS.index(">"),
            endpoint=4,
        )
        tap = GeneratedEvent(tick=60, lane=3, kind="tap", score=0.8)
        accepted = enforce_hand_capacity([slide, tap], self.ring_catalog(), 14.0)
        accepted_slide = next(event for event in accepted if event.kind == "slide")
        accepted_tap = next(event for event in accepted if event.kind == "tap")
        interval = slide_motion_interval(accepted_slide)
        self.assertEqual(interval[0], 48)
        self.assertGreaterEqual(interval[1], 96)
        self.assertTrue(accepted_slide.slide_was_slowed)
        self.assertNotEqual(accepted_tap.lane, 3)
        self.assertEqual(playability_conflicts(accepted)["slide_path_tap_conflicts"], 0)

    def test_edge_slide_direction_uses_upper_and_lower_screen_halves(self) -> None:
        upper = GeneratedEvent(tick=0, lane=7, kind="slide", score=1.0)
        upper.slide_template = {
            "operators": [">"], "relative_paths": [[3]],
            "durations": [{"raw": "4:1", "duration_ticks": 48, "wait_ticks": None}],
        }
        lower = GeneratedEvent(tick=0, lane=3, kind="slide", score=1.0)
        lower.slide_template = {
            "operators": [">"], "relative_paths": [[3]],
            "durations": [{"raw": "4:1", "duration_ticks": 48, "wait_ticks": None}],
        }
        self.assertEqual(slide_corridor_lanes(upper)[0], {7, 8, 1, 2})
        self.assertEqual(slide_corridor_lanes(lower)[0], {3, 2, 1, 8, 7, 6})

    def test_grand_p_slide_moves_active_taps_to_opposite_side(self) -> None:
        catalog = [{
            "branch_count": 1,
            "operators": ["pp"],
            "relative_paths": [[3]],
            "durations": [{"raw": "192:108", "duration_ticks": 108, "wait_ticks": 48}],
            "levels": {},
            "count_12_15": 1,
        }]
        slide = GeneratedEvent(
            tick=0, lane=1, kind="slide", score=1.0,
            operator_id=OPERATORS.index("pp"), endpoint=4,
        )
        taps = [
            GeneratedEvent(tick=tick, lane=lane, kind="tap", score=0.8)
            for tick, lane in ((48, 2), (72, 3), (96, 2))
        ]
        accepted = enforce_hand_capacity([slide, *taps], catalog, 12.6)
        active_taps = [event for event in accepted if event.kind == "tap"]
        self.assertEqual(len(active_taps), 3)
        self.assertTrue(all(event.lane in {5, 6, 7, 8} for event in active_taps))
        self.assertEqual(playability_conflicts(accepted)["slide_path_tap_conflicts"], 0)

    def test_wifi_slide_occupies_two_hands_and_has_three_tails(self) -> None:
        catalog = [{
            "branch_count": 1,
            "operators": ["w"],
            "relative_paths": [[3]],
            "durations": [{"raw": "4:1", "duration_ticks": 48, "wait_ticks": 48}],
            "levels": {},
            "count_12_15": 1,
        }]
        wifi = GeneratedEvent(
            tick=0, lane=1, kind="slide", score=1.0,
            operator_id=OPERATORS.index("w"), endpoint=4,
        )
        head_tap = GeneratedEvent(tick=0, lane=6, kind="tap", score=0.8)
        moving_tap = GeneratedEvent(tick=60, lane=7, kind="tap", score=0.8)
        accepted = enforce_hand_capacity([wifi, head_tap, moving_tap], catalog, 14.0)
        accepted_wifi = next(event for event in accepted if event.kind == "slide")
        self.assertEqual(slide_tail_lanes(accepted_wifi), {3, 4, 5})
        self.assertFalse(any(event.kind == "tap" for event in accepted))
        self.assertEqual(playability_conflicts(accepted)["max_hand_demand"], 2)

    def test_wifi_blocks_all_three_tail_lanes_during_clearance(self) -> None:
        catalog = [{
            "branch_count": 1,
            "operators": ["w"],
            "relative_paths": [[3]],
            "durations": [{"raw": "4:1", "duration_ticks": 48, "wait_ticks": 48}],
            "levels": {},
            "count_12_15": 1,
        }]
        wifi = GeneratedEvent(
            tick=0, lane=1, kind="slide", score=1.0,
            operator_id=OPERATORS.index("w"), endpoint=4,
        )
        tail_taps = [
            GeneratedEvent(tick=108, lane=lane, kind="tap", score=0.8)
            for lane in (3, 4, 5)
        ]
        accepted = enforce_hand_capacity([wifi, *tail_taps], catalog, 14.0)
        accepted_tail_lanes = {
            event.lane for event in accepted if event.kind == "tap" and event.tick == 108
        }
        self.assertFalse(accepted_tail_lanes & {3, 4, 5})

    def test_long_objects_are_removed_from_sixteenth_streams(self) -> None:
        hold = GeneratedEvent(tick=0, lane=1, kind="hold", score=1.0, duration=96)
        taps_in_stream = [
            GeneratedEvent(tick=tick, lane=lane, kind="tap", score=0.8)
            for tick, lane in ((12, 3), (24, 4), (36, 3))
        ]
        accepted = enforce_hand_capacity([hold, *taps_in_stream], [], 14.0)
        self.assertFalse(any(event.kind == "hold" for event in accepted))
        self.assertEqual(playability_conflicts(accepted)["long_object_sixteenth_conflicts"], 0)

    def test_one_hand_can_play_eighths_during_a_hold(self) -> None:
        hold = GeneratedEvent(tick=0, lane=1, kind="hold", score=1.0, duration=72)
        taps = [
            GeneratedEvent(tick=tick, lane=3, kind="tap", score=0.8)
            for tick in (24, 48)
        ]
        accepted = enforce_hand_capacity([hold, *taps], [], 14.0)
        self.assertEqual(len(accepted), 3)
        self.assertEqual(playability_conflicts(accepted)["long_object_sixteenth_conflicts"], 0)

    def test_long_object_tail_occupies_a_hand_until_one_sixteenth_later(self) -> None:
        hold_left = GeneratedEvent(tick=0, lane=1, kind="hold", score=1.0, duration=48)
        hold_right = GeneratedEvent(tick=0, lane=5, kind="hold", score=0.95, duration=48)
        tap_at_tail = GeneratedEvent(tick=48, lane=3, kind="tap", score=0.9)
        tap_at_release = GeneratedEvent(tick=60, lane=5, kind="tap", score=0.7)
        accepted = enforce_hand_capacity(
            [hold_left, hold_right, tap_at_tail, tap_at_release], [], 14.0
        )
        accepted_ticks = {event.tick for event in accepted if event.kind == "tap"}
        self.assertNotIn(48, accepted_ticks)
        self.assertIn(60, accepted_ticks)
        self.assertLessEqual(playability_conflicts(accepted)["max_hand_demand"], 2)

    def test_slide_tail_collision_requires_sixth_note_clearance(self) -> None:
        catalog = [{
            "branch_count": 1,
            "operators": ["-"],
            "relative_paths": [[1]],
            "durations": [{"raw": "4:1", "duration_ticks": 48, "wait_ticks": None}],
            "levels": {},
            "count_12_15": 1,
        }]
        first = GeneratedEvent(tick=0, lane=1, kind="slide", score=1.0, operator_id=0)
        too_close = GeneratedEvent(tick=24, lane=1, kind="slide", score=0.9, operator_id=0)
        accepted_close = enforce_hand_capacity([first, too_close], catalog, 14.0)
        self.assertEqual(sum(event.kind == "slide" for event in accepted_close), 1)
        self.assertEqual(playability_conflicts(accepted_close)["slide_tail_clearance_conflicts"], 0)

        first = GeneratedEvent(tick=0, lane=1, kind="slide", score=1.0, operator_id=0)
        clear = GeneratedEvent(tick=32, lane=1, kind="slide", score=0.9, operator_id=0)
        accepted_clear = enforce_hand_capacity([first, clear], catalog, 14.0)
        self.assertEqual(sum(event.kind == "slide" for event in accepted_clear), 2)
        self.assertEqual(playability_conflicts(accepted_clear)["slide_tail_clearance_conflicts"], 0)

    def test_shape_geometry_distinguishes_long_and_short_slides(self) -> None:
        short = slide_branch_length(1, "-", [2])
        long_edge = slide_branch_length(1, ">", [5])
        self.assertGreater(long_edge, short * 3)


if __name__ == "__main__":
    unittest.main()
