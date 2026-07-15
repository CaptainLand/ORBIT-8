from __future__ import annotations

import unittest

import torch

from maimai_ai.arranger import OfficialPatternArranger
from maimai_ai.patterns import (
    MIRROR_MODES,
    PATTERN_INTERACTION,
    PATTERN_JACK,
    PATTERN_SWEEP,
    detect_pattern_labels,
    mirror_event,
    mirror_lane,
    mirror_operator,
)


def taps(lanes: list[int], gap: int = 12) -> list[dict]:
    return [
        {"tick": index * gap, "lane": lane, "note_type": "tap", "branches": []}
        for index, lane in enumerate(lanes)
    ]


class MirrorTests(unittest.TestCase):
    def test_lane_maps_are_involutions(self) -> None:
        for mode in MIRROR_MODES:
            for lane in range(1, 9):
                self.assertEqual(mirror_lane(mirror_lane(lane, mode), mode), lane)

    def test_reflections_swap_directional_operators(self) -> None:
        for mode in ("horizontal", "vertical"):
            self.assertEqual(mirror_operator("<", mode), ">")
            self.assertEqual(mirror_operator("p", mode), "q")
            self.assertEqual(mirror_operator("pp", mode), "qq")
            self.assertEqual(mirror_operator("s", mode), "z")
        self.assertEqual(mirror_operator("p", "half_turn"), "p")

    def test_slide_start_path_and_end_are_mirrored(self) -> None:
        event = {
            "tick": 0,
            "lane": 1,
            "note_type": "slide",
            "branches": [{"operator": "p", "path_lanes": [3], "end_lane": 3}],
        }
        mirrored = mirror_event(event, "horizontal")
        self.assertEqual(mirrored["lane"], 8)
        self.assertEqual(mirrored["branches"][0]["operator"], "q")
        self.assertEqual(mirrored["branches"][0]["path_lanes"], [6])
        self.assertEqual(mirrored["branches"][0]["end_lane"], 6)


class PatternTests(unittest.TestCase):
    def test_detects_interaction_sweep_and_jack(self) -> None:
        self.assertEqual(detect_pattern_labels(taps([3, 4, 3, 4])), [PATTERN_INTERACTION] * 4)
        self.assertEqual(detect_pattern_labels(taps([8, 1, 2, 3])), [PATTERN_SWEEP] * 4)
        self.assertEqual(detect_pattern_labels(taps([5, 5, 5])), [PATTERN_JACK] * 3)

    def test_interaction_is_preserved_by_every_mirror(self) -> None:
        source = taps([3, 4, 3, 4])
        for mode in MIRROR_MODES:
            mirrored = [mirror_event(event, mode) for event in source]
            self.assertEqual(detect_pattern_labels(mirrored), [PATTERN_INTERACTION] * 4)

    def test_eighth_notes_are_not_patterns(self) -> None:
        self.assertEqual(detect_pattern_labels(taps([3, 4, 3, 4], gap=24)), [0] * 4)
        self.assertEqual(detect_pattern_labels(taps([5, 5, 5, 5], gap=24)), [0] * 4)
        self.assertEqual(detect_pattern_labels(taps([1, 2, 3, 4], gap=24)), [0] * 4)

    def test_interaction_and_jack_require_sixteenth_notes(self) -> None:
        self.assertEqual(detect_pattern_labels(taps([3, 4, 3, 4], gap=8)), [0] * 4)
        self.assertEqual(detect_pattern_labels(taps([5, 5, 5, 5], gap=8)), [0] * 4)

    def test_sweeps_accept_sixteenth_twenty_fourth_and_thirty_second(self) -> None:
        for gap in (12, 8, 6):
            self.assertEqual(detect_pattern_labels(taps([1, 2, 3, 4], gap=gap)), [PATTERN_SWEEP] * 4)

    def test_decoder_emits_an_interaction_run(self) -> None:
        model = OfficialPatternArranger().eval()
        with torch.no_grad():
            model.pattern_head.weight.zero_()
            model.pattern_head.bias.zero_()
            model.pattern_head.bias[PATTERN_INTERACTION] = 10.0
            model.delta_head.weight.zero_()
            model.delta_head.bias.zero_()
        length = 4
        batch = {
            "tick": torch.arange(length).view(1, length) * 12,
            "event_type": torch.zeros((1, length), dtype=torch.long),
            "duration": torch.zeros((1, length), dtype=torch.long),
            "is_break": torch.zeros((1, length), dtype=torch.long),
            "is_ex": torch.zeros((1, length), dtype=torch.long),
            "simultaneous": torch.zeros((1, length), dtype=torch.long),
            "level": torch.tensor([14.0]),
        }
        generated = model.generate(batch, first_lane=2)
        self.assertEqual(generated["lane"][0].tolist(), [2, 3, 2, 3])


if __name__ == "__main__":
    unittest.main()
