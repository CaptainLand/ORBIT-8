import unittest

import numpy as np
import torch

from generate_maimai import rhythm_plan_events
from v2.rhythm_model import OrbitV2RhythmModel
from v2.rhythm_model_16m import OrbitV2RhythmModel16M
from v2.targets import gap_class, metrical_targets


class OrbitV2Tests(unittest.TestCase):
    def test_16m_model_parameter_budget_and_interface(self):
        model = OrbitV2RhythmModel16M().eval()
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.assertGreaterEqual(parameter_count, 15_500_000)
        self.assertLessEqual(parameter_count, 16_500_000)
        with torch.no_grad():
            output = model(torch.randn(1, 132, 64), torch.randn(1, 5))
        self.assertEqual(output["event"].shape, (1, 5, 64))
        self.assertEqual(output["subdivision"].shape, (1, 8, 64))

    def test_model_outputs_hierarchical_heads(self):
        model = OrbitV2RhythmModel().eval()
        with torch.no_grad():
            output = model(torch.randn(1, 132, 64), torch.randn(1, 5))
        self.assertEqual(output["event"].shape, (1, 5, 64))
        self.assertEqual(output["subdivision"].shape, (1, 8, 64))
        self.assertEqual(output["accent"].shape, (1, 1, 64))

    def test_gap_classes_match_maimai_subdivisions(self):
        self.assertEqual(gap_class(24), 2)
        self.assertEqual(gap_class(16), 3)
        self.assertEqual(gap_class(12), 4)
        self.assertEqual(gap_class(8), 5)
        self.assertEqual(gap_class(6), 6)

    def test_metrical_targets_label_sixteenth_run_and_accent(self):
        event = torch.zeros(1, 5, 64)
        count = torch.zeros(1, 64, dtype=torch.long)
        for tick in (12, 24, 36, 48):
            event[0, 0, tick] = 1
            count[0, tick] = 1
        count[0, 24] = 2
        valid = torch.ones(1, 64, dtype=torch.bool)
        subdivision, accent, onset = metrical_targets(event, count, valid)
        self.assertTrue(torch.all(subdivision[0, torch.tensor([12, 24, 36, 48])] == 4))
        self.assertEqual(float(accent[0, 24]), 1.0)
        self.assertEqual(int(onset.sum()), 4)

    def test_legacy_decoder_interface_remains_supported(self):
        total_ticks = 192
        events = np.full((5, total_ticks), 0.1, dtype=np.float32)
        counts = np.zeros((3, total_ticks), dtype=np.float32)
        counts[0] = 0.8
        counts[1] = 0.2
        onsets = np.zeros(total_ticks, dtype=np.float32)
        onsets[::24] = 1.0
        controls = np.asarray([12.0, 8.0, 0.0, 0.0, 0.0], dtype=np.float32)
        generated = rhythm_plan_events(events, counts, onsets, controls, total_ticks)
        self.assertTrue(generated)


if __name__ == "__main__":
    unittest.main()
