import unittest

import torch

from trans02.rhythm_model import Trans02RhythmModel


class Trans02RhythmModelTests(unittest.TestCase):
    def test_forward_preserves_rhythm_plan_interface(self):
        model = Trans02RhythmModel().eval()
        features = torch.randn(1, 132, 64)
        controls = torch.randn(1, 5)

        with torch.no_grad():
            outputs = model(features, controls)

        self.assertEqual(outputs["event"].shape, (1, 5, 64))
        self.assertEqual(outputs["count"].shape, (1, 3, 64))
        self.assertEqual(outputs["onset"].shape, (1, 1, 64))


if __name__ == "__main__":
    unittest.main()
