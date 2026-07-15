from __future__ import annotations

import unittest

import torch

from trans1.model import Trans1Arranger


class Trans1ModelTests(unittest.TestCase):
    @staticmethod
    def batch(length: int = 12) -> dict[str, torch.Tensor]:
        return {
            "tick": torch.arange(length)[None] * 12,
            "event_type": torch.zeros(1, length, dtype=torch.long),
            "duration": torch.zeros(1, length, dtype=torch.long),
            "is_break": torch.zeros(1, length, dtype=torch.long),
            "is_ex": torch.zeros(1, length, dtype=torch.long),
            "simultaneous": torch.zeros(1, length, dtype=torch.long),
            "previous_delta": torch.full((1, length), 8, dtype=torch.long),
            "target_pattern": torch.zeros(1, length, dtype=torch.long),
            "level": torch.tensor([13.0]),
            "mask": torch.ones(1, length),
            "valid_length": torch.tensor([length]),
        }

    def test_forward_and_generate_interfaces_match_v1(self) -> None:
        model = Trans1Arranger().eval()
        batch = self.batch()
        output = model(batch)
        generated = model.generate(batch, refinement_steps=2)
        self.assertEqual(output["delta"].shape, (1, 12, 8))
        self.assertEqual(output["operator"].shape[-1], 12)
        self.assertEqual(generated["lane"].shape, (1, 12))
        self.assertTrue(torch.all((generated["lane"] >= 0) & (generated["lane"] < 8)))

    def test_free_running_forward_does_not_depend_on_teacher_tokens(self) -> None:
        torch.manual_seed(7)
        model = Trans1Arranger().eval()
        first = self.batch()
        second = self.batch()
        second["target_pattern"] = torch.full_like(second["target_pattern"], 2)
        second["previous_delta"] = torch.zeros_like(second["previous_delta"])
        with torch.no_grad():
            first_output = model(first, teacher_forcing_ratio=0.0)
            second_output = model(second, teacher_forcing_ratio=0.0)
        self.assertTrue(torch.equal(first_output["delta"], second_output["delta"]))
        self.assertTrue(torch.equal(first_output["pattern"], second_output["pattern"]))


if __name__ == "__main__":
    unittest.main()
