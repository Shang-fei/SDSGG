import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[1]
LOSSES_PATH = (
    ROOT
    / "maskrcnn_benchmark"
    / "modeling"
    / "roi_heads"
    / "relation_head"
    / "mtm"
    / "losses.py"
)
SPEC = importlib.util.spec_from_file_location("mtm_losses", LOSSES_PATH)
LOSSES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LOSSES)


class StructureReferenceTest(unittest.TestCase):
    def setUp(self):
        self.visual = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.projected = torch.tensor(
            [[1.0, 0.0], [0.5, 3.0**0.5 / 2.0]]
        )
        self.text = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])

    def test_selects_projected_or_text_structure_reference(self):
        projected_loss, _ = LOSSES.structure_losses(
            self.visual,
            self.projected,
            self.text,
            reference="projected",
        )
        text_loss, _ = LOSSES.structure_losses(
            self.visual,
            self.projected,
            self.text,
            reference="text",
        )

        self.assertAlmostEqual(projected_loss.item(), 0.5)
        self.assertAlmostEqual(text_loss.item(), 1.0)

    def test_rejects_unknown_structure_reference(self):
        with self.assertRaisesRegex(ValueError, "STRUCTURE_REFERENCE"):
            LOSSES.structure_losses(
                self.visual,
                self.projected,
                self.text,
                reference="unknown",
            )

    def test_default_preserves_projected_reference(self):
        defaults = (
            ROOT / "maskrcnn_benchmark" / "config" / "defaults.py"
        ).read_text()
        self.assertIn(
            'MTM.LOSS.STRUCTURE_REFERENCE = "projected"',
            defaults,
        )


if __name__ == "__main__":
    unittest.main()
