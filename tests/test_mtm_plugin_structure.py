import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MTM_PACKAGE = (
    ROOT
    / "maskrcnn_benchmark"
    / "modeling"
    / "roi_heads"
    / "relation_head"
    / "mtm"
)


class MTMPluginStructureTest(unittest.TestCase):
    def test_plugin_package_exposes_only_the_public_integration_api(self):
        expected_files = {
            "__init__.py",
            "plugin.py",
            "predicate.py",
            "text_teacher.py",
            "projector.py",
            "ship.py",
            "losses.py",
            "diagnostics.py",
        }
        self.assertTrue(MTM_PACKAGE.is_dir())
        self.assertTrue(expected_files.issubset({path.name for path in MTM_PACKAGE.iterdir()}))

        module = ast.parse((MTM_PACKAGE / "__init__.py").read_text())
        exported = set()
        for node in module.body:
            if isinstance(node, ast.ImportFrom):
                exported.update(alias.name for alias in node.names)
        self.assertEqual(
            exported,
            {"MTMOutput", "MTMPredicateSpec", "build_mtm_plugin"},
        )

    def test_defaults_use_grouped_mtm_configuration(self):
        defaults = (ROOT / "maskrcnn_benchmark" / "config" / "defaults.py").read_text()
        required = (
            "MTM.PROJECTOR.INPUT_DIM",
            "MTM.TEXT_TEACHER.SVD_ENABLED",
            "MTM.LOSS.ALIGNMENT_WEIGHT",
            'MTM.LOSS.STRUCTURE_DISTANCE = "l1"',
            "MTM.SHIP.PSEUDO_NOVEL_RATIO",
            "MTM.SHIP.TEXT_ADAPTER.ENABLED",
            "MTM.SHIP.BASE_REPLAY.ENABLED",
            "MTM.SHIP.OPTIMIZER.LR",
            "MTM.INFERENCE.SCORE_WEIGHT",
            "MTM.DEBUG.INTERVAL_STEPS",
        )
        for name in required:
            self.assertIn(name, defaults)

        removed = (
            "MTM.INPUT_DIM",
            "MTM.TEXT_SVD_ENABLED",
            "MTM.ALIGN_WEIGHT",
            "MTM.SHIP_PSEUDO_RATIO",
            "MTM.INFERENCE_WEIGHT",
            "MTM.DEBUG_INTERVAL",
        )
        for name in removed:
            self.assertNotIn(name, defaults)

    def test_predictors_depend_only_on_the_plugin_api(self):
        predictors = (
            ROOT
            / "maskrcnn_benchmark"
            / "modeling"
            / "roi_heads"
            / "relation_head"
            / "roi_relation_predictors.py"
        ).read_text()
        self.assertIn("from .mtm import MTMPredicateSpec, build_mtm_plugin", predictors)
        self.assertNotIn("from .mtm_ship import MTMShipBranch", predictors)
        self.assertNotIn("MTMShipBranch(", predictors)

    def test_ship_optimizer_does_not_match_parameter_name_strings(self):
        solver = (ROOT / "maskrcnn_benchmark" / "solver" / "build.py").read_text()
        self.assertNotIn('"mtm_branch.generator"', solver)
        self.assertIn("ship_parameters", solver)
        self.assertIn("validate_optimizer_parameters", solver)

        training = (ROOT / "tools" / "relation_train_net.py").read_text()
        self.assertIn('"mtm_ship_optimizer"', training)
        self.assertNotIn('"ship_optimizer"', training)

    def test_loss_keys_remain_stable(self):
        expected = {
            "loss_mtm_align",
            "loss_mtm_visual_structure",
            "loss_mtm_text_structure",
            "loss_mtm_ship_recon",
            "loss_mtm_ship_kl",
        }
        source = (MTM_PACKAGE / "losses.py").read_text()
        for key in expected:
            self.assertIn(key, source)

    def test_structure_distance_supports_l1_and_l2(self):
        source = (MTM_PACKAGE / "losses.py").read_text()
        self.assertIn('distance == "l1"', source)
        self.assertIn('distance == "l2"', source)
        self.assertIn("difference.pow(2).mean()", source)


if __name__ == "__main__":
    unittest.main()
