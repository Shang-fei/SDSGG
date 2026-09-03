import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from visualization.rpmt_feature_analysis import (
    BalancedReservoir,
    ProjectorCapture,
    align_feature_arrays,
    export_figure,
    filter_feature_arrays,
    joint_reduce,
    positive_relation_records,
    sample_dataset_occurrences,
    structure_metrics,
)


class PositiveRelationRecordsTest(unittest.TestCase):
    def test_returns_only_positive_relations_in_test_pair_order(self):
        relation_matrix = np.array(
            [
                [0, 3, 0],
                [4, 0, 5],
                [0, 0, 0],
            ],
            dtype=np.int64,
        )

        records = positive_relation_records(relation_matrix)

        self.assertEqual(records, [(0, 1, 3, 0), (1, 0, 4, 2), (1, 2, 5, 3)])


class BalancedReservoirTest(unittest.TestCase):
    def test_caps_each_predicate_and_is_reproducible(self):
        first = BalancedReservoir(max_per_class=3, seed=2027)
        second = BalancedReservoir(max_per_class=3, seed=2027)
        records = [
            {"predicate_id": predicate, "sample_id": sample}
            for predicate in (1, 2)
            for sample in range(20)
        ]
        for record in records:
            first.add(record["predicate_id"], record)
            second.add(record["predicate_id"], record)

        self.assertEqual(first.counts_seen, {1: 20, 2: 20})
        self.assertEqual(first.records(), second.records())
        self.assertEqual(
            {key: len(value) for key, value in first.by_class().items()},
            {1: 3, 2: 3},
        )


class StructureMetricsTest(unittest.TestCase):
    def test_identity_mapping_has_zero_error_and_perfect_preservation(self):
        visual = np.array(
            [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

        metrics = structure_metrics(visual, visual, k=1)

        self.assertAlmostEqual(metrics["structure_mae"], 0.0, places=12)
        self.assertAlmostEqual(metrics["structure_spearman"], 1.0, places=12)
        self.assertAlmostEqual(metrics["knn_overlap@1"], 1.0, places=12)


class FigurePreparationTest(unittest.TestCase):
    def test_empty_predicate_selection_is_rejected(self):
        arrays = {
            "predicate_name": np.array(["on", "under"]),
            "visual": np.eye(2),
        }
        with self.assertRaisesRegex(ValueError, "SELECTED_PREDICATES"):
            filter_feature_arrays(arrays, [])

    def test_joint_pca_preserves_named_lengths_and_is_deterministic(self):
        rng = np.random.default_rng(7)
        inputs = {"visual": rng.normal(size=(5, 4)), "text": rng.normal(size=(3, 4))}

        first = joint_reduce(inputs, method="pca", seed=2027)
        second = joint_reduce(inputs, method="pca", seed=2027)

        self.assertEqual(first["visual"].shape, (5, 2))
        self.assertEqual(first["text"].shape, (3, 2))
        np.testing.assert_allclose(first["visual"], second["visual"])

    def test_export_writes_pdf_and_high_resolution_png(self):
        import matplotlib.pyplot as plt

        with TemporaryDirectory() as directory:
            figure, axis = plt.subplots()
            axis.plot([0, 1], [0, 1])
            paths = export_figure(figure, Path(directory) / "candidate")
            plt.close(figure)

            self.assertEqual({path.suffix for path in paths}, {".pdf", ".png"})
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))


class _FakeTarget:
    def __init__(self, relation):
        self.relation = np.asarray(relation)

    def get_field(self, name):
        if name != "relation":
            raise KeyError(name)
        return self.relation


class _FakeDataset:
    ind_to_predicates = ["__background__", "on", "under"]

    def __init__(self):
        self.targets = [
            _FakeTarget([[0, 1], [2, 0]]),
            _FakeTarget([[0, 1], [0, 0]]),
        ]

    def __len__(self):
        return len(self.targets)

    def get_groundtruth(self, index):
        return self.targets[index]


class DatasetSamplingTest(unittest.TestCase):
    def test_support_is_complete_but_sampling_respects_selected_predicates(self):
        support, occurrences = sample_dataset_occurrences(
            _FakeDataset(), ["under"], max_per_class=5, seed=2027
        )

        self.assertEqual(support, {"on": 2, "under": 1})
        self.assertEqual(len(occurrences), 1)
        self.assertEqual(occurrences[0]["predicate_name"], "under")
        self.assertEqual(occurrences[0]["pair_position"], 1)

    def test_ablation_arrays_are_reordered_to_full_model_sample_keys(self):
        full = {
            "sample_key": np.array(["4:0:1:2", "9:1:0:1"]),
            "projected": np.array([[1.0, 0.0], [0.0, 1.0]]),
        }
        ablation = {
            "sample_key": np.array(["9:1:0:1", "4:0:1:2"]),
            "projected": np.array([[0.2, 0.8], [0.7, 0.3]]),
        }

        ordered_full, ordered_ablation = align_feature_arrays(full, ablation)

        np.testing.assert_array_equal(ordered_full["sample_key"], ordered_ablation["sample_key"])
        np.testing.assert_allclose(
            ordered_ablation["projected"], np.array([[0.7, 0.3], [0.2, 0.8]])
        )

    def test_projector_capture_copies_input_visual_and_projected_features(self):
        capture = ProjectorCapture()
        raw = np.array([[8.0, 9.0]])
        visual = np.array([[1.0, 2.0]])
        projected = np.array([[3.0, 4.0]])

        capture(None, (raw, True), (visual, projected))

        self.assertEqual(len(capture.calls), 1)
        np.testing.assert_array_equal(capture.calls[0]["raw"], raw)
        np.testing.assert_array_equal(capture.calls[0]["visual"], visual)
        np.testing.assert_array_equal(capture.calls[0]["projected"], projected)


if __name__ == "__main__":
    unittest.main()
