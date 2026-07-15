import unittest

import numpy as np

from maimai_ai.dynamic_audio_dataset import DynamicAudioChartDataset
from maimai_ai.dynamic_arranger_dataset import DynamicOraclePlanDataset


class DynamicDatasetTests(unittest.TestCase):
    @staticmethod
    def _resampling_fixture(dataset_type):
        dataset = dataset_type.__new__(dataset_type)
        dataset.seed = 1234
        dataset.epoch = 0
        dataset.samples_per_epoch = 48
        dataset.cache_bucket_size = 12
        dataset.category_names = ("regular", "dense")
        dataset.category_probabilities = np.asarray([0.6, 0.4], dtype=np.float64)
        dataset.rows_by_category = {
            "regular": [
                {"song_id": f"song_{index}", "chart_id": f"chart_{index}"}
                for index in range(8)
            ],
            "dense": [
                {"song_id": f"dense_{index}", "chart_id": f"dense_chart_{index}"}
                for index in range(5)
            ],
        }
        return dataset

    def test_audio_sample_plan_is_reproducible_and_changes_each_epoch(self):
        dataset = self._resampling_fixture(DynamicAudioChartDataset)
        dataset.set_epoch(0)
        first = [(category, row["chart_id"]) for category, row in dataset.sample_plan]
        dataset.set_epoch(1)
        second = [(category, row["chart_id"]) for category, row in dataset.sample_plan]
        dataset.set_epoch(0)
        repeated = [(category, row["chart_id"]) for category, row in dataset.sample_plan]
        self.assertNotEqual(first, second)
        self.assertEqual(first, repeated)

    def test_arranger_sample_plan_is_reproducible_and_changes_each_epoch(self):
        dataset = self._resampling_fixture(DynamicOraclePlanDataset)
        dataset.set_epoch(0)
        first = [(category, row["chart_id"]) for category, row in dataset.sample_plan]
        dataset.set_epoch(2)
        second = [(category, row["chart_id"]) for category, row in dataset.sample_plan]
        self.assertNotEqual(first, second)

    def test_padded_crop_preserves_data_and_extrapolates_time(self):
        chart = np.zeros((48, 1000), dtype=np.uint8)
        chart[0, 200:210] = 1
        timing = np.arange(1001, dtype=np.float32) * 5.0
        cropped, times = DynamicAudioChartDataset._padded_crop(chart, timing, 192, 384)
        self.assertEqual(cropped.shape, (48, 3072))
        self.assertEqual(times.shape, (3073,))
        self.assertEqual(int(cropped[0, 8:18].sum()), 10)
        self.assertTrue(np.all(cropped[:, 384:] == 0))
        self.assertAlmostEqual(float(times[385] - times[384]), 5.0)


if __name__ == "__main__":
    unittest.main()
