import unittest

import numpy as np

from maimai_ai.dynamic_audio_dataset import DynamicAudioChartDataset


class DynamicDatasetTests(unittest.TestCase):
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
