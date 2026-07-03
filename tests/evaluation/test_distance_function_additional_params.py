import unittest

from pyrecest.evaluation.get_distance_function import get_distance_function


class DistanceFunctionAdditionalParamsTest(unittest.TestCase):
    def test_euclidean_mtt_rejects_non_mapping_params(self):
        for params in ([], ["cutoff_distance"], "cutoff_distance", 2.5):
            with self.subTest(params=params):
                with self.assertRaisesRegex(ValueError, "additional_params.*mapping"):
                    get_distance_function("euclidean_mtt", params)


if __name__ == "__main__":
    unittest.main()
