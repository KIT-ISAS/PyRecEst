import unittest
from decimal import Decimal

import numpy as np
from pyrecest.reproducibility import seed_all


class _OverflowingScalar:
    shape = ()

    def item(self):
        raise OverflowError("simulated overflow")


class ReproducibilitySeedValidationTest(unittest.TestCase):
    def test_seed_all_rejects_invalid_seed_values(self):
        invalid_seeds = (
            True,
            False,
            -1,
            2**32,
            np.array(2**32, dtype=np.uint64),
            1.5,
            Decimal("1.0000000000000000000000001"),
            float("nan"),
            float("inf"),
            "8",
            [1],
            np.array([1]),
            object(),
            _OverflowingScalar(),
        )

        for seed in invalid_seeds:
            with self.subTest(seed=repr(seed)):
                with self.assertRaisesRegex(
                    ValueError,
                    "seed must be a non-negative integer or None",
                ):
                    seed_all(seed)

    def test_seed_all_accepts_integer_like_scalar_seed(self):
        self.assertIsNone(seed_all(None))
        self.assertEqual(seed_all(np.array(7.0)), 7)
        self.assertEqual(seed_all(Decimal("7")), 7)
        self.assertEqual(seed_all(2**32 - 1), 2**32 - 1)


if __name__ == "__main__":
    unittest.main()
