"""Regression tests for simulation RNG state isolation."""

import importlib
import unittest
from unittest.mock import patch

import numpy as np

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import random, to_numpy

simulation_module = importlib.import_module(
    "pyrecest.evaluation.generate_simulated_scenarios"
)


def _backend_random_scalar() -> float:
    return float(np.asarray(to_numpy(random.rand())).reshape(()))


class TestGenerateSimulatedScenariosRngState(unittest.TestCase):
    def _assert_rng_states_preserved(self, action) -> None:
        original_backend_state = random.get_state()
        original_numpy_state = np.random.get_state()
        try:
            random.seed(314159)
            np.random.seed(271828)
            expected_backend_state = random.get_state()
            expected_numpy_state = np.random.get_state()

            expected_backend_value = _backend_random_scalar()
            expected_numpy_value = float(np.random.random())

            random.set_state(expected_backend_state)
            np.random.set_state(expected_numpy_state)
            action()

            self.assertEqual(_backend_random_scalar(), expected_backend_value)
            self.assertEqual(float(np.random.random()), expected_numpy_value)
        finally:
            random.set_state(original_backend_state)
            np.random.set_state(original_numpy_state)

    @staticmethod
    def _checked_config(_simulation_params):
        return {"all_seeds": [7, 11], "n_timesteps": 1}

    @staticmethod
    def _groundtruth(_simulation_params):
        random.rand()
        return [np.array([0.0])]

    @staticmethod
    def _measurements(_groundtruth, _simulation_params):
        np.random.random()
        return [np.array([[0.0]])]

    def test_generation_preserves_backend_and_numpy_rng_states(self):
        def generate():
            with (
                patch.object(
                    simulation_module,
                    "check_and_fix_config",
                    side_effect=self._checked_config,
                ),
                patch.object(
                    simulation_module,
                    "generate_groundtruth",
                    side_effect=self._groundtruth,
                ),
                patch.object(
                    simulation_module,
                    "generate_measurements",
                    side_effect=self._measurements,
                ),
            ):
                groundtruths, measurements = (
                    simulation_module.generate_simulated_scenarios({})
                )
                self.assertEqual(groundtruths.shape, (2, 1))
                self.assertEqual(measurements.shape, (2, 1))

        self._assert_rng_states_preserved(generate)

    def test_generation_restores_rng_states_when_generation_fails(self):
        def failing_groundtruth(_simulation_params):
            random.rand()
            np.random.random()
            raise RuntimeError("synthetic generation failure")

        def generate():
            with (
                patch.object(
                    simulation_module,
                    "check_and_fix_config",
                    side_effect=self._checked_config,
                ),
                patch.object(
                    simulation_module,
                    "generate_groundtruth",
                    side_effect=failing_groundtruth,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "synthetic generation failure"):
                    simulation_module.generate_simulated_scenarios({})

        self._assert_rng_states_preserved(generate)


if __name__ == "__main__":
    unittest.main()
