import math
import unittest

from ran_llm_xapp.config import ExperimentConfig
from ran_llm_xapp.metrics import (
    reliability_outage_fraction,
    utility_s1_sigmoid,
    utility_s2_log_ratio,
)


class TestMetrics(unittest.TestCase):
    def test_utility_s1_sigmoid_at_demand_is_high(self) -> None:
        cfg = ExperimentConfig()
        u = utility_s1_sigmoid(cfg.sigma1, cfg.sigma1, a=cfg.a, b=cfg.b)
        self.assertGreater(u, 0.99)

    def test_utility_s2_log_ratio_at_demand_is_one(self) -> None:
        cfg = ExperimentConfig()
        u = utility_s2_log_ratio(cfg.sigma2, cfg.sigma2, c=cfg.c)
        self.assertAlmostEqual(u, 1.0, places=7)

    def test_reliability_outage_fraction_centered_window(self) -> None:
        # u = [0.0, 0.5, 1.0, 0.5, 0.0], threshold=0.6, Tw=4
        # At t=2, window=[0,4) -> [0.0,0.5,1.0,0.5] -> 3/4 below threshold.
        u = [0.0, 0.5, 1.0, 0.5, 0.0]
        theta = reliability_outage_fraction(u, threshold=0.6, Tw=4)
        self.assertAlmostEqual(theta[2], 0.75, places=7)

    def test_reliability_outage_fraction_truncates_boundaries(self) -> None:
        # At t=0, window would extend left; we truncate and divide by effective count.
        u = [0.0, 0.5, 1.0, 0.5, 0.0]
        theta = reliability_outage_fraction(u, threshold=0.6, Tw=4)
        self.assertAlmostEqual(theta[0], 1.0, places=7)  # [0.0,0.5] -> 2/2

    def test_reliability_ignores_nan(self) -> None:
        u = [float("nan"), 0.0, 1.0]
        theta = reliability_outage_fraction(u, threshold=0.6, Tw=3)
        # At t=1, window covers all; NaN ignored -> [0.0,1.0] -> 1/2
        self.assertAlmostEqual(theta[1], 0.5, places=7)


if __name__ == "__main__":
    unittest.main()

