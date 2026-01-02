import unittest

from ran_llm_xapp.metrics import action_to_prbs


class TestMapping(unittest.TestCase):
    def test_equal_action_maps_to_equal_prbs(self) -> None:
        r1, r2 = action_to_prbs((64, 64), 128)
        self.assertEqual((r1, r2), (64, 64))
        self.assertLessEqual(r1 + r2, 128)

    def test_proportional_action_budget_fixer(self) -> None:
        # Without fixer: ceil(102.4)=103 and ceil(25.6)=26 -> 129 > 128.
        r1, r2 = action_to_prbs((40, 10), 128)
        self.assertEqual((r1, r2), (102, 26))
        self.assertEqual(r1 + r2, 128)

    def test_mapping_respects_sum_constraint(self) -> None:
        for action in [(1, 1), (1, 128), (128, 1), (128, 128), (77, 3)]:
            r1, r2 = action_to_prbs(action, 128)
            self.assertGreaterEqual(r1, 0)
            self.assertGreaterEqual(r2, 0)
            self.assertLessEqual(r1 + r2, 128)


if __name__ == "__main__":
    unittest.main()

