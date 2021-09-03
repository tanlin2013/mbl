import unittest
import numpy as np
from mbl.model.utils import Entanglement


class TestEntanglement(unittest.TestCase):

    inv_sq2 = 1 / np.sqrt(2)
    agent = Entanglement(
        np.array(
            [[0, 1, 0, 0],
             [inv_sq2, 0, inv_sq2, 0],
             [-1 * inv_sq2, 0, inv_sq2, 0],
             [0, 0, 0, 1]]
        )
    )

    def test_singular_values(self):
        np.testing.assert_array_equal(
            self.agent._singular_values(),
            [
                np.array([self.inv_sq2, self.inv_sq2]),
                np.array([1, 0]),
                np.array([self.inv_sq2, self.inv_sq2]),
                np.array([1, 0])
            ]
        )

    def test_von_neumann_entropy(self):
        np.testing.assert_array_almost_equal(
            self.agent.von_neumann_entropy(),
            [np.log(2), np.nan, np.log(2), np.nan],
            decimal=8
        )


if __name__ == '__main__':
    unittest.main()
