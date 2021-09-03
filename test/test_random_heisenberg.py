import unittest
import numpy as np
from mbl.model import RandomHeisenberg


class TesRandomHeisenberg(unittest.TestCase):

    ham = np.array(
        [[0.25, 0, 0, 0],
         [0, -0.25, 0.5, 0],
         [0, 0.5, -0.25, 0],
         [0, 0, 0, 0.25]]
    )
    agent = RandomHeisenberg(N=2, h=0, penalty=0, s_target=0, trial_id=0)

    def test_matrix(self):
        np.testing.assert_array_equal(self.agent.matrix, self.ham)


if __name__ == '__main__':
    unittest.main()
