import unittest
import numpy as np
import pandas as pd
from tensornetwork import ncon
from tnpy.operators import SpinOperators
from mbl.model import RandomHeisenbergED, SpectralFoldedRandomHeisenbergED


class TesRandomHeisenbergED(unittest.TestCase):

    ham = np.array(
        [[0.25, 0, 0, 0],
         [0, -0.25, 0.5, 0],
         [0, 0.5, -0.25, 0],
         [0, 0, 0, 0.25]]
    )
    agent = RandomHeisenbergED(N=2, h=0, penalty=0, s_target=0, trial_id=0)

    def test_mpo(self):
        Sp, Sm, Sz, I2, O2 = SpinOperators()
        np.testing.assert_array_equal(
            np.array([I2, 0.5 * Sp, 0.5 * Sm, O2, Sz, O2]),
            self.agent.mpo[0].tensor
        )
        np.testing.assert_array_equal(
            np.array([[O2], [Sm], [Sp], [Sz], [Sz], [I2]]).reshape((6, 2, 2)),
            self.agent.mpo[1].tensor
        )

    def test_matrix(self):
        np.testing.assert_array_equal(self.agent.ed.matrix, self.ham)

    def test_df(self):
        pd.testing.assert_frame_equal(
            pd.DataFrame(
                {
                    'LevelID': list(range(4)),
                    'En': [-0.75, 0.25, 0.25, 0.25],
                    'TotalSz': [0., 1., 0., -1.],
                    'EdgeEntropy': [np.log(2), np.nan, np.log(2), np.nan],
                    'BipartiteEntropy': [np.log(2), np.nan, np.log(2), np.nan],
                    'SystemSize': [2, 2, 2, 2],
                    'Disorder': [0, 0, 0, 0],
                    'Penalty': [0, 0, 0, 0],
                    'STarget': [0, 0, 0, 0],
                    'TrialID': [0, 0, 0, 0]
                }
            ),
            self.agent.df
        )


class TestSpectralFoldedRandomHeisenbergED(unittest.TestCase):

    ham = np.array(
        [[0.25, 0, 0, 0],
         [0, -0.25, 0.5, 0],
         [0, 0.5, -0.25, 0],
         [0, 0, 0, 0.25]]
    )
    agent = SpectralFoldedRandomHeisenbergED(N=2, h=0, penalty=0, s_target=0, trial_id=0)

    def test_mpo(self):
        Sp, Sm, Sz, I2, O2 = SpinOperators()
        W1 = np.array([I2, 0.5 * Sp, 0.5 * Sm, O2, Sz, O2])
        np.testing.assert_array_equal(
            ncon([W1, W1], [(-1, '-a1', 1), (-2, 1, '-b2')]).reshape((36, 2, 2)),
            self.agent.mpo[0].tensor
        )
        W2 = np.array([[O2], [Sm], [Sp], [Sz], [Sz], [I2]]).reshape((6, 2, 2))
        np.testing.assert_array_equal(
            ncon([W2, W2], [(-1, '-a1', 1), (-2, 1, '-b2')]).reshape((36, 2, 2)),
            self.agent.mpo[1].tensor
        )


if __name__ == '__main__':
    unittest.main()
