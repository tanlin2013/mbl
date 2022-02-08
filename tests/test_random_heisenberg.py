import unittest
import numpy as np
import pandas as pd
from mbl.model import (
    RandomHeisenbergED,
    RandomHeisenbergTSDRG
)


class TesRandomHeisenbergED(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TesRandomHeisenbergED, self).__init__(*args, **kwargs)
        self.ham = np.array(
            [[0.25, 0, 0, 0],
             [0, -0.25, 0.5, 0],
             [0, 0.5, -0.25, 0],
             [0, 0, 0, 0.25]]
        )
        self.agent = RandomHeisenbergED(n=2, h=0, penalty=0, s_target=0, trial_id=0)

    def test_matrix(self):
        np.testing.assert_array_equal(self.agent.ed.matrix, self.ham)

    def test_sorting_order(self):
        self.assertIsNone(self.agent.sorting_order)

    def test_evals(self):
        np.testing.assert_array_equal(
            self.agent.evals,
            np.array([-0.75, 0.25, 0.25, 0.25])
        )

    def test_total_sz(self):
        np.testing.assert_array_equal(
            self.agent.total_sz,
            np.array([0, 1, 0, -1])
        )

    def test_entanglement_entropy(self):
        np.testing.assert_allclose(
            self.agent.entanglement_entropy(0),
            np.array([np.log(2), np.nan, np.log(2), np.nan]),
            atol=1e-12
        )

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
                    'TrialID': [0, 0, 0, 0],
                    'Seed': [self.agent.model.seed] * 4,
                    'Penalty': [0, 0, 0, 0],
                    'STarget': [0, 0, 0, 0],
                    'Offset': [0, 0, 0, 0]
                }
            ),
            self.agent.df
        )


class TestRandomHeisenbergTSDRG(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestRandomHeisenbergTSDRG, self).__init__(*args, **kwargs)
        self.agent = RandomHeisenbergTSDRG(
            n=6, h=0.1, chi=2**6, trial_id=0, seed=2021, penalty=0, s_target=0
        )
        self.ed_agent = RandomHeisenbergED(
            n=6, h=0.1, trial_id=0, seed=2021, penalty=0, s_target=0
        )

    def test_evals(self):
        np.testing.assert_allclose(
            self.agent.evals,
            self.ed_agent.evals,
            atol=1e-12
        )

    def test_variance(self):
        np.testing.assert_allclose(
            self.agent.variance,
            np.zeros(self.agent.tsdrg.chi),
            atol=1e-12
        )

    def test_total_sz(self):
        np.testing.assert_allclose(
            self.agent.total_sz,
            self.ed_agent.total_sz,
            atol=1e-12
        )

    def test_edge_entropy(self):
        np.testing.assert_allclose(
            self.agent.edge_entropy,
            np.nan_to_num(self.ed_agent.entanglement_entropy(0)),
            atol=1e-12
        )

    def test_df(self):
        n_row = self.agent.tsdrg.chi
        pd.testing.assert_frame_equal(
            pd.DataFrame(
                {
                    'LevelID': list(range(n_row)),
                    'En': self.ed_agent.evals.tolist(),
                    'Variance': np.zeros(n_row),
                    'TotalSz': self.ed_agent.total_sz.tolist(),
                    'EdgeEntropy': np.nan_to_num(self.ed_agent.entanglement_entropy(0)).tolist(),
                    'TruncationDim': [self.agent.tsdrg.chi] * n_row,
                    'SystemSize': [self.agent.model.n] * n_row,
                    'Disorder': [self.agent.model.h] * n_row,
                    'TrialID': [self.agent.model.trial_id] * n_row,
                    'Seed': [self.agent.model.seed] * n_row,
                    'Penalty': [self.agent.folded_model.penalty] * n_row,
                    'STarget': [self.agent.folded_model.s_target] * n_row,
                    'Offset': [self.agent.folded_model.offset] * n_row
                }
            ),
            self.agent.df,
            atol=1e-12
        )
