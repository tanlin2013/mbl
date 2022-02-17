import unittest
import numpy as np
import pandas as pd
from mbl.experiment import (
    RandomHeisenbergED,
    RandomHeisenbergTSDRG,
    Columns
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
                    Columns.level_id: list(range(4)),
                    Columns.en: [-0.75, 0.25, 0.25, 0.25],
                    Columns.total_sz: [0., 1., 0., -1.],
                    Columns.edge_entropy: [np.log(2), np.nan, np.log(2), np.nan],
                    Columns.bipartite_entropy: [np.log(2), np.nan, np.log(2), np.nan],
                    Columns.system_size: [2, 2, 2, 2],
                    Columns.disorder: [0, 0, 0, 0],
                    Columns.trial_id: [0, 0, 0, 0],
                    Columns.seed: [self.agent.model.seed] * 4,
                    Columns.penalty: [0, 0, 0, 0],
                    Columns.s_target: [0, 0, 0, 0],
                    Columns.offset: [0, 0, 0, 0]
                }
            ),
            self.agent.df
        )


class TestRandomHeisenbergTSDRG(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestRandomHeisenbergTSDRG, self).__init__(*args, **kwargs)
        self.agent = RandomHeisenbergTSDRG(
            n=6, h=0.5, chi=2**6
        )
        self.ed_agent = RandomHeisenbergED(
            self.agent.model.n, self.agent.model.h, seed=self.agent.model.seed
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
                    Columns.level_id: list(range(n_row)),
                    Columns.en: self.ed_agent.evals.tolist(),
                    Columns.variance: np.zeros(n_row),
                    Columns.total_sz: self.ed_agent.total_sz.tolist(),
                    Columns.edge_entropy: np.nan_to_num(self.ed_agent.entanglement_entropy(0)).tolist(),
                    Columns.truncation_dim: [self.agent.tsdrg.chi] * n_row,
                    Columns.system_size: [self.agent.model.n] * n_row,
                    Columns.disorder: [self.agent.model.h] * n_row,
                    Columns.trial_id: [self.agent.model.trial_id] * n_row,
                    Columns.seed: [self.agent.model.seed] * n_row,
                    Columns.penalty: [self.agent.folded_model.penalty] * n_row,
                    Columns.s_target: [self.agent.folded_model.s_target] * n_row,
                    Columns.offset: [self.agent.folded_model.offset] * n_row
                }
            ),
            self.agent.df,
            atol=1e-12
        )
