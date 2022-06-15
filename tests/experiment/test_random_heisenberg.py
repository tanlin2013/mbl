import pytest
import numpy as np
import pandas as pd

from mbl.name_space import Columns
from mbl.schema import RandomHeisenbergEDSchema, RandomHeisenbergFoldingTSDRGSchema
from mbl.experiment.random_heisenberg import (
    RandomHeisenbergED,
    RandomHeisenbergFoldingTSDRG,
)


class TestRandomHeisenbergED:
    @pytest.fixture(scope="class")
    def ham(self):
        return np.array(
            [[0.25, 0, 0, 0], [0, -0.25, 0.5, 0], [0, 0.5, -0.25, 0], [0, 0, 0, 0.25]]
        )

    @pytest.fixture(scope="class")
    def agent(self):
        return RandomHeisenbergED(n=2, h=0, penalty=0, s_target=0, trial_id=0)

    def test_matrix(self, agent, ham):
        np.testing.assert_array_equal(agent.ed.matrix, ham)

    def test_sorting_order(self, agent):
        assert agent.sorting_order is None

    def test_evals(self, agent):
        np.testing.assert_array_equal(agent.evals, np.array([-0.75, 0.25, 0.25, 0.25]))

    def test_total_sz(self, agent):
        np.testing.assert_array_equal(agent.total_sz, np.array([0, 1, 0, -1]))

    def test_entanglement_entropy(self, agent):
        np.testing.assert_allclose(
            agent.entanglement_entropy(0),
            np.array([np.log(2), np.nan, np.log(2), np.nan]),
            atol=1e-12,
        )

    def test_compute_df(self, agent):
        df = agent.compute_df()
        pd.testing.assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    Columns.level_id: np.arange(4),
                    Columns.en: [-0.75, 0.25, 0.25, 0.25],
                    Columns.total_sz: [0.0, 1.0, 0.0, -1.0],
                    Columns.edge_entropy: [np.log(2), np.nan, np.log(2), np.nan],
                    Columns.bipartite_entropy: [np.log(2), np.nan, np.log(2), np.nan],
                    Columns.system_size: 2,
                    Columns.disorder: 0 * np.ones(4),
                    Columns.trial_id: agent.model.trial_id,
                    Columns.seed: agent.model.seed,
                    Columns.penalty: 0 * np.ones(4),
                    Columns.s_target: 0,
                    Columns.offset: 0 * np.ones(4),
                }
            ),
        )
        RandomHeisenbergEDSchema.validate(df, lazy=True)


class TestRandomHeisenbergFoldingTSDRG:
    @pytest.fixture(scope="class")
    def agent(self):
        return RandomHeisenbergFoldingTSDRG(n=6, h=0.5, chi=2 ** 6)

    @pytest.fixture(scope="class")
    def ed_agent(self, agent):
        return RandomHeisenbergED(agent.model.n, agent.model.h, seed=agent.model.seed)

    def test_evals(self, agent, ed_agent):
        np.testing.assert_allclose(agent.evals, ed_agent.evals, atol=1e-12)

    def test_variance(self, agent):
        np.testing.assert_allclose(
            agent.variance, np.zeros(agent.tsdrg.chi), atol=1e-12
        )

    def test_total_sz(self, agent, ed_agent):
        np.testing.assert_allclose(agent.total_sz, ed_agent.total_sz, atol=1e-12)

    def test_edge_entropy(self, agent, ed_agent):
        np.testing.assert_allclose(
            agent.edge_entropy,
            np.nan_to_num(ed_agent.entanglement_entropy(0)),
            atol=1e-12,
        )

    def test_compute_df(self, agent, ed_agent):
        n_row = agent.tsdrg.chi
        df = agent.compute_df()
        pd.testing.assert_frame_equal(
            df,
            pd.DataFrame(
                {
                    Columns.level_id: np.arange(n_row),
                    Columns.en: ed_agent.evals,
                    Columns.variance: np.zeros(n_row),
                    Columns.total_sz: ed_agent.total_sz,
                    Columns.edge_entropy: np.nan_to_num(
                        ed_agent.entanglement_entropy(0)
                    ),
                    Columns.truncation_dim: agent.tsdrg.chi,
                    Columns.system_size: agent.model.n,
                    Columns.disorder: agent.model.h * np.ones(n_row),
                    Columns.trial_id: agent.model.trial_id,
                    Columns.seed: agent.model.seed,
                    Columns.penalty: 0 * np.ones(n_row),
                    Columns.s_target: 0,
                    Columns.offset: 0 * np.ones(n_row),
                    Columns.max_en: np.nan * np.ones(n_row),
                    Columns.min_en: np.nan * np.ones(n_row),
                    Columns.relative_offset: 0 * np.ones(n_row),
                }
            ),
            atol=1e-12,
        )
        RandomHeisenbergFoldingTSDRGSchema.validate(df, lazy=True)