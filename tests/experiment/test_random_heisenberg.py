from unittest.mock import patch
from contextlib import nullcontext as does_not_raise

import pytest
import numpy as np
import pandas as pd
from tnpy.tsdrg import TreeTensorNetworkSDRG

from mbl.name_space import Columns
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
        return RandomHeisenbergED(n=2, h=0, penalty=0, s_target=0)

    def test_matrix(self, agent, ham):
        np.testing.assert_array_equal(agent.ed.matrix, ham)

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


class TestRandomHeisenbergFoldingTSDRG:
    @pytest.mark.parametrize(
        "max_en, min_en, relative_offset, as_expected",
        [
            (np.nan, -3.5, 0.5, does_not_raise()),
            (2, -2, 0.5, does_not_raise()),
            (-3, 2, 0.5, pytest.raises(AssertionError)),
            (2, -2, 3, pytest.raises(AssertionError)),
        ],
    )
    @patch.object(TreeTensorNetworkSDRG, "run", return_value=None)
    @patch.object(TreeTensorNetworkSDRG, "measurements", return_value=None)
    def test_offset(
        self,
        TreeTensorNetworkSDRG,
        TreeTensorNetworkMeasurements,
        max_en,
        min_en,
        relative_offset,
        as_expected,
    ):
        with as_expected:
            tsgrg = RandomHeisenbergFoldingTSDRG(
                n=6,
                h=0.5,
                chi=16,
                max_en=max_en,
                min_en=min_en,
                relative_offset=relative_offset,
            )
            if np.isnan([max_en, min_en]).any():
                assert tsgrg._folded_model.offset == 0
            else:
                assert min_en <= tsgrg._folded_model.offset <= max_en

    @pytest.fixture(scope="class")
    def tsdrg_agent(self):
        return RandomHeisenbergFoldingTSDRG(n=6, h=0.5, chi=2**6)

    @pytest.fixture(scope="class")
    def ed_agent(self, tsdrg_agent):
        return RandomHeisenbergED(
            tsdrg_agent.model.n, tsdrg_agent.model.h, seed=tsdrg_agent.model.seed
        )

    def test_evals(self, tsdrg_agent, ed_agent):
        np.testing.assert_allclose(tsdrg_agent.evals, ed_agent.evals, atol=1e-12)

    def test_variance(self, tsdrg_agent):
        np.testing.assert_allclose(
            tsdrg_agent.variance, np.zeros(tsdrg_agent.tsdrg.chi), atol=1e-12
        )

    def test_total_sz(self, tsdrg_agent, ed_agent):
        np.testing.assert_allclose(tsdrg_agent.total_sz, ed_agent.total_sz, atol=1e-12)

    def test_edge_entropy(self, tsdrg_agent, ed_agent):
        np.testing.assert_allclose(
            tsdrg_agent.edge_entropy,
            np.nan_to_num(ed_agent.entanglement_entropy(0)),
            atol=1e-12,
        )

    def test_compute_df(self, tsdrg_agent, ed_agent):
        n_row = tsdrg_agent.tsdrg.chi
        df = tsdrg_agent.compute_df()
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
                    Columns.truncation_dim: tsdrg_agent.tsdrg.chi,
                    Columns.system_size: tsdrg_agent.model.n,
                    Columns.disorder: tsdrg_agent.model.h * np.ones(n_row),
                    Columns.trial_id: tsdrg_agent.model.trial_id,
                    Columns.seed: tsdrg_agent.model.seed,
                    Columns.penalty: 0 * np.ones(n_row),
                    Columns.s_target: 0,
                    Columns.offset: 0 * np.ones(n_row),
                    Columns.max_en: np.nan * np.ones(n_row),
                    Columns.min_en: np.nan * np.ones(n_row),
                    Columns.relative_offset: 0 * np.ones(n_row),
                    Columns.method: "min",
                }
            ),
            atol=1e-12,
        )
