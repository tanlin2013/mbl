import pytest
import numpy as np

from mbl.name_space import Columns
from mbl.experiment.random_heisenberg import RandomHeisenbergED
from mbl.analysis.energy_bounds import EnergyBounds


@pytest.mark.parametrize("n, h, seed", [(8, 0.5, 2022), (8, 10.0, 2022)])
@pytest.mark.parametrize("chi", [32, 64])
@pytest.mark.parametrize("method", ["min", "max"])
def test_athena_query(n, h, seed, chi, method):
    en = EnergyBounds.athena_query(n=n, h=h, seed=seed, chi=chi, method=method)
    ed = RandomHeisenbergED(n=n, h=h, seed=seed)
    np.testing.assert_allclose(en, getattr(np, method)(ed.evals), atol=1e-12)


@pytest.mark.parametrize("n, h, seed", [(10, 0.5, 2040), (10, 10.0, 2040)])
@pytest.mark.parametrize("chi", [64])
def test_retrieve(n, h, seed, chi):
    en_bounds = EnergyBounds.retrieve(n=n, h=h, seed=seed, chi=chi)
    assert Columns.max_en, Columns.min_en in en_bounds
    ed = RandomHeisenbergED(n=n, h=h, seed=seed)
    np.testing.assert_allclose(en_bounds[Columns.max_en], np.max(ed.evals), atol=1e-6)
    np.testing.assert_allclose(en_bounds[Columns.min_en], np.min(ed.evals), atol=1e-6)
