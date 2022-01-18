import numpy as np
import pandas as pd
from time import time
from tnpy.exact_diagonalization import ExactDiagonalization
from tnpy.tsdrg import TSDRG
from tnpy.model import RandomHeisenberg, SpectralFoldedRandomHeisenberg
from mbl.model.utils import (
    TotalSz,
    Entanglement
)


class RandomHeisenbergED(RandomHeisenberg):

    def __init__(self, N: int, h: float, penalty: float = 0, s_target: int = 0, trial_id: int = None, seed: int = None):
        """

        Args:
            N: System size.
            h: Disorder strength.
            penalty: Penalty strength (or Lagrangian multiplier).
            s_target: The targeting total Sz charge sector.
            trial_id: ID of the current disorder trial.
            seed: Random seed used to initialize the pseudo-random number generator.
        """
        super(RandomHeisenbergED, self).__init__(N, h, penalty, s_target, trial_id, seed)
        self.ed = ExactDiagonalization(self.mpo)
        self._total_sz = TotalSz(N, self.ed.evecs).val
        _entanglement = Entanglement(self.ed.evecs)
        self._edge_entropy = _entanglement.von_neumann_entropy(position=1)
        self._bipartite_entropy = _entanglement.von_neumann_entropy(position=N//2)

    @property
    def total_sz(self):
        return self._total_sz

    @property
    def df(self) -> pd.DataFrame:
        n_row = len(self.ed.evecs)
        return pd.DataFrame(
            {
                'LevelID': list(range(n_row)),
                'En': self.ed.evals,
                'TotalSz': self._total_sz,
                'EdgeEntropy': self._edge_entropy,
                'BipartiteEntropy': self._bipartite_entropy,
                'SystemSize': [self.N] * n_row,
                'Disorder': [self.h] * n_row,
                'Penalty': [self.penalty] * n_row,
                'STarget': [self.s_target] * n_row,
                'TrialID': [self.trial_id] * n_row
            }
        )


class SpectralFoldedRandomHeisenbergED(RandomHeisenbergED, SpectralFoldedRandomHeisenberg):

    def __init__(self, *args, **kwargs):
        super(SpectralFoldedRandomHeisenbergED, self).__init__(*args, **kwargs)


class RandomHeisenbergTSDRG:

    def __init__(self, N: int, h: float, chi: int, penalty: float = 0, s_target: int = 0, trial_id: int = None, seed: int = None):
        seed = int(time()) if seed is None else seed
        self.model = RandomHeisenberg(N, h, penalty, s_target, trial_id, seed)
        self.folded_model = SpectralFoldedRandomHeisenberg(N, h, penalty, s_target, trial_id, seed=self.model.seed)
        self.tsdrg = TSDRG(self.folded_model.mpo, chi)
        self.tsdrg.run()
        self._evals = np.diag(self.tsdrg.energies(self.model.mpo))

    @property
    def sorting_order(self) -> np.ndarray:
        return np.argsort(self._evals)

    @property
    def evals(self) -> np.ndarray:
        return self._evals[self.sorting_order]

    @property
    def variance(self) -> np.ndarray:
        var = self.tsdrg.energies(self.folded_model.mpo) - np.square(self.tsdrg.energies(self.model.mpo))
        return np.sum(var[self.sorting_order], axis=1)

    @property
    def df(self) -> pd.DataFrame:
        n_row = len(self.evals)
        return pd.DataFrame(
            {
                'LevelID': list(range(n_row)),
                'En': self.evals,
                'Variance': self.variance,
                'EdgeEntropy': self.tsdrg.entanglement_entropy(
                    on_site=0,
                    energy_level=np.argmin(self.sorting_order)
                ),
                'SystemSize': [self.model.N] * n_row,
                'Disorder': [self.model.h] * n_row,
                'TruncationDim': [self.tsdrg.chi] * n_row,
                'Penalty': [self.model.penalty] * n_row,
                'STarget': [self.model.s_target] * n_row,
                'TrialID': [self.model.trial_id] * n_row,
                'Seed': [self.model.seed] * n_row
            }
        )
