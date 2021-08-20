import numpy as np
import pandas as pd
from mbl.model.utils import (
    SpinOperators,
    Hamiltonian,
    TotalSz
)


class RandomHeisenberg(Hamiltonian):

    def __init__(self, N: int, h: float, penalty: float, s_target: int, trial_id: int):
        """

        Args:
            N: System size.
            h: Disorder strength.
            penalty: Penalty strength (or Lagrangian multiplier).
            s_target: The targeting total Sz charge sector.
            trial_id: ID of the current disorder trial.
        """
        self.N = N
        self.h = h
        self.penalty = penalty
        self.s_target = s_target
        self.trial_id = trial_id
        super(RandomHeisenberg, self).__init__(N, self._mpo)
        self._total_sz = TotalSz(N, self.eigvec).val

    def _mpo(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()

        alpha = self.penalty * (0.25 + self.s_target ** 2 / self.N)
        beta = np.random.uniform(-self.h, self.h) - 2.0 * self.penalty * self.s_target

        return np.array(
            [[I2, 0.5 * Sp, 0.5 * Sm, 2.0 * self.penalty * Sz, Sz, alpha * I2 + beta * Sz],
             [O2, O2, O2, O2, O2, Sm],
             [O2, O2, O2, O2, O2, Sp],
             [O2, O2, O2, I2, O2, Sz],
             [O2, O2, O2, O2, O2, Sz],
             [O2, O2, O2, O2, O2, I2]],
            dtype=float
        )

    @property
    def total_sz(self):
        return self._total_sz

    @property
    def df(self) -> pd.DataFrame:
        n_row = len(self.eigval)
        return pd.DataFrame(
            {
                'LevelID': list(range(n_row)),
                'En': self.eigval,
                'TotalSz': self._total_sz,
                'SystemSize': [self.N] * n_row,
                'Disorder': [self.h] * n_row,
                'Penalty': [self.penalty] * n_row,
                'STarget': [self.s_target] * n_row,
                'TrialID': [self.trial_id] * n_row
            }
        )
