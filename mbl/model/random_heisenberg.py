import numpy as np
from mbl.model.utils import (
    SpinOperators,
    Hamiltonian,
    TotalSz
)


class RandomHeisenberg(Hamiltonian):

    def __init__(self, N: int, h: float, penalty: float, s_target: int):
        """

        Args:
            N: System size.
            h: Disorder strength.
            penalty: Penalty strength (or Lagrangian multiplier).
            s_target: The targeting total Sz charge sector.
        """
        self.N = N
        self.h = h
        self.penalty = penalty
        self.s_target = s_target
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
