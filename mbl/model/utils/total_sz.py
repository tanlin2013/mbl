import numpy as np
from mbl.model.utils import (
    SpinOperators,
    Hamiltonian
)


class TotalSz(Hamiltonian):

    def __init__(self, N: int, eigvec: np.ndarray):
        super(TotalSz, self).__init__(N, self._mpo, solve=False)
        self._val = np.diag(eigvec.T @ self.matrix @ eigvec)

    def _mpo(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()
        return np.array(
            [[I2, Sz],
             [O2, I2]]
        )

    @property
    def val(self) -> np.ndarray:
        return self._val
