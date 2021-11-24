import numpy as np
from tnpy.operators import (
    SpinOperators,
    MPO,
    FullHamiltonian
)


class TotalSz(FullHamiltonian):

    def __init__(self, N: int, eigvec: np.ndarray):
        self._mpo = MPO(N, self._elem)
        super(TotalSz, self).__init__(self.mpo)
        self._val = np.diag(eigvec.T @ self.matrix @ eigvec)

    def _elem(self, site: int) -> np.ndarray:
        Sp, Sm, Sz, I2, O2 = SpinOperators()
        return np.array(
            [[I2, Sz],
             [O2, I2]]
        )

    @property
    def mpo(self):
        return self._mpo

    @property
    def val(self) -> np.ndarray:
        return self._val
