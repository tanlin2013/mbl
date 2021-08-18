import numpy as np
from typing import Callable


class Hamiltonian:

    def __init__(self, N: int, mpo: Callable):
        self.N = N
        self._matrix = None
        self.matrix = mpo

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, mpo: Callable):
        d = mpo(0).shape[2]
        chi = mpo(0).shape[0]
        for site in range(self.N):
            if site == 0:
                self._matrix = mpo(site)[0, :, :, :]
                self._matrix = self._matrix.reshape((1, chi, d ** (site + 1), d ** (site + 1)))
            elif site == self.N - 1:
                self._matrix = np.tensordot(self._matrix, mpo(site)[:, -1, :, :], axes=(1, 0))
                self._matrix = self._matrix.reshape((d ** (site + 1), d ** (site + 1)))
            else:
                self._matrix = np.tensordot(self._matrix, mpo(site), axes=(1, 0))
                self._matrix = self._matrix.reshape((1, chi, d**(site+1), d**(site+1)))
