import numpy as np
from typing import Callable, Tuple


class Hamiltonian:

    def __init__(self, N: int, mpo: Callable, solve: bool = True):
        """

        Args:
            N: System size.
            mpo: Matrix Product Operator (MPO) as a function of site,
                with open boundary condition.
            solve: Solve the full Hamiltonian with exact diagonalization.
                Default Ture.
        """
        self.N = N
        self._matrix = None
        self.matrix = mpo
        self._eigval, self._eigvec = self._eigen_solver() if solve else (None, None)

    @property
    def matrix(self) -> np.ndarray:
        return self._matrix

    @matrix.setter
    def matrix(self, mpo: Callable):
        """
        Constructing the full Hamiltonian with MPO.

        Args:
            mpo:

        Returns:

        """
        d = mpo(0).shape[2]
        chi = mpo(0).shape[0]
        for site in range(self.N):
            if site == 0:
                self._matrix = mpo(site)[0, :, :, :]
            elif site == self.N - 1:
                self._matrix = np.tensordot(self._matrix, mpo(site)[:, -1, :, :], axes=(0, 0))
                self._matrix = np.swapaxes(self._matrix, 1, 2)
                self._matrix = self._matrix.reshape((d ** self.N, d ** self.N))
            else:
                self._matrix = np.tensordot(self._matrix, mpo(site), axes=(0, 0))
                self._matrix = np.swapaxes(self._matrix, 0, 2)
                self._matrix = np.swapaxes(self._matrix, 1, 3)
                self._matrix = np.swapaxes(self._matrix, 1, 2)
                self._matrix = self._matrix.reshape((chi, d ** (site+1), d ** (site+1)))

    @property
    def eigval(self) -> np.ndarray:
        return self._eigval

    @property
    def eigvec(self) -> np.ndarray:
        return self._eigvec

    def _eigen_solver(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Exact diagonalization for the hull Hamiltonian,
        which is expected to be a Hermition matrix.

        Returns:
            w: The eigenvalues in ascending order, each repeated according to its multiplicity.
            v: The column v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i].

        """
        return np.linalg.eigh(self._matrix)
