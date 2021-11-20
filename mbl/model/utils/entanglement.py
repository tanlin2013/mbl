import numpy as np


class Entanglement:

    def __init__(self, eigvec: np.ndarray):
        self._eigvec = eigvec

    @staticmethod
    def singular_values(v: np.ndarray, position: int = 1) -> np.ndarray:
        assert not np.isnan(v).any(), "v contains nan value"
        assert v.ndim == 1, "v is supposed to be an 1d array"
        assert 0 < position < len(v) - 1, \
            f"position can't be negative or larger than the system size {len(v)}, got {position}"
        return np.linalg.svd(v.reshape(2 ** position, -1))[1]

    def von_neumann_entropy(self, position: int = 1) -> np.ndarray:
        def definition(v: np.ndarray):
            ss = np.square(self.singular_values(v, position))
            return -1 * np.sum(ss @ np.log(ss))
        return np.array([definition(v) for v in self._eigvec.T])
