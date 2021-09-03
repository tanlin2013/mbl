import numpy as np
from typing import List


class Entanglement:

    def __init__(self, eigvec: np.ndarray):
        self._eigvec = eigvec

    def _singular_values(self) -> List[np.ndarray]:
        return [np.linalg.svd(v.reshape(-1, 2))[1] for v in self._eigvec.T]

    def von_neumann_entropy(self) -> np.ndarray:
        squared_sv = [np.square(sv) for sv in self._singular_values()]
        return np.array([-1 * np.sum(ss @ np.log(ss)) for ss in squared_sv])
