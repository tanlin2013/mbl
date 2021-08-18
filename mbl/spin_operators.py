import numpy as np
from collections import namedtuple


class SpinOperators:

    def __init__(self, spin: float = 0.5):
        self.spin = spin

    def __new__(cls, spin: float = 0.5):
        super(SpinOperators, cls).__init__(spin)
        SOp = namedtuple('SpinOperators', ['Sp', 'Sm', 'Sz', 'I2', 'O2'])
        return SOp(Sp=np.array([[0, 1], [0, 0]], dtype=float),
                   Sm=np.array([[0, 0], [1, 0]], dtype=float),
                   Sz=spin * np.array([[1, 0], [0, -1]], dtype=float),
                   I2=np.identity(2, dtype=float),
                   O2=np.zeros((2, 2), dtype=float))
