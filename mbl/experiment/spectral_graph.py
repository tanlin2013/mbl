import numpy as np
import networkx as nx

# from tnpy.model import RandomHeisenberg


class Affinity:
    def __init__(self, array: np.ndarray):
        self._graph = nx.from_numpy_array(array)

    @property
    def graph(self) -> nx.Graph:
        return self._graph
