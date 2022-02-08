import logging
import pickle
import numpy as np
import pandas as pd
from time import time
from tnpy.operators import FullHamiltonian
from tnpy.model import RandomHeisenberg, TotalSz
from tnpy.exact_diagonalization import ExactDiagonalization
from tnpy.tsdrg import (
    TreeTensorNetworkSDRG as tSDRG,
    TreeTensorNetworkMeasurements
)


class RandomHeisenbergED:

    def __init__(self, n: int, h: float, trial_id: int = None, seed: int = None,
                 penalty: float = 0, s_target: int = 0, offset: float = 0, spectral_folding: bool = False):
        """

        Args:
            n: System size.
            h: Disorder strength.
            trial_id: ID of the current disorder trial.
            seed: Random seed used to initialize the pseudo-random number generator.
            penalty: Penalty strength (or Lagrangian multiplier).
            s_target: The targeting total Sz charge sector.
            offset:
            spectral_folding:
        """
        seed = int(time()) if seed is None else seed
        self.model = RandomHeisenberg(n=n, h=h, trial_id=trial_id, seed=seed)
        if spectral_folding:
            logging.info("Using spectral folding method.")
            self.folded_model = RandomHeisenberg(
                n=n, h=h, trial_id=trial_id, seed=seed, penalty=penalty, s_target=s_target, offset=offset
            )
            self.ed = ExactDiagonalization(self.folded_model.mpo.square())
            self._evals = np.diag(self.ed.evecs.T @ FullHamiltonian(self.model.mpo) @ self.ed.evecs)
            self._sorting_order = np.argsort(self._evals)
        else:
            self.folded_model = self.model
            self.ed = ExactDiagonalization(self.model.mpo)
            self._evals = self.ed.evals
            self._sorting_order = None

    @property
    def sorting_order(self) -> np.ndarray:
        return self._sorting_order

    @property
    def evals(self) -> np.ndarray:
        return self._evals if self.sorting_order is None else self._evals[self.sorting_order]

    @property
    def total_sz(self) -> np.ndarray:
        _total_sz = FullHamiltonian(TotalSz(self.model.n).mpo).matrix
        exp_val = np.diag(self.ed.evecs.T @ _total_sz @ self.ed.evecs)
        return exp_val if self.sorting_order is None else exp_val[self.sorting_order]

    def entanglement_entropy(self, site: int) -> np.ndarray:
        entropy = np.array(
            [self.ed.entanglement_entropy(site, level_idx) for level_idx in range(len(self.ed.matrix))]
        )
        return entropy if self.sorting_order is None else entropy[self.sorting_order]

    @property
    def df(self) -> pd.DataFrame:
        n_row = len(self.ed.evecs)
        return pd.DataFrame(
            {
                'LevelID': list(range(n_row)),
                'En': self.evals.tolist(),
                'TotalSz': self.total_sz.tolist(),
                'EdgeEntropy': self.entanglement_entropy(site=0).tolist(),
                'BipartiteEntropy': self.entanglement_entropy(site=self.model.n // 2 - 1).tolist(),
                'SystemSize': [self.model.n] * n_row,
                'Disorder': [self.model.h] * n_row,
                'TrialID': [self.model.trial_id] * n_row,
                'Seed': [self.model.seed] * n_row,
                'Penalty': [self.folded_model.penalty] * n_row,
                'STarget': [self.folded_model.s_target] * n_row,
                'Offset': [self.folded_model.offset] * n_row
            }
        )


class RandomHeisenbergTSDRG(TreeTensorNetworkMeasurements):

    def __init__(self, n: int, h: float, chi: int, trial_id: int = None, seed: int = None,
                 penalty: float = 0, s_target: int = 0, offset: float = 0):
        seed = int(time()) if seed is None else seed
        self.model = RandomHeisenberg(n=n, h=h, trial_id=trial_id, seed=seed)
        self.folded_model = RandomHeisenberg(
            n=n, h=h, trial_id=trial_id, seed=seed, penalty=penalty, s_target=s_target, offset=offset
        )
        self.tsdrg = tSDRG(self.folded_model.mpo.square(), chi=chi)
        self.tsdrg.run()
        super(RandomHeisenbergTSDRG, self).__init__(self.tsdrg.tree)
        self._evals = self.expectation_value(self.model.mpo)
        self._sorting_order = np.argsort(self._evals)

    @property
    def sorting_order(self) -> np.ndarray:
        return self._sorting_order

    @property
    def evals(self) -> np.ndarray:
        return self._evals[self.sorting_order]

    @property
    def variance(self) -> np.ndarray:
        var = self.expectation_value(self.model.mpo.square()) \
              - np.square(self.expectation_value(self.model.mpo))
        return var[self.sorting_order]

    @property
    def total_sz(self) -> np.ndarray:
        return self.expectation_value(TotalSz(self.model.n).mpo)[self.sorting_order]

    @property
    def edge_entropy(self) -> np.ndarray:
        return np.array([
            self.entanglement_entropy(site=0, level_idx=level_idx) for level_idx in range(self.tsdrg.chi)
        ])[self.sorting_order]

    @property
    def df(self) -> pd.DataFrame:
        n_row = self.tsdrg.chi
        return pd.DataFrame(
            {
                'LevelID': list(range(n_row)),
                'En': self.evals.tolist(),
                'Variance': self.variance.tolist(),
                'TotalSz': self.total_sz.tolist(),
                'EdgeEntropy': self.edge_entropy.tolist(),
                'TruncationDim': [self.tsdrg.chi] * n_row,
                'SystemSize': [self.model.n] * n_row,
                'Disorder': [self.model.h] * n_row,
                'TrialID': [self.model.trial_id] * n_row,
                'Seed': [self.model.seed] * n_row,
                'Penalty': [self.folded_model.penalty] * n_row,
                'STarget': [self.folded_model.s_target] * n_row,
                'Offset': [self.folded_model.offset] * n_row
            }
        )

    def save_tree(self, filename: str):
        logging.info(f"Dumping tree into file {filename}.p")
        pickle.dump(self.tsdrg.tree, open(f'{filename}.p', 'wb'))
