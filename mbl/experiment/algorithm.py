import abc
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tnpy.operators import FullHamiltonian, MatrixProductOperator
from tnpy.model import Model1D, TotalSz
from tnpy.exact_diagonalization import ExactDiagonalization
from tnpy.tsdrg import TreeTensorNetworkSDRG, HighEnergyTreeTensorNetworkSDRG

from mbl import logger


class Experiment1D(abc.ABC):
    def __init__(self, model: Model1D):
        """
        Abstractive class for running algorithm against the 1-dimensional model.
        One should inherit this class and implement
        :meth:`Experiment1D._mpo_run_method` and :meth:`Experiment1D.compute_df`.

        Args:
            model: An 1d model.
        """
        self._model = model

    @property
    def model(self) -> Model1D:
        return self._model

    @abc.abstractmethod
    def _mpo_run_method(self) -> MatrixProductOperator:
        return self._model.mpo

    @abc.abstractmethod
    def compute_df(self) -> pd.DataFrame:
        return NotImplemented


class EDExperiment(Experiment1D):
    def __init__(self, model: Model1D):
        """
        Run an exact diagonalization experiment against the ``model``.

        Args:
            model: An 1d model.
        """
        super().__init__(model)
        self._ed = ExactDiagonalization(self._mpo_run_method())
        self._evals = self._ed.evals

    @property
    def ed(self) -> ExactDiagonalization:
        """
        A reference to the exact diagonalization solver.
        """
        return self._ed

    def _mpo_run_method(self) -> MatrixProductOperator:
        return self._model.mpo

    @property
    def evals(self) -> np.ndarray:
        """
        Eigenvalues in ascending order.
        """
        return self._evals

    @property
    def total_sz(self) -> np.ndarray:
        r"""
        The total :math:`S^z` for every eigenvectors.

        Returns:
            An array in the order same as the eigenvalues.
        """
        _total_sz = FullHamiltonian(TotalSz(self.model.n).mpo).matrix
        return np.diag(self.ed.evecs.T @ _total_sz @ self.ed.evecs)

    def entanglement_entropy(self, site: int) -> np.ndarray:
        """
        Compute the von Neumann entanglement entropy for every eigenvectors.

        Args:
            site: The site to which the bi-partition is taken.

        Returns:
            An array in the order same as the eigenvalues.
        """
        return np.array(
            [
                self.ed.entanglement_entropy(site, level_idx)
                for level_idx in range(len(self.ed.matrix))
            ]
        )

    @abc.abstractmethod
    def compute_df(self) -> pd.DataFrame:
        return NotImplemented


class TSDRGExperiment(Experiment1D):
    def __init__(self, model: Model1D, chi: int, method: str = "min"):
        """
        Run a Tree Tensor Strong-Disorder Renormalization Group (tSDRG) experiment
        against the ``model``.

        Args:
            model: An 1d model.
            chi: The bond dimensions.
            method: Keep the highest or lowest ``chi`` eigenvectors in projection.
        """
        super().__init__(model)
        self._tsdrg = {
            "min": TreeTensorNetworkSDRG(self._mpo_run_method(), chi=chi),
            "max": HighEnergyTreeTensorNetworkSDRG(self._mpo_run_method(), chi=chi),
        }[method]
        self._tsdrg.run()
        self._evals = self.tsdrg.measurements.expectation_value(self.model.mpo)
        self._sorting_order = np.argsort(self._evals)

    @abc.abstractmethod
    def _mpo_run_method(self) -> MatrixProductOperator:
        return self._model.mpo

    @property
    def tsdrg(self) -> TreeTensorNetworkSDRG:
        """
        A reference to the TSDRG solver.
        """
        return self._tsdrg

    @property
    def evals(self) -> np.ndarray:
        """
        Eigenvalues in ascending order.
        """
        return self._evals[self._sorting_order]

    @property
    def variance(self) -> np.ndarray:
        """
        Energy variance.

        Returns:
            An array in the order same as the eigenvalues.
        """
        var = self.tsdrg.measurements.expectation_value(
            self.model.mpo.square()
        ) - np.square(self.tsdrg.measurements.expectation_value(self.model.mpo))
        return var[self._sorting_order]

    @property
    def total_sz(self) -> np.ndarray:
        r"""
        The total :math:`S^z` for every eigenvectors.

        Returns:
            An array in the order same as the eigenvalues.
        """
        return self.tsdrg.measurements.expectation_value(TotalSz(self.model.n).mpo)[
            self._sorting_order
        ]

    @property
    def edge_entropy(self) -> np.ndarray:
        """
        Compute the von Neumann entanglement entropy for every eigenvectors,
        where the bi-partition is taken on the 1st bond.
        That is, part :math:`A` contains only one site,
        leaving the rest of sites in part :math:`B`.

        Returns:
            An array in the order same as the eigenvalues.
        """
        return np.array(
            [
                self.tsdrg.measurements.entanglement_entropy(
                    site=0, level_idx=level_idx
                )
                for level_idx in range(self.tsdrg.chi)
            ]
        )[self._sorting_order]

    @abc.abstractmethod
    def compute_df(self) -> pd.DataFrame:
        return NotImplemented

    def save_tree(self, filename: str):
        """
        Serialize the :class:`~tnpy.tsdrg.TensorTree` as a pickle binary.

        Args:
            filename:

        References:
            `<https://docs.python.org/3/library/pickle.html>`_
        """
        assert Path(filename).suffix == ".p"
        logger.info(f"Dumping tree into file {filename}")
        with open(filename, "wb") as f:
            pickle.dump(self.tsdrg.tree, f)
