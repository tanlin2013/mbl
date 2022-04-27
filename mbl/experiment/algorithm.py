import abc
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from mlflow import log_metric, log_param, log_artifact
from tnpy.operators import FullHamiltonian, MatrixProductOperator
from tnpy.model import ModelBase, TotalSz
from tnpy.exact_diagonalization import ExactDiagonalization
from tnpy.tsdrg import TreeTensorNetworkSDRG

from mbl.name_space import Columns


class Experiment1D(abc.ABC):
    def __init__(self, model: ModelBase):
        self._model = model

    @property
    def model(self) -> ModelBase:
        return self._model

    @abc.abstractmethod
    def _mpo_run_method(self) -> MatrixProductOperator:
        return self._model.mpo

    @abc.abstractmethod
    def compute_df(self) -> pd.DataFrame:
        return NotImplemented


class EDExperiment(Experiment1D):
    def __init__(self, model: ModelBase):
        super().__init__(model)
        self._ed = ExactDiagonalization(self._mpo_run_method())
        self._evals = self._ed.evals

    @property
    def ed(self) -> ExactDiagonalization:
        return self._ed

    def _mpo_run_method(self) -> MatrixProductOperator:
        return self._model.mpo

    @property
    def evals(self) -> np.ndarray:
        return self._evals

    @property
    def total_sz(self) -> np.ndarray:
        _total_sz = FullHamiltonian(TotalSz(self.model.n).mpo).matrix
        return np.diag(self.ed.evecs.T @ _total_sz @ self.ed.evecs)

    def entanglement_entropy(self, site: int) -> np.ndarray:
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
    def __init__(self, model: ModelBase, chi: int):
        super().__init__(model)
        log_param(Columns.truncation_dim, chi)
        self._tsdrg = TreeTensorNetworkSDRG(self._mpo_run_method(), chi=chi)
        self._tsdrg.run()
        self._evals = self.tsdrg.measurements.expectation_value(self.model.mpo)
        self._sorting_order = np.argsort(self._evals)

    @abc.abstractmethod
    def _mpo_run_method(self) -> MatrixProductOperator:
        return self._model.mpo

    @property
    def tsdrg(self) -> TreeTensorNetworkSDRG:
        return self._tsdrg

    @property
    def evals(self) -> np.ndarray:
        return self._evals[self._sorting_order]

    @property
    def variance(self) -> np.ndarray:  # TODO: check off-diagonal warning
        var = self.tsdrg.measurements.expectation_value(
            self.model.mpo.square()
        ) - np.square(self.tsdrg.measurements.expectation_value(self.model.mpo))
        return var[self._sorting_order]

    @property
    def total_sz(self) -> np.ndarray:
        return self.tsdrg.measurements.expectation_value(TotalSz(self.model.n).mpo)[
            self._sorting_order
        ]

    @property
    def edge_entropy(self) -> np.ndarray:
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

    def save_tree(self, filename: str, artifact_path: str = None):
        filename = Path(f"{filename}.p")
        logging.info(f"Dumping tree into file {filename}")
        with open(filename, "wb") as f:
            pickle.dump(self.tsdrg.tree, f)
        log_artifact(str(filename), artifact_path=artifact_path)