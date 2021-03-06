import uuid
from time import time

import numpy as np
import pandas as pd
from pandera import check_output
from tnpy.operators import MatrixProductOperator
from tnpy.model import RandomHeisenberg

from mbl.name_space import Columns
from mbl.schema import (
    RandomHeisenbergEDSchema,
    RandomHeisenbergTSDRGSchema,
    RandomHeisenbergFoldingTSDRGSchema,
)
from mbl.experiment.algorithm import EDExperiment, TSDRGExperiment


class RandomHeisenbergED(EDExperiment):
    def __init__(
        self,
        n: int,
        h: float,
        seed: int = None,
        penalty: float = 0,
        s_target: int = 0,
        offset: float = 0,
    ):
        """
        Run an exact diagonalization experiment against the random Heisenberg model.

        Args:
            n: System size.
            h: Disorder strength.
            seed: Random seed used to initialize the pseudo-random number generator.
                If not given (None), current time will be used as the seed.
            penalty: Penalty strength (or Lagrangian multiplier).
            s_target: The targeting total Sz charge sector.
            offset: An overall shift to the spectrum.
        """
        seed = int(time()) if seed is None else seed
        super().__init__(
            model=RandomHeisenberg(
                n=n,
                h=h,
                trial_id=uuid.uuid4().hex,
                seed=seed,
                penalty=penalty,
                s_target=s_target,
                offset=offset,
            )
        )

    @check_output(RandomHeisenbergEDSchema.to_schema(), lazy=True)
    def compute_df(self) -> pd.DataFrame:
        r"""
        Summarize the experiment with these physical quantities,

        * Energies
        * Total :math:`S^z`
        * The edge entropy
        * The bi-partition entropy

        Returns:
            A dataframe containing input parameters and the physical quantities.

        Raises:
            :class:`pandera.errors.SchemaError`: If data is inconsistent with
             the schema, :class:`mbl.schema.RandomHeisenbergEDSchema`.
        """
        n_row = len(self.ed.evecs)
        return pd.DataFrame(
            {
                Columns.level_id: np.arange(n_row),
                Columns.en: self.evals,
                Columns.total_sz: self.total_sz,
                Columns.edge_entropy: self.entanglement_entropy(site=0),
                Columns.bipartite_entropy: self.entanglement_entropy(
                    site=self.model.n // 2 - 1
                ),
                Columns.system_size: self.model.n,
                Columns.disorder: self.model.h * np.ones(n_row),
                Columns.trial_id: self.model.trial_id,
                Columns.seed: self.model.seed,
                Columns.penalty: self.model.penalty * np.ones(n_row),
                Columns.s_target: self.model.s_target,
                Columns.offset: self.model.offset * np.ones(n_row),
            }
        )


class RandomHeisenbergTSDRG(TSDRGExperiment):
    def __init__(
        self,
        n: int,
        h: float,
        chi: int,
        seed: int = None,
        penalty: float = 0,
        s_target: int = 0,
        offset: float = 0,
        overall_const: float = 1,
        method: str = "min",
    ):
        """
        Run a Tree Tensor Strong-Disorder Renormalization Group (tSDRG) experiment
        against the random Heisenberg model.

        Args:
            n: System size.
            h: Disorder strength.
            chi: The bond dimensions.
            seed: Random seed used to initialize the pseudo-random number generator.
                If not given (None), current time will be used as the seed.
            penalty: Penalty strength (or Lagrangian multiplier).
            s_target: The targeting total Sz charge sector.
            offset: An overall shift to the spectrum.
            overall_const: An overall constant multiplied on the spectrum.
            method: Keep the highest or lowest ``chi`` eigenvectors in projection.
        """
        seed = int(time()) if seed is None else seed
        self._overall_const = overall_const
        self._method = method
        super().__init__(
            model=RandomHeisenberg(
                n=n,
                h=h,
                trial_id=uuid.uuid4().hex,
                seed=seed,
                penalty=penalty,
                s_target=s_target,
                offset=offset,
            ),
            chi=chi,
            method=method,
        )

    def _mpo_run_method(self) -> MatrixProductOperator:
        return self._overall_const * self._model.mpo

    @check_output(RandomHeisenbergTSDRGSchema.to_schema(), lazy=True)
    def compute_df(self) -> pd.DataFrame:
        r"""
        Summarize the experiment with these physical quantities,

        * Energies
        * Energy variance
        * Total :math:`S^z`
        * The edge entropy

        Returns:
            A dataframe containing input parameters and the physical quantities.

        Raises:
            :class:`pandera.errors.SchemaError`: If data is inconsistent with
             the schema, :class:`mbl.schema.RandomHeisenbergTSDRGSchema`.
        """
        n_row = self.tsdrg.chi
        return pd.DataFrame(
            {
                Columns.level_id: np.arange(n_row),
                Columns.en: self.evals,
                Columns.variance: self.variance,
                Columns.total_sz: self.total_sz,
                Columns.edge_entropy: self.edge_entropy,
                Columns.truncation_dim: self.tsdrg.chi,
                Columns.system_size: self.model.n,
                Columns.disorder: self.model.h * np.ones(n_row),
                Columns.trial_id: self.model.trial_id,
                Columns.seed: self.model.seed,
                Columns.penalty: self.model.penalty * np.ones(n_row),
                Columns.s_target: self.model.s_target,
                Columns.offset: self.model.offset * np.ones(n_row),
                Columns.overall_const: self._overall_const * np.ones(n_row),
                Columns.method: self._method,
            }
        )


class RandomHeisenbergFoldingTSDRG(TSDRGExperiment):
    def __init__(
        self,
        n: int,
        h: float,
        chi: int,
        seed: int = None,
        penalty: float = 0,
        s_target: int = 0,
        max_en: float = np.nan,
        min_en: float = np.nan,
        relative_offset: float = 0,
        method: str = "min",
    ):
        r"""
        Run a Tree Tensor Strong-Disorder Renormalization Group (tSDRG) experiment
        against the random Heisenberg model with spectral folding,

        .. math::

            H \rightarrow (H - \epsilon)^2,

        where :math:`\epsilon = E_{min} + (E_{max} - E_{min}) \times \epsilon_{rel}`

        Args:
            n: System size.
            h: Disorder strength.
            chi: The bond dimensions.
            seed: Random seed used to initialize the pseudo-random number generator.
                If not given (None), current time will be used as the seed.
            penalty: Penalty strength (or Lagrangian multiplier).
            s_target: The targeting total Sz charge sector.
            max_en: An estimated upper bound of the spectrum.
                Default :class:`~numpy.nan` for not performing spectral folding.
            min_en: An estimated lower bound of the spectrum.
                Default :class:`~numpy.nan` for not performing spectral folding.
            relative_offset: A relative shift to the spectrum with respect to
                ``max_en`` and ``min_en``.
            method: Keep the highest or lowest ``chi`` eigenvectors in projection.
        """
        trial_id = uuid.uuid4().hex
        seed = int(time()) if seed is None else seed
        if not np.isnan([max_en, min_en]).any():
            assert max_en > min_en
        assert 0 <= relative_offset <= 1
        self._folded_model = RandomHeisenberg(
            n=n,
            h=h,
            trial_id=trial_id,
            seed=seed,
            penalty=penalty,
            s_target=s_target,
            offset=np.nan_to_num(min_en + (max_en - min_en) * relative_offset),
        )
        self._max_en = max_en
        self._min_en = min_en
        self._relative_offset = relative_offset
        self._method = method
        super().__init__(
            model=RandomHeisenberg(n=n, h=h, trial_id=trial_id, seed=seed),
            chi=chi,
            method=method,
        )

    def _mpo_run_method(self) -> MatrixProductOperator:
        return self._folded_model.mpo.square()

    @check_output(RandomHeisenbergFoldingTSDRGSchema.to_schema(), lazy=True)
    def compute_df(self) -> pd.DataFrame:
        r"""
        Summarize the experiment with these physical quantities,

        * Energies
        * Energy variance
        * Total :math:`S^z`
        * The edge entropy

        Returns:
            A dataframe containing input parameters and the physical quantities.

        Raises:
            :class:`pandera.errors.SchemaError`: If data is inconsistent with
             the schema, :class:`mbl.schema.RandomHeisenbergFoldingTSDRGSchema`.
        """
        n_row = self.tsdrg.chi
        return pd.DataFrame(
            {
                Columns.level_id: np.arange(n_row),
                Columns.en: self.evals,
                Columns.variance: self.variance,
                Columns.total_sz: self.total_sz,
                Columns.edge_entropy: self.edge_entropy,
                Columns.truncation_dim: self.tsdrg.chi,
                Columns.system_size: self.model.n,
                Columns.disorder: self.model.h * np.ones(n_row),
                Columns.trial_id: self.model.trial_id,
                Columns.seed: self.model.seed,
                Columns.penalty: self._folded_model.penalty * np.ones(n_row),
                Columns.s_target: self._folded_model.s_target,
                Columns.offset: self._folded_model.offset * np.ones(n_row),
                Columns.max_en: self._max_en * np.ones(n_row),
                Columns.min_en: self._min_en * np.ones(n_row),
                Columns.relative_offset: self._relative_offset * np.ones(n_row),
                Columns.method: self._method,
            }
        )
