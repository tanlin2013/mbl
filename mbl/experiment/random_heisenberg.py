import uuid
from time import time

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

        Args:
            n: System size.
            h: Disorder strength.
            seed: Random seed used to initialize the pseudo-random number generator.
            penalty: Penalty strength (or Lagrangian multiplier).
            s_target: The targeting total Sz charge sector.
            offset:
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
        n_row = len(self.ed.evecs)
        return pd.DataFrame(
            {
                Columns.level_id: list(range(n_row)),
                Columns.en: self.evals.tolist(),
                Columns.total_sz: self.total_sz.tolist(),
                Columns.edge_entropy: self.entanglement_entropy(site=0).tolist(),
                Columns.bipartite_entropy: self.entanglement_entropy(
                    site=self.model.n // 2 - 1
                ).tolist(),
                Columns.system_size: [self.model.n] * n_row,
                Columns.disorder: [self.model.h] * n_row,
                Columns.trial_id: [self.model.trial_id] * n_row,
                Columns.seed: [self.model.seed] * n_row,
                Columns.penalty: [self.model.penalty] * n_row,
                Columns.s_target: [self.model.s_target] * n_row,
                Columns.offset: [self.model.offset] * n_row,
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
    ):
        seed = int(time()) if seed is None else seed
        self._overall_const = overall_const
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
        )

    def _mpo_run_method(self) -> MatrixProductOperator:
        return self._overall_const * self._model.mpo

    @check_output(RandomHeisenbergTSDRGSchema.to_schema(), lazy=True)
    def compute_df(self) -> pd.DataFrame:
        n_row = self.tsdrg.chi
        return pd.DataFrame(
            {
                Columns.level_id: list(range(n_row)),
                Columns.en: self.evals.tolist(),
                Columns.variance: self.variance.tolist(),
                Columns.total_sz: self.total_sz.tolist(),
                Columns.edge_entropy: self.edge_entropy.tolist(),
                Columns.truncation_dim: [self.tsdrg.chi] * n_row,
                Columns.system_size: [self.model.n] * n_row,
                Columns.disorder: [self.model.h] * n_row,
                Columns.trial_id: [self.model.trial_id] * n_row,
                Columns.seed: [self.model.seed] * n_row,
                Columns.penalty: [self.model.penalty] * n_row,
                Columns.s_target: [self.model.s_target] * n_row,
                Columns.offset: [self.model.offset] * n_row,
                Columns.overall_const: [self._overall_const] * n_row,
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
        max_en: float = 0,
        min_en: float = 0,
        relative_offset: float = 0,
    ):
        trial_id = uuid.uuid4().hex
        seed = int(time()) if seed is None else seed
        self._folded_model = RandomHeisenberg(
            n=n,
            h=h,
            trial_id=trial_id,
            seed=seed,
            penalty=penalty,
            s_target=s_target,
            offset=(max_en - min_en) * relative_offset,
        )
        self._max_en = max_en
        self._min_en = min_en
        self._relative_offset = relative_offset
        super().__init__(
            model=RandomHeisenberg(n=n, h=h, trial_id=trial_id, seed=seed), chi=chi
        )

    def _mpo_run_method(self) -> MatrixProductOperator:
        return self._folded_model.mpo.square()

    @check_output(RandomHeisenbergFoldingTSDRGSchema.to_schema(), lazy=True)
    def compute_df(self) -> pd.DataFrame:
        n_row = self.tsdrg.chi
        return pd.DataFrame(
            {
                Columns.level_id: list(range(n_row)),
                Columns.en: self.evals.tolist(),
                Columns.variance: self.variance.tolist(),
                Columns.total_sz: self.total_sz.tolist(),
                Columns.edge_entropy: self.edge_entropy.tolist(),
                Columns.truncation_dim: [self.tsdrg.chi] * n_row,
                Columns.system_size: [self.model.n] * n_row,
                Columns.disorder: [self.model.h] * n_row,
                Columns.trial_id: [self.model.trial_id] * n_row,
                Columns.seed: [self.model.seed] * n_row,
                Columns.penalty: [self._folded_model.penalty] * n_row,
                Columns.s_target: [self._folded_model.s_target] * n_row,
                Columns.offset: [self._folded_model.offset] * n_row,
                Columns.max_en: [self._max_en] * n_row,
                Columns.min_en: [self._min_en] * n_row,
                Columns.relative_offset: [self._relative_offset] * n_row,
            }
        )
