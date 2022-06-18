from dataclasses import dataclass
from typing import Dict

from boto3 import Session
import awswrangler as wr

from mbl.name_space import Columns


class EnergyBounds:
    @dataclass
    class Metadata:
        database: str = "random_heisenberg"
        table: str = "tsdrg"

    @classmethod
    def query_elements(
        cls,
        n: int,
        h: float,
        overall_const: float = 1,
        penalty: float = 0.0,
        s_target: int = 0,
        seed: int = None,
        chi: int = None,
    ):
        query = [
            f"({Columns.system_size} = {n})",
            f"({Columns.disorder} = {h})",
            f"({Columns.overall_const} = {overall_const})",
            f"({Columns.truncation_dim} = {chi})",
            f"({Columns.penalty} = {penalty})",
            f"({Columns.s_target} = {s_target})",
        ]
        if seed is not None:
            query.append(f"({Columns.seed} = {seed})")
        return query

    @classmethod
    def athena_query(
        cls,
        n: int,
        h: float,
        overall_const: float = 1,
        penalty: float = 0.0,
        s_target: int = 0,
        seed: int = None,
        chi: int = None,
        method: str = None,
        boto3_session: Session = None,
        **kwargs,
    ) -> float:
        query_elements = cls.query_elements(
            n=n,
            h=h,
            overall_const=overall_const,
            penalty=penalty,
            s_target=s_target,
            seed=seed,
            chi=chi,
        )
        key = getattr(Columns, f"{method}_en")
        return wr.athena.read_sql_query(
            f"""
            SELECT {method.upper()}({Columns.en}) as {key}
            FROM {cls.Metadata.table}
            WHERE {' AND '.join(query_elements)}
            LIMIT 1
            """,
            database=cls.Metadata.database,
            boto3_session=boto3_session,
            **kwargs,
        )[key][0]

    @classmethod
    def retrieve(
        cls,
        n: int,
        h: float,
        overall_const: float = 1,
        penalty: float = 0.0,
        s_target: int = 0,
        seed: int = None,
        chi: int = None,
        boto3_session: Session = None,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Naive setup for energy bounds without considering
        the effect of truncation errors and finite-size effect.

        Args:
            n:
            h:
            overall_const:
            penalty:
            s_target:
            seed:
            chi:
            boto3_session:
            **kwargs: Additional kwargs passed to athena sql query.

        Returns:

        Notes:
            The upside down spectrum (for which we apply -1 to the Hamiltonian) saved on
            AWS Athena doesn't restore the original spectrum up to an overall constant.
        """
        return {
            k: cls.athena_query(
                n=n,
                h=h,
                overall_const=overall_const,
                penalty=penalty,
                s_target=s_target,
                seed=seed,
                chi=chi,
                method=v,
                boto3_session=boto3_session,
                **kwargs,
            )
            for k, v in [(Columns.max_en, "max"), (Columns.min_en, "min")]
        }
