from dataclasses import dataclass

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
        return [
            f"({Columns.system_size} = {n})",
            f"({Columns.disorder} = {h})",
            f"({Columns.overall_const} = {overall_const})",
            f"({Columns.truncation_dim} = {chi})",
            f"({Columns.penalty} = {penalty})",
            f"({Columns.s_target} = {s_target})",
            f"({Columns.seed} = {seed})",
        ]

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
    ):
        query_elements = cls.query_elements(
            n=n,
            h=h,
            overall_const=overall_const,
            penalty=penalty,
            s_target=s_target,
            seed=seed,
            chi=chi,
        )
        return wr.athena.read_sql_query(
            f"SELECT {Columns.en} "
            f"FROM {cls.Metadata.table} "
            f"WHERE {' AND '.join(query_elements)} "
            f"ORDER BY {Columns.en}",
            database=cls.Metadata.database,
        )
