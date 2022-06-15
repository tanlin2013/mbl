from dataclasses import dataclass


@dataclass(frozen=True)
class Columns:
    level_id: str = "level_id"
    en: str = "en"
    variance: str = "variance"
    total_sz: str = "total_sz"
    edge_entropy: str = "edge_entropy"
    bipartite_entropy: str = "bipartite_entropy"
    truncation_dim: str = "truncation_dim"
    system_size: str = "system_size"
    disorder: str = "disorder"
    trial_id: str = "trial_id"
    seed: str = "seed"
    penalty: str = "penalty"
    s_target: str = "s_target"
    offset: str = "offset"
    energy_gap: str = "energy_gap"
    gap_ratio: str = "gap_ratio"
    overall_const: str = "overall_const"
    max_en: str = "max_en"
    min_en: str = "min_en"
    relative_offset: str = "relative_offset"
    method: str = "method"
