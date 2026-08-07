from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QSMConfig:
    """Symbol bands based on the article's approximate 60/30/10% examples."""

    very_strong: float = 0.60
    strong: float = 0.30
    moderate: float = 0.10
    symbol_very_strong: str = "★★★"
    symbol_strong: str = "★★"
    symbol_moderate: str = "★"
    symbol_weak: str = "◦"


def influence_symbol(value: float, config: QSMConfig | None = None) -> str:
    config = config or QSMConfig()
    if value >= config.very_strong:
        return config.symbol_very_strong
    if value >= config.strong:
        return config.symbol_strong
    if value >= config.moderate:
        return config.symbol_moderate
    return config.symbol_weak


def qualitative_symbolic_matrix(
    distribution: np.ndarray,
    config: QSMConfig | None = None,
) -> np.ndarray:
    p = np.asarray(distribution, dtype=np.float64)
    config = config or QSMConfig()
    if p.ndim != 2:
        raise ValueError("distribution must be 2D [neurons, features]")
    vectorized = np.vectorize(lambda value: influence_symbol(float(value), config), otypes=[object])
    return vectorized(p)


def qsm_to_markdown(
    symbols: np.ndarray,
    feature_names: list[str] | tuple[str, ...],
    roles: np.ndarray | None = None,
) -> str:
    matrix = np.asarray(symbols, dtype=object)
    if matrix.ndim != 2:
        raise ValueError("symbols must be 2D")
    if len(feature_names) != matrix.shape[1]:
        raise ValueError("feature_names length must match matrix columns")
    if roles is not None and len(roles) != matrix.shape[0]:
        raise ValueError("roles length must match matrix rows")
    header = ["Neuron"] + list(feature_names) + (["Role"] if roles is not None else [])
    rows = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for i, row in enumerate(matrix):
        values = [f"n{i}"] + [str(v) for v in row]
        if roles is not None:
            values.append(str(roles[i]))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)
