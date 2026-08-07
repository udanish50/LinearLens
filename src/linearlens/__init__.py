"""Linear Lens: non-interventional neural-network interpretability utilities."""

from .analysis import ComponentReport, analyze_linear_component
from .entropy import RoleThresholds, classify_first_layer, classify_hidden_layer
from .influence import influence_distribution, mean_absolute_influence
from .qsm import QSMConfig, qualitative_symbolic_matrix

__all__ = [
    "ComponentReport",
    "QSMConfig",
    "RoleThresholds",
    "analyze_linear_component",
    "classify_first_layer",
    "classify_hidden_layer",
    "influence_distribution",
    "mean_absolute_influence",
    "qualitative_symbolic_matrix",
]

__version__ = "0.1.0"
