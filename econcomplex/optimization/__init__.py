"""
ECI Optimization (Stojkoski & Hidalgo 2026, Research Policy 55, 105454).

Target-oriented optimization layer for strategic diversification:
calibrate a stepping-stone forecast model, compute the effort required
for new specializations, and select minimal-effort portfolios that reach
an ECI (or growth) target.
"""

from .steppingstone import (
    calibrate_steppingstone,
    effort_matrix,
    forecast_specialization,
)
from .portfolio import eci_optimization
from .growth_target import (
    calibrate_growth_model,
    expected_growth,
    eci_target_for_growth,
)
from .diffusion import (
    proximity_network,
    activation_probabilities,
    calibrate_contagion,
    diversification_strategy,
    expected_diversification_time,
    compare_strategies,
    optimize_sequence,
)

__all__ = [
    "calibrate_steppingstone",
    "effort_matrix",
    "forecast_specialization",
    "eci_optimization",
    "calibrate_growth_model",
    "expected_growth",
    "eci_target_for_growth",
    "proximity_network",
    "activation_probabilities",
    "calibrate_contagion",
    "diversification_strategy",
    "expected_diversification_time",
    "compare_strategies",
    "optimize_sequence",
]
