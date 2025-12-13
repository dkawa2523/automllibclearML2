"""Inference utility subpackage used by the CLI script."""

from .model_utils import load_models
from .parameters import (
    parse_param_spec,
    parse_variables_from_config,
    generate_grid_values,
    enumerate_parameter_combinations,
    convert_combinations_to_df,
)
from .search import run_grid_search, run_optimization
from .results import save_results, save_matrices_and_scores
from .plots import (
    plot_predictions,
    plot_correlation_heatmap,
    plot_agreement_heatmap,
    plot_consistency_bars,
)
from .metrics import compute_consistency_scores

__all__ = [
    "load_models",
    "parse_param_spec",
    "parse_variables_from_config",
    "generate_grid_values",
    "enumerate_parameter_combinations",
    "convert_combinations_to_df",
    "run_grid_search",
    "run_optimization",
    "save_results",
    "save_matrices_and_scores",
    "plot_predictions",
    "plot_correlation_heatmap",
    "plot_agreement_heatmap",
    "plot_consistency_bars",
    "compute_consistency_scores",
]
