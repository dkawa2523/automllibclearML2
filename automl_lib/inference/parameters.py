"""Parameter parsing helpers for inference workflows."""

from __future__ import annotations

import itertools
import json
from typing import Any, Dict, List

import pandas as pd


def parse_param_spec(json_path: str) -> List[Dict[str, Any]]:
    """Parse parameter specification JSON file into normalized dictionaries."""

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    variables = data.get("variables")
    if not variables:
        raise ValueError("Parameter specification must contain a 'variables' list")
    parsed: List[Dict[str, Any]] = []
    for var in variables:
        name = var.get("name")
        vtype = var.get("type")
        method = var.get("method")
        if not (name and vtype and method):
            raise ValueError(f"Variable definition incomplete: {var}")
        vtype = str(vtype).lower()
        method = str(method).lower()
        if method == "range":
            if "min" not in var or "max" not in var or "step" not in var:
                raise ValueError(f"Range method requires 'min', 'max' and 'step': {var}")
        elif method == "values":
            if "values" not in var or not isinstance(var["values"], (list, tuple)):
                raise ValueError(f"Values method requires a 'values' list: {var}")
        else:
            raise ValueError(f"Unknown method '{method}' for variable {name}")
        parsed.append({
            "name": name,
            "type": vtype,
            "method": method,
            "min": var.get("min"),
            "max": var.get("max"),
            "step": var.get("step"),
            "values": var.get("values"),
        })
    return parsed


def parse_variables_from_config(vars_spec: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate and normalize variable definitions loaded from YAML config."""

    parsed: List[Dict[str, Any]] = []
    for var in vars_spec:
        name = var.get("name")
        vtype = var.get("type")
        method = var.get("method")
        if not name or not vtype or not method:
            raise ValueError(f"Variable definition incomplete: {var}")
        vtype = str(vtype).lower()
        method = str(method).lower()
        if method == "range":
            if "min" not in var or "max" not in var or "step" not in var:
                raise ValueError(f"Range method requires 'min', 'max' and 'step': {var}")
        elif method == "values":
            vals = var.get("values")
            if vals is None or not isinstance(vals, (list, tuple)):
                raise ValueError(f"Values method requires 'values' list: {var}")
        else:
            raise ValueError(f"Unknown method '{method}' for variable {name}")
        parsed.append({
            "name": name,
            "type": vtype,
            "method": method,
            "min": var.get("min"),
            "max": var.get("max"),
            "step": var.get("step"),
            "values": var.get("values"),
        })
    return parsed


def generate_grid_values(var: Dict[str, Any]) -> List[Any]:
    """Generate a list of candidate values for a variable definition."""

    if var["method"] == "range":
        vmin = var["min"]
        vmax = var["max"]
        step = var["step"]
        if vmin is None or vmax is None or step is None:
            raise ValueError(f"Range variable '{var['name']}' must have min, max and step")
        if var["type"] == "int":
            return list(range(int(vmin), int(vmax) + 1, int(step)))
        n = int((float(vmax) - float(vmin)) / float(step)) + 1
        return [float(vmin) + i * float(step) for i in range(n)]
    return list(var["values"])


def enumerate_parameter_combinations(vars_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute the Cartesian product of all variable values."""

    value_lists: List[List[Any]] = []
    names: List[str] = []
    for var in vars_list:
        values = generate_grid_values(var)
        value_lists.append(values)
        names.append(var["name"])
    combos: List[Dict[str, Any]] = []
    for combination in itertools.product(*value_lists):
        combos.append({name: val for name, val in zip(names, combination)})
    return combos


def convert_combinations_to_df(combos: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert list of parameter combinations to a DataFrame."""

    if not combos:
        return pd.DataFrame()
    return pd.DataFrame(combos)


__all__ = [
    "parse_param_spec",
    "parse_variables_from_config",
    "generate_grid_values",
    "enumerate_parameter_combinations",
    "convert_combinations_to_df",
]
