from __future__ import annotations

from pathlib import Path

import numpy as np

from rethinking_neural_id.artifacts import load_metric_json, vit_metric_path
from rethinking_neural_id.paths import RepoPaths

DEFAULT_VIT_MODELS = ("vit-base", "dinov3-vitb16", "dinov3-vitl16")

MODEL_COLORS = {
    "vit-base": "#0072B2",
    "dinov3-vitb16": "#D55E00",
    "dinov3-vitl16": "#009E73",
}


def vit_figs_dir(paths: RepoPaths | None = None) -> Path:
    repo_paths = paths or RepoPaths.default()
    target = repo_paths.vit_figs_root
    target.mkdir(parents=True, exist_ok=True)
    return target


def layer_slice(*, exclude_first: bool = True, exclude_last: bool = False) -> slice:
    start = 1 if exclude_first else 0
    stop = -1 if exclude_last else None
    return slice(start, stop)


def relative_depth(length: int) -> np.ndarray:
    index = np.arange(length)
    return index / (length - 1 if length > 1 else 1.0)


def load_vit_metric(
    method: str,
    model: str,
    *,
    dataset: str = "imagenet7",
    category: str = "mix",
    paths: RepoPaths | None = None,
) -> dict:
    repo_paths = paths or RepoPaths.default()
    path = vit_metric_path(repo_paths, method, dataset, category, model)
    if not path.exists():
        raise FileNotFoundError(f"Missing {method} metric file for model '{model}': {path}")
    return load_metric_json(path)


def as_float_array(payload: dict, key: str, *, sl: slice | None = None) -> np.ndarray:
    values = payload[key]
    if sl is not None:
        values = values[sl]
    return np.asarray(values, dtype=float)


def as_optional_float_array(payload: dict, key: str, *, sl: slice | None = None) -> np.ndarray:
    values = payload[key]
    if sl is not None:
        values = values[sl]
    return np.asarray([np.nan if value is None else value for value in values], dtype=float)


def as_float_matrix(payload: dict, key: str, *, sl: slice | None = None) -> np.ndarray:
    rows = payload[key]
    if sl is not None:
        rows = rows[sl]
    rows = [row for row in rows if row is not None]
    matrix = np.asarray(rows, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"{key} must be convertible to a 2D array after removing missing layers.")
    return matrix


def load_gride(
    model: str,
    *,
    dataset: str = "imagenet7",
    category: str = "mix",
    paths: RepoPaths | None = None,
    sl: slice | None = None,
) -> dict[str, np.ndarray]:
    payload = load_vit_metric("gride", model, dataset=dataset, category=category, paths=paths)
    return {
        "id_ls": as_float_matrix(payload, "id_ls", sl=sl),
        "err_ls": as_float_matrix(payload, "err_ls", sl=sl),
        "r_ls": as_float_matrix(payload, "r_ls", sl=sl),
        "id": as_optional_float_array(payload, "id", sl=sl),
        "err": as_optional_float_array(payload, "err", sl=sl),
        "r": as_optional_float_array(payload, "r", sl=sl),
    }


def load_entropy(
    model: str,
    *,
    dataset: str = "imagenet7",
    category: str = "mix",
    paths: RepoPaths | None = None,
    sl: slice | None = None,
) -> np.ndarray:
    payload = load_vit_metric("entropy", model, dataset=dataset, category=category, paths=paths)
    return as_float_array(payload, "entropy", sl=sl)


def load_avg_l2(
    model: str,
    *,
    dataset: str = "imagenet7",
    category: str = "mix",
    paths: RepoPaths | None = None,
    sl: slice | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    payload = load_vit_metric("avg_l2", model, dataset=dataset, category=category, paths=paths)
    return as_float_array(payload, "mean", sl=sl), as_float_array(payload, "std", sl=sl)


def load_avg_cosine(
    model: str,
    *,
    dataset: str = "imagenet7",
    category: str = "mix",
    paths: RepoPaths | None = None,
    sl: slice | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    payload = load_vit_metric("avg_cosine", model, dataset=dataset, category=category, paths=paths)
    mean = as_float_array(payload, "mean", sl=sl)
    std = np.sqrt(as_float_array(payload, "var", sl=sl))
    return mean, std
