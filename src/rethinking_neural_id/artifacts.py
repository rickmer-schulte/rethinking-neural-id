from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from .paths import RepoPaths
from .registry import get_llm_model_spec, get_vit_model_spec


def ensure_parent(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _as_representation_rows(values: Any) -> list[np.ndarray]:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError("Representation arrays must have shape (N, D).")
    return [np.asarray(row) for row in array]


def load_representation_dict(path: str | Path) -> dict[int, np.ndarray]:
    with open(path, "rb") as handle:
        payload = pickle.load(handle)
    return {int(layer): np.asarray(values) for layer, values in payload.items()}


def save_representation_dict(path: str | Path, representations: dict[int, Any]) -> Path:
    target = ensure_parent(path)
    payload = {
        int(layer): _as_representation_rows(values)
        for layer, values in representations.items()
    }
    with open(target, "wb") as handle:
        pickle.dump(payload, handle)
    return target


def load_metric_json(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_metric_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = ensure_parent(path)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return target


def save_numpy_array(path: str | Path, array: Any) -> Path:
    target = ensure_parent(path)
    np.save(target, array)
    return target


def llm_representation_path(paths: RepoPaths, model_key: str, dataset: str, shard: str) -> Path:
    spec = get_llm_model_spec(model_key)
    return (
        paths.reps_root
        / spec.rep_dir
        / dataset
        / f"hidden_{spec.result_key}_{dataset}_shard_{shard}.pickle"
    )


def llm_metric_path(
    paths: RepoPaths,
    method: str,
    dataset: str,
    shard: str,
    model_key: str,
    step: int | None = None,
) -> Path:
    spec = get_llm_model_spec(model_key)
    suffix = f"_step{step}" if step is not None else ""
    return (
        paths.llm_metrics_root
        / method
        / f"hidden_{dataset}_{shard}_{spec.result_key}{suffix}.json"
    )


def vit_representation_path(paths: RepoPaths, model_key: str, dataset: str, category: str) -> Path:
    spec = get_vit_model_spec(model_key)
    return (
        paths.reps_root
        / spec.rep_dir
        / dataset
        / f"hidden_{spec.result_key}_{dataset}-{category}.pickle"
    )


def vit_metric_path(
    paths: RepoPaths,
    method: str,
    dataset: str,
    category: str,
    model_key: str,
    step: int | None = None,
) -> Path:
    spec = get_vit_model_spec(model_key)
    suffix = f"_step{step}" if step is not None else ""
    return (
        paths.vit_metrics_root
        / method
        / f"hidden_{dataset}-{category}_{spec.result_key}{suffix}.json"
    )


def cnn_metrics_dir(paths: RepoPaths, trained: bool = True) -> Path:
    if trained:
        return paths.cnn_metrics_root
    return paths.cnn_results_root / "metrics_untrained"


def cnn_metric_path(paths: RepoPaths, arch: str, metric_name: str, trained: bool = True) -> Path:
    return cnn_metrics_dir(paths, trained=trained) / f"{arch}_{metric_name}.npy"


def infer_vit_representation_metadata(path: str | Path) -> dict[str, str]:
    stem = Path(path).stem
    parts = stem.split("_", 2)
    if len(parts) != 3 or parts[0] != "hidden":
        raise ValueError(f"Cannot infer ViT metadata from path: {path}")

    model_key = parts[1]
    if "-" not in parts[2]:
        raise ValueError(f"Cannot infer dataset/category from path: {path}")

    dataset, category = parts[2].rsplit("-", 1)
    return {
        "model_key": model_key,
        "dataset": dataset,
        "category": category,
    }
