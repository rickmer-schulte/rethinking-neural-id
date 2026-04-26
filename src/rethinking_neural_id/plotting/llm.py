from __future__ import annotations

from pathlib import Path

import numpy as np

from rethinking_neural_id.artifacts import llm_metric_path, load_metric_json
from rethinking_neural_id.paths import RepoPaths

DEFAULT_LLM_MODELS = ("llama", "mistral", "pythia")

MODEL_COLORS = {
    "llama": "#1f77b4",
    "mistral": "#d62728",
    "pythia": "#2ca02c",
}


def llm_figs_dir(paths: RepoPaths | None = None) -> Path:
    repo_paths = paths or RepoPaths.default()
    target = repo_paths.llm_figs_root
    target.mkdir(parents=True, exist_ok=True)
    return target


def layer_slice(*, exclude_first: bool = True, exclude_last: bool = False) -> slice:
    start = 1 if exclude_first else 0
    stop = -1 if exclude_last else None
    return slice(start, stop)


def load_llm_metric(
    method: str,
    model: str,
    *,
    dataset: str = "wikitext",
    shard: str = "aa",
    paths: RepoPaths | None = None,
) -> dict:
    repo_paths = paths or RepoPaths.default()
    path = llm_metric_path(repo_paths, method, dataset, shard, model)
    if not path.exists():
        raise FileNotFoundError(f"Missing {method} metric file for model '{model}': {path}")
    return load_metric_json(path)


def as_float_array(payload: dict, key: str, *, sl: slice | None = None) -> np.ndarray:
    if sl is not None:
        return np.asarray(payload[key][sl], dtype=float)
    return np.asarray(payload[key], dtype=float)


def load_gride(
    model: str,
    *,
    dataset: str = "wikitext",
    shard: str = "aa",
    paths: RepoPaths | None = None,
    sl: slice | None = None,
) -> dict[str, np.ndarray | None]:
    payload = load_llm_metric("gride", model, dataset=dataset, shard=shard, paths=paths)
    result: dict[str, np.ndarray | None] = {
        "id_ls": as_float_array(payload, "id_ls", sl=sl),
        "id": as_float_array(payload, "id", sl=sl),
        "r_ls": as_float_array(payload, "r_ls", sl=sl),
        "r": as_float_array(payload, "r", sl=sl),
        "err_ls": as_float_array(payload, "err_ls", sl=sl) if "err_ls" in payload else None,
    }
    return result


def load_entropy(
    model: str,
    *,
    dataset: str = "wikitext",
    shard: str = "aa",
    paths: RepoPaths | None = None,
    sl: slice | None = None,
) -> np.ndarray:
    payload = load_llm_metric("entropy", model, dataset=dataset, shard=shard, paths=paths)
    return as_float_array(payload, "entropy", sl=sl)


def load_avg_l2(
    model: str,
    *,
    dataset: str = "wikitext",
    shard: str = "aa",
    paths: RepoPaths | None = None,
    sl: slice | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    payload = load_llm_metric("avg_l2", model, dataset=dataset, shard=shard, paths=paths)
    return as_float_array(payload, "mean", sl=sl), as_float_array(payload, "std", sl=sl)


def load_avg_cosine(
    model: str,
    *,
    dataset: str = "wikitext",
    shard: str = "aa",
    paths: RepoPaths | None = None,
    sl: slice | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    payload = load_llm_metric("avg_cosine", model, dataset=dataset, shard=shard, paths=paths)
    mean = as_float_array(payload, "mean", sl=sl)
    std = np.sqrt(as_float_array(payload, "var", sl=sl))
    return mean, std
