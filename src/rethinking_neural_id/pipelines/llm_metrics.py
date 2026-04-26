from __future__ import annotations

from pathlib import Path
from typing import Any

from rethinking_neural_id.artifacts import (
    llm_metric_path,
    llm_representation_path,
    load_representation_dict,
    save_metric_json,
)
from rethinking_neural_id.metrics.layerwise import run_layerwise_metrics
from rethinking_neural_id.paths import RepoPaths


def compute_llm_metrics(
    *,
    model: str,
    dataset: str,
    shard: str,
    method: str,
    random_seed: int = 32,
    step: int | None = None,
    k: int = 5,
    gride_k_max: int = 64,
    block_size: int = 1000,
    paths: RepoPaths | None = None,
) -> tuple[Path, dict[str, Any]]:
    repo_paths = paths or RepoPaths.default()
    reps_path = llm_representation_path(repo_paths, model, dataset, shard)
    reps = load_representation_dict(reps_path)
    results = run_layerwise_metrics(
        reps,
        method,
        random_seed=random_seed,
        step=step,
        k=k,
        gride_k_max=gride_k_max,
        block_size=block_size,
    )
    output_path = llm_metric_path(repo_paths, method, dataset, shard, model, step=step)
    save_metric_json(output_path, results)
    return output_path, results
