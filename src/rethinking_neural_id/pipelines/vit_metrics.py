from __future__ import annotations

from pathlib import Path
from typing import Any

from rethinking_neural_id.artifacts import (
    infer_vit_representation_metadata,
    load_representation_dict,
    save_metric_json,
    vit_metric_path,
    vit_representation_path,
)
from rethinking_neural_id.metrics.layerwise import run_layerwise_metrics
from rethinking_neural_id.paths import RepoPaths


def compute_vit_metrics(
    *,
    pickle_path: str | Path | None = None,
    model: str | None = None,
    dataset: str | None = None,
    category: str | None = None,
    method: str,
    random_seed: int = 32,
    step: int | None = None,
    k: int = 5,
    gride_k_max: int = 64,
    block_size: int = 1000,
    paths: RepoPaths | None = None,
) -> tuple[Path, dict[str, Any]]:
    repo_paths = paths or RepoPaths.default()

    if pickle_path is None:
        if model is None or dataset is None or category is None:
            raise ValueError("Provide either pickle_path or model+dataset+category.")
        resolved_pickle = vit_representation_path(repo_paths, model, dataset, category)
        metadata = {"model_key": model, "dataset": dataset, "category": category}
    else:
        resolved_pickle = Path(pickle_path)
        metadata = infer_vit_representation_metadata(resolved_pickle)

    reps = load_representation_dict(resolved_pickle)
    results = run_layerwise_metrics(
        reps,
        method,
        random_seed=random_seed,
        step=step,
        k=k,
        gride_k_max=gride_k_max,
        block_size=block_size,
    )
    output_path = vit_metric_path(
        repo_paths,
        method,
        metadata["dataset"],
        metadata["category"],
        metadata["model_key"],
        step=step,
    )
    save_metric_json(output_path, results)
    return output_path, results
