from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neighbors import NearestNeighbors

try:
    from dadapy.data import Data
except ImportError: 
    Data = None


def _to_float32(values: np.ndarray) -> np.ndarray:
    if values.dtype != np.float32:
        return values.astype(np.float32, copy=False)
    return values


def _normalize_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    values = _to_float32(values)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return values / norms


def mean_var_pairwise_cosine(values: np.ndarray, block_size: int = 1000) -> tuple[float, float]:
    values = _normalize_rows(values)
    n_points = values.shape[0]
    if n_points < 2:
        return float("nan"), float("nan")

    total_pairs = n_points * (n_points - 1) // 2
    sum_values = 0.0
    sum_values_sq = 0.0

    for start in range(0, n_points, block_size):
        block_i = values[start : start + block_size]
        gram_ii = block_i @ block_i.T
        upper_indices = np.triu_indices(gram_ii.shape[0], k=1)
        on_diag = gram_ii[upper_indices]
        sum_values += float(on_diag.sum())
        sum_values_sq += float(np.square(on_diag).sum())

        other = start + block_size
        while other < n_points:
            block_j = values[other : other + block_size]
            gram_ij = block_i @ block_j.T
            sum_values += float(gram_ij.sum())
            sum_values_sq += float(np.square(gram_ij).sum())
            other += block_size

    mean = sum_values / total_pairs
    variance = sum_values_sq / total_pairs - mean * mean
    return mean, max(variance, 0.0)


def knn_avg_distances(
    values: np.ndarray,
    k: int = 5,
    metric: str = "euclidean",
    algorithm: str = "auto",
    leaf_size: int = 40,
    n_jobs: int = -1,
) -> dict[str, list[float]]:
    values = _to_float32(values)
    n_points = values.shape[0]
    if n_points <= 1:
        return {"means": [float("nan")] * k, "vars": [float("nan")] * k}

    if metric not in {"euclidean", "cosine"}:
        raise ValueError("metric must be 'euclidean' or 'cosine'")

    n_neighbors = min(k + 1, n_points)
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        algorithm=algorithm,
        metric=metric,
        leaf_size=leaf_size,
        n_jobs=n_jobs,
    )
    nn.fit(values if metric == "euclidean" else _normalize_rows(values))
    distances, _ = nn.kneighbors(values, return_distance=True)
    distances = distances[:, 1:n_neighbors]

    if distances.shape[1] < k:
        pad_width = k - distances.shape[1]
        distances = np.pad(distances, ((0, 0), (0, pad_width)), constant_values=np.nan)

    means = np.nanmean(distances, axis=0).astype(np.float64).tolist()
    variances = np.nanvar(distances, axis=0).astype(np.float64).tolist()
    return {
        "means": [float(value) for value in means],
        "vars": [float(value) for value in variances],
    }


def entropy(values: np.ndarray, center: bool = True, eff_rank: bool = False, method: str = "cov") -> float:
    values = np.asarray(values, dtype=np.float32)
    if center:
        values = values - values.mean(axis=0, keepdims=True)

    if method == "cov":
        covariance = (values.T @ values) / max(1, values.shape[0] - 1)
        eigenvalues = np.linalg.eigvalsh(covariance)
        eigenvalues = np.clip(eigenvalues, 0.0, None)
    elif method == "svd":
        singular_values = np.linalg.svd(values, compute_uv=False)
        eigenvalues = singular_values * singular_values
    else:
        raise ValueError("method must be 'cov' or 'svd'")

    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0

    probabilities = eigenvalues / total
    score = float(-np.sum(probabilities * np.log(probabilities + 1e-12)))
    if eff_rank:
        return float(np.exp(score))
    return score


def _require_dadapy() -> None:
    if Data is None:
        raise RuntimeError(
            "dadapy is required for 'twonn' and 'gride'. Install the project with the 'metrics' extra."
        )


def _unique_data(layer_values: np.ndarray, min_points: int) -> tuple[Any, bool]:
    _require_dadapy()
    data_obj = Data(layer_values)
    data_obj.remove_identical_points()
    if data_obj.N < min_points:
        return data_obj, False
    return data_obj, True


def _normalize_representations(representations: dict[int, Any]) -> tuple[dict[int, np.ndarray], list[int], int]:
    normalized = {int(layer): np.asarray(values) for layer, values in representations.items()}
    layer_indices = sorted(normalized)
    n_layers = layer_indices[-1] + 1 if layer_indices else 0
    return normalized, layer_indices, n_layers


def run_layerwise_metrics(
    representations: dict[int, Any],
    method: str,
    *,
    random_seed: int = 32,
    step: int | None = None,
    k: int = 5,
    gride_k_max: int = 64,
    block_size: int = 1000,
) -> dict[str, Any]:
    np.random.seed(random_seed)

    reps, layer_indices, n_layers = _normalize_representations(representations)

    if method == "twonn":
        results = {
            "id": [None] * n_layers,
            "err": [None] * n_layers,
            "r": [None] * n_layers,
        }
        for layer in layer_indices:
            data_obj, ok = _unique_data(reps[layer], min_points=3)
            if not ok:
                print(f"Layer: {layer}, TwoNN ID skipped (unique points < 3)")
                continue
            data_obj.compute_distances(maxk=2)
            value_id, value_err, value_r = data_obj.compute_id_2NN()
            results["id"][layer] = float(value_id) if value_id is not None else None
            results["err"][layer] = float(value_err) if value_err is not None else None
            results["r"][layer] = float(value_r) if value_r is not None else None
            print(
                f"Layer: {layer}, TwoNN ID: id={results['id'][layer]}, "
                f"err={results['err'][layer]}, r={results['r'][layer]}"
            )
        return results

    if method == "entropy":
        results = {"entropy": [None] * n_layers}
        for layer in layer_indices:
            score = entropy(reps[layer])
            results["entropy"][layer] = float(score)
            print(f"Layer: {layer}, Estimated Entropy: {results['entropy'][layer]:.6f}")
        return results

    if method == "gride":
        results = {
            "id_ls": [None] * n_layers,
            "err_ls": [None] * n_layers,
            "r_ls": [None] * n_layers,
            "id": [None] * n_layers,
            "err": [None] * n_layers,
            "r": [None] * n_layers,
        }
        for layer in layer_indices:
            data_obj, ok = _unique_data(reps[layer], min_points=5)
            if not ok:
                print(f"Layer: {layer}, GRIDE skipped (unique points < 5)")
                continue

            maxk = min(gride_k_max, data_obj.N - 1)
            if maxk < 2:
                print(f"Layer: {layer}, GRIDE skipped (maxk < 2)")
                continue

            data_obj.compute_distances(maxk=maxk)
            id_list, err_list, r_list = data_obj.return_id_scaling_gride(range_max=maxk)

            result_id_ls = [float(value) for value in id_list]
            result_err_ls = [float(value) for value in err_list]
            result_r_ls = [float(value) for value in r_list]
            results["id_ls"][layer] = result_id_ls
            results["err_ls"][layer] = result_err_ls
            results["r_ls"][layer] = result_r_ls

            if step is None:
                results["id"][layer] = float(np.nanmean(result_id_ls))
                results["err"][layer] = float(np.nanmean(result_err_ls))
                results["r"][layer] = float(np.nanmean(result_r_ls))
            else:
                index = step if 0 <= step < len(result_id_ls) else -1
                results["id"][layer] = float(result_id_ls[index])
                results["err"][layer] = float(result_err_ls[index])
                results["r"][layer] = float(result_r_ls[index])
            print(
                f"Layer: {layer}, GRIDE ID: id={results['id'][layer]:.6f}, "
                f"err={results['err'][layer]:.6f}, r={results['r'][layer]:.6f}"
            )
        return results

    if method == "avg_l2":
        results = {"mean": [None] * n_layers, "std": [None] * n_layers}
        for layer in layer_indices:
            norms = np.linalg.norm(reps[layer].astype(np.float64), axis=1, ord=2)
            results["mean"][layer] = float(np.mean(norms, dtype=np.float64))
            results["std"][layer] = float(np.std(norms, dtype=np.float64))
            print(
                f"Layer: {layer}, Avg L2 norm: mean={results['mean'][layer]:.6f}, "
                f"std={results['std'][layer]:.6f}"
            )
        return results

    if method == "avg_cosine":
        results = {"mean": [None] * n_layers, "var": [None] * n_layers}
        for layer in layer_indices:
            mean, variance = mean_var_pairwise_cosine(reps[layer], block_size=block_size)
            results["mean"][layer] = float(mean)
            results["var"][layer] = float(variance)
            print(
                f"Layer: {layer}, Avg cosine similarity: "
                f"mean={results['mean'][layer]:.6f}, var={results['var'][layer]:.6f}"
            )
        return results

    if method == "knn":
        results = {"k": int(k), "means": [None] * n_layers, "vars": [None] * n_layers}
        for layer in layer_indices:
            output = knn_avg_distances(reps[layer], k=k, metric="euclidean")
            results["means"][layer] = [float(value) for value in output["means"]]
            results["vars"][layer] = [float(value) for value in output["vars"]]
            print(
                f"Layer: {layer}, kNN distances (k=1..{k}): "
                f"means={results['means'][layer]}, vars={results['vars'][layer]}"
            )
        return results

    raise ValueError(f"Unknown method: {method}")
