import torch
import torch.nn.functional as F


def cosine_sim_pairs(
    values: torch.Tensor,
    *,
    center: bool = False,
    eps: float = 1e-12,
    include_diagonal: bool = False,
    unbiased_std: bool = False,
):
    assert values.dim() == 2, "X must be (N, D)"
    n_points = values.shape[0]
    if n_points < 2:
        return float("nan"), float("nan")

    if center:
        values = values - values.mean(dim=0, keepdim=True)

    normalized = F.normalize(values, p=2, dim=1, eps=eps)
    cosine = normalized @ normalized.T

    if include_diagonal:
        comparisons = cosine.reshape(-1)
    else:
        indices = torch.triu_indices(n_points, n_points, offset=1, device=values.device)
        comparisons = cosine[indices[0], indices[1]]

    mean = comparisons.mean().item()
    std = comparisons.std(unbiased=unbiased_std).item()
    return mean, std


def knn_avg_l2_dist(
    values: torch.Tensor,
    k: int = 5,
    center: bool = False,
    eps: float = 0.0,
) -> torch.Tensor:
    assert values.dim() == 2
    n_points = values.shape[0]
    if n_points < 2:
        return torch.full((k,), float("nan"), dtype=values.dtype, device=values.device)

    if center:
        values = values - values.mean(dim=0, keepdim=True)

    distances = torch.cdist(values, values)
    inf = torch.tensor(float("inf"), dtype=distances.dtype, device=distances.device)
    distances.fill_diagonal_(inf)

    knn_values, _ = torch.topk(distances, min(k, n_points - 1), dim=1, largest=False, sorted=True)
    if knn_values.shape[1] < k:
        padding = torch.full(
            (n_points, k - knn_values.shape[1]),
            float("nan"),
            dtype=knn_values.dtype,
            device=knn_values.device,
        )
        knn_values = torch.cat((knn_values, padding), dim=1)

    return torch.nanmean(knn_values, dim=0)
