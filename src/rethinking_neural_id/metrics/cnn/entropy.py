import torch


def est_entropy(values, center=True, eff_rank=False):
    if center:
        values = values - values.mean(dim=0, keepdims=True)
    gram = values @ values.T
    gram = normalize_gram(gram)
    entropy = von_neumann_entropy(gram)
    if eff_rank:
        entropy = torch.exp(entropy)
    return entropy


def normalize_gram(values):
    return values / torch.trace(values)

# Function below adapted from: 
# https://github.com/uk-cliplab/representation-itl/blob/main/src/repitl/matrix_itl.py

def von_neumann_entropy(values, low_rank=False, rank=None):
    n_points = values.shape[0]
    eigenvalues, _ = torch.linalg.eigh(values)
    if low_rank:
        low_rank_eigenvalues = torch.zeros_like(eigenvalues)
        low_rank_eigenvalues[-rank:] = eigenvalues[-rank:]
        remainder = eigenvalues.sum() - low_rank_eigenvalues.sum()
        low_rank_eigenvalues[: (n_points - rank)] = remainder / (n_points - rank)
        mask = torch.gt(low_rank_eigenvalues, 0.0)
        positive = low_rank_eigenvalues[mask]
    else:
        mask = torch.gt(eigenvalues, 0.0)
        positive = eigenvalues[mask]

    positive = positive / positive.sum()
    return -1 * torch.sum(positive * torch.log(positive))
