# Code adapted from:
# https://github.com/ansuini/IntrinsicDimDeep/blob/master/scripts/pretrained/hunchback.py

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

from rethinking_neural_id.artifacts import cnn_metric_path, save_numpy_array
from rethinking_neural_id.metrics.cnn import cosine_sim_pairs, est_entropy, estimate, knn_avg_l2_dist
from rethinking_neural_id.paths import RepoPaths
from rethinking_neural_id.registry import CNN_ARCHS, get_image_categories, get_image_dataset_dir


def _pick_device(torch):
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _load_model_by_arch(torchvision, arch_name: str, trained: bool):
    constructor = getattr(torchvision.models, arch_name)
    if trained:
        try:
            weights_attr = getattr(torchvision.models, f"{arch_name}_Weights", None)
            default_weights = weights_attr.DEFAULT if weights_attr is not None else None
            if default_weights is not None:
                return constructor(weights=default_weights)
            return constructor(weights="IMAGENET1K_V1")
        except Exception:
            return constructor(pretrained=True)

    try:
        return constructor(weights=None)
    except Exception:
        return constructor(pretrained=False)


def _get_depths(model):
    count = 0
    modules = ["input"]
    names = ["input"]
    depths = [0]

    for module in model.features:
        name = module.__class__.__name__
        if "Conv2d" in name or "Linear" in name:
            count += 1
        if "MaxPool2d" in name:
            modules.append(module)
            depths.append(count)
            names.append("MaxPool2d")

    for module in model.classifier:
        if "Linear" in module.__class__.__name__:
            modules.append(module)
            count += 1
            depths.append(count + 1)
            names.append("Linear")

    return modules, names, np.asarray(depths)


def _get_layer_depth(layer):
    count = 0
    for module in layer:
        for child in module.children():
            if "Conv" in child.__class__.__name__:
                count += 1
    return count


def _get_resnet_depths(model):
    modules = ["input", model.maxpool]
    names = ["input", "maxpool"]
    depths = [0, 1]
    count = 1

    for layer_name in ("layer1", "layer2", "layer3", "layer4"):
        layer = getattr(model, layer_name)
        count += _get_layer_depth(layer)
        modules.append(layer)
        names.append(layer_name)
        depths.append(count)

    count += 1
    modules.append(model.avgpool)
    names.append("avgpool")
    depths.append(count)

    count += 1
    modules.append(model.fc)
    names.append("fc")
    depths.append(count)
    return modules, names, np.asarray(depths)


def compute_cnn_metrics(
    *,
    arch: str,
    nsamples: int = 500,
    batch_size: int = 16,
    k_nn: int = 5,
    resamples: int = 5,
    trained: bool = True,
    dataset: str = "imagenet7",
    paths: RepoPaths | None = None,
):
    import torch
    import torchvision
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    if arch not in CNN_ARCHS:
        raise ValueError(f"Unknown CNN architecture: {arch}")

    repo_paths = paths or RepoPaths.default()
    image_root = repo_paths.raw_images_root / get_image_dataset_dir(dataset)
    category_tags = get_image_categories(dataset)
    n_objects = max(len(category_tags) - 1, 0)

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(999)
    try:
        torch.set_float32_matmul_precision("medium")
    except Exception:
        pass

    print("Instantiating model:", arch, "trained" if trained else "untrained")
    model = _load_model_by_arch(torchvision, arch, trained)
    device = _pick_device(torch)
    model = model.to(device)
    model.eval()
    print(f"Training mode: {model.training}")

    if "resnet" in arch:
        modules, names, depths = _get_resnet_depths(model)
    else:
        modules, names, depths = _get_depths(model)

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    data_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    n_layers = len(modules)
    n_imgs = int(np.floor(nsamples * 0.9))
    id_values = np.zeros((n_objects + 1, n_layers))
    id_errors = np.zeros((n_objects + 1, n_layers))
    entropy_values = np.zeros((n_objects + 1, n_layers))
    entropy_errors = np.zeros((n_objects + 1, n_layers))
    l2_values = np.zeros((n_objects + 1, n_layers))
    l2_errors = np.zeros((n_objects + 1, n_layers))
    cosine_values = np.zeros((n_objects + 1, n_layers))
    cosine_errors = np.zeros((n_objects + 1, n_layers))
    knn_values = np.zeros((n_objects + 1, n_layers, k_nn))
    knn_errors = np.zeros((n_objects + 1, n_layers, k_nn))
    ambient_dimensions = []

    for category_index, tag in enumerate(category_tags):
        data_folder = image_root / tag
        image_dataset = datasets.ImageFolder(str(data_folder), data_transforms)
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        for layer_index, module in tqdm(enumerate(modules), total=n_layers):
            print(f"Processing category {category_index}: layer ({layer_index}/{n_layers})")
            output_tensor = None

            for batch_index, batch in enumerate(dataloader):
                if batch_index * batch_size > nsamples:
                    break

                inputs, _targets = batch
                if module == "input":
                    activations = inputs
                else:
                    hook_outputs = []

                    def hook(_module, _hook_input, hook_output):
                        hook_outputs.append(hook_output)

                    handle = module.register_forward_hook(hook)
                    with torch.no_grad():
                        _ = model(inputs.to(device))
                    handle.remove()
                    activations = hook_outputs[0]

                flattened = activations.reshape(inputs.shape[0], -1).detach().cpu()
                if output_tensor is None:
                    output_tensor = flattened
                else:
                    output_tensor = torch.cat((output_tensor, flattened), dim=0)

            if output_tensor is None:
                raise RuntimeError(f"No activations were collected for {arch} layer {layer_index}.")

            ambient_dimensions.append(output_tensor.shape[1])

            ids = []
            entropies = []
            l2_distances = []
            cosine_scores = []
            knn_distances = []

            for _ in range(resamples):
                permutation = np.random.permutation(output_tensor.shape[0])[:n_imgs]
                sampled = output_tensor[permutation, :]
                distance_matrix = squareform(pdist(sampled.numpy(), "euclidean"))

                try:
                    estimate_values = estimate(distance_matrix)
                    ids.append([estimate_values[2], estimate_values[3]])
                    entropies.append(float(est_entropy(sampled)))
                    l2_distances.append(float(np.linalg.norm(sampled.numpy(), axis=1, ord=2).mean()))
                    cosine_scores.append(list(cosine_sim_pairs(sampled)))
                    knn_distances.append(knn_avg_l2_dist(sampled, k=k_nn))
                except Exception:
                    continue

            ids = np.asarray(ids)
            if ids.shape[0] > 0:
                id_values[category_index, layer_index] = np.mean(ids[:, 0])
                id_errors[category_index, layer_index] = np.std(ids[:, 0])
            else:
                id_values[category_index, layer_index] = np.nan
                id_errors[category_index, layer_index] = np.nan

            entropies = np.asarray(entropies)
            if entropies.shape[0] > 0:
                entropy_values[category_index, layer_index] = np.mean(entropies)
                entropy_errors[category_index, layer_index] = np.std(entropies)
            else:
                entropy_values[category_index, layer_index] = np.nan
                entropy_errors[category_index, layer_index] = np.nan

            l2_distances = np.asarray(l2_distances)
            if l2_distances.shape[0] > 0:
                l2_values[category_index, layer_index] = np.mean(l2_distances)
                l2_errors[category_index, layer_index] = np.std(l2_distances)
            else:
                l2_values[category_index, layer_index] = np.nan
                l2_errors[category_index, layer_index] = np.nan

            cosine_scores = np.asarray(cosine_scores)
            if cosine_scores.shape[0] > 0:
                cosine_values[category_index, layer_index] = np.mean(cosine_scores[:, 0])
                cosine_errors[category_index, layer_index] = np.std(cosine_scores[:, 0])
            else:
                cosine_values[category_index, layer_index] = np.nan
                cosine_errors[category_index, layer_index] = np.nan

            if knn_distances:
                stacked_knn = torch.stack(knn_distances, dim=0).cpu().numpy()
                knn_values[category_index, layer_index, :] = np.nanmean(stacked_knn, axis=0)
                knn_errors[category_index, layer_index, :] = np.nanstd(stacked_knn, axis=0)
            else:
                knn_values[category_index, layer_index, :] = np.full((k_nn,), np.nan)
                knn_errors[category_index, layer_index, :] = np.full((k_nn,), np.nan)

    saved_paths = {
        "ID": save_numpy_array(cnn_metric_path(repo_paths, arch, "ID", trained=trained), id_values),
        "IDerr": save_numpy_array(cnn_metric_path(repo_paths, arch, "IDerr", trained=trained), id_errors),
        "ENTROPY": save_numpy_array(
            cnn_metric_path(repo_paths, arch, "ENTROPY", trained=trained),
            entropy_values,
        ),
        "ENTROPYerr": save_numpy_array(
            cnn_metric_path(repo_paths, arch, "ENTROPYerr", trained=trained),
            entropy_errors,
        ),
        "L2_DIST": save_numpy_array(cnn_metric_path(repo_paths, arch, "L2_DIST", trained=trained), l2_values),
        "L2_DISTerr": save_numpy_array(
            cnn_metric_path(repo_paths, arch, "L2_DISTerr", trained=trained),
            l2_errors,
        ),
        "COSSIM": save_numpy_array(
            cnn_metric_path(repo_paths, arch, "COSSIM", trained=trained),
            cosine_values,
        ),
        "COSSIMerr": save_numpy_array(
            cnn_metric_path(repo_paths, arch, "COSSIMerr", trained=trained),
            cosine_errors,
        ),
        "KNN_DIST": save_numpy_array(
            cnn_metric_path(repo_paths, arch, "KNN_DIST", trained=trained),
            knn_values,
        ),
        "KNN_DISTerr": save_numpy_array(
            cnn_metric_path(repo_paths, arch, "KNN_DISTerr", trained=trained),
            knn_errors,
        ),
        "ambdims": save_numpy_array(
            cnn_metric_path(repo_paths, arch, "ambdims", trained=trained),
            np.asarray(ambient_dimensions),
        ),
        "depths": save_numpy_array(cnn_metric_path(repo_paths, arch, "depths", trained=trained), depths),
        "names": save_numpy_array(
            cnn_metric_path(repo_paths, arch, "names", trained=trained),
            np.asarray(names, dtype=object),
        ),
    }
    return saved_paths
