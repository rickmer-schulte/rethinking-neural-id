from __future__ import annotations

from pathlib import Path

from rethinking_neural_id.artifacts import save_representation_dict


def _vit_collate_fn(batch):
    import torch

    images, labels = zip(*batch)
    return list(images), torch.tensor(labels)


def extract_vit_representations(
    *,
    model_name: str,
    batch_size: int,
    data_root: str | Path,
    category_tag: str,
    nsamples: int,
    output_path: str | Path,
    num_workers: int = 4,
) -> Path:
    import sys

    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from transformers import AutoImageProcessor, AutoModel

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[ViT] device is {device}", file=sys.stderr)

    print(f"[ViT] Loading model: {model_name}", file=sys.stderr)
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True).to(device)
    model.eval()

    category_dir = Path(data_root) / category_tag
    print(f"[ViT] Using data folder: {category_dir}", file=sys.stderr)
    dataset = datasets.ImageFolder(root=str(category_dir), transform=None)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_vit_collate_fn,
    )

    def model_pass(images_batch):
        inputs = processor(images=images_batch, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)

        per_layer_activations = []
        for raw_activation in outputs.hidden_states:
            cls_acts = raw_activation[:, 0, :].detach().cpu().numpy()
            per_layer_activations.append([cls_acts[index] for index in range(cls_acts.shape[0])])
        return per_layer_activations

    states: dict[int, list] = {}
    n_seen = 0
    for images_batch, _labels in dataloader:
        if n_seen >= nsamples:
            break

        remaining = nsamples - n_seen
        if len(images_batch) > remaining:
            images_batch = images_batch[:remaining]

        current_output = model_pass(images_batch)
        for layer_idx, layer_values in enumerate(current_output):
            states.setdefault(layer_idx, []).extend(layer_values)

        n_seen += len(images_batch)
        print(f"[ViT] Processed {n_seen}/{nsamples} images for {category_tag}", file=sys.stderr)

    for layer_idx in states:
        states[layer_idx] = states[layer_idx][:nsamples]

    saved = save_representation_dict(output_path, states)
    print(f"[ViT] Saving to {saved}", file=sys.stderr)
    print("[ViT] Done.", file=sys.stderr)
    return saved
