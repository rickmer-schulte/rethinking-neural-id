from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str
    rep_dir: str
    result_key: str
    default_hf_name: str | None = None


LLM_MODELS = {
    "llama": ModelSpec(
        key="llama",
        rep_dir="llama-3.1-8B",
        result_key="llama",
        default_hf_name="meta-llama/Llama-3.1-8B",
    ),
    "mistral": ModelSpec(
        key="mistral",
        rep_dir="mistral-7B-v0.3",
        result_key="mistral",
        default_hf_name="mistralai/Mistral-7B-v0.3",
    ),
    "pythia": ModelSpec(
        key="pythia",
        rep_dir="pythia-6.9b",
        result_key="pythia",
        default_hf_name="EleutherAI/pythia-6.9b",
    ),
}

VIT_MODELS = {
    "vit-base": ModelSpec(
        key="vit-base",
        rep_dir="vit-base",
        result_key="vit-base",
        default_hf_name="google/vit-base-patch16-224",
    ),
    "dinov3-vitb16": ModelSpec(
        key="dinov3-vitb16",
        rep_dir="dinov3-vitb16",
        result_key="dinov3-vitb16",
    ),
    "dinov3-vitl16": ModelSpec(
        key="dinov3-vitl16",
        rep_dir="dinov3-vitl16",
        result_key="dinov3-vitl16",
    ),
}

LLM_DATASETS = ("wikitext",)
IMAGE_DATASET_DIRS = {
    "imagenet7": "imagenet_training_single_objs",
}
IMAGENET7_CATEGORIES = (
    "n01882714",
    "n02086240",
    "n02087394",
    "n02094433",
    "n02100583",
    "n02100735",
    "n02279972",
    "mix",
)

CNN_ARCHS = (
    "alexnet",
    "vgg11",
    "vgg13",
    "vgg16",
    "vgg19",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
)


def get_llm_model_spec(key: str) -> ModelSpec:
    try:
        return LLM_MODELS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown LLM model key: {key}") from exc


def get_vit_model_spec(key: str) -> ModelSpec:
    try:
        return VIT_MODELS[key]
    except KeyError as exc:
        raise KeyError(f"Unknown ViT model key: {key}") from exc


def get_image_dataset_dir(dataset: str) -> str:
    try:
        return IMAGE_DATASET_DIRS[dataset]
    except KeyError as exc:
        raise KeyError(f"Unknown image dataset key: {dataset}") from exc


def get_image_categories(dataset: str) -> tuple[str, ...]:
    if dataset == "imagenet7":
        return IMAGENET7_CATEGORIES
    raise KeyError(f"Unknown image dataset key: {dataset}")
