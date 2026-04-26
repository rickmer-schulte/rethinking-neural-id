import argparse

from rethinking_neural_id.artifacts import vit_representation_path
from rethinking_neural_id.paths import RepoPaths
from rethinking_neural_id.pipelines.vit_extract import extract_vit_representations
from rethinking_neural_id.registry import IMAGE_DATASET_DIRS, VIT_MODELS, get_image_dataset_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract ViT CLS-token representations.")
    parser.add_argument("--model-name", required=True, help="Hugging Face model name.")
    parser.add_argument("--model-key", required=True, choices=sorted(VIT_MODELS))
    parser.add_argument("--dataset", default="imagenet7", choices=sorted(IMAGE_DATASET_DIRS))
    parser.add_argument("--category-tag", required=True)
    parser.add_argument("--nsamples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_paths = RepoPaths.default()
    data_root = repo_paths.raw_images_root / get_image_dataset_dir(args.dataset)
    output_path = vit_representation_path(
        repo_paths,
        args.model_key,
        args.dataset,
        args.category_tag,
    )
    saved = extract_vit_representations(
        model_name=args.model_name,
        batch_size=args.batch_size,
        data_root=data_root,
        category_tag=args.category_tag,
        nsamples=args.nsamples,
        output_path=output_path,
        num_workers=args.num_workers,
    )
    print(f"Saved representations to: {saved}")


if __name__ == "__main__":
    main()
