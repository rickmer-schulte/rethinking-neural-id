import argparse

from rethinking_neural_id.pipelines.vit_metrics import compute_vit_metrics
from rethinking_neural_id.registry import IMAGE_DATASET_DIRS, VIT_MODELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute ViT layerwise metrics from saved representations.")
    parser.add_argument("--pickle-path", type=str, default=None)
    parser.add_argument("--model", type=str, choices=sorted(VIT_MODELS), default=None)
    parser.add_argument("--dataset", type=str, choices=sorted(IMAGE_DATASET_DIRS), default=None)
    parser.add_argument("--category", type=str, default=None)
    parser.add_argument(
        "--method",
        type=str,
        default="gride",
        choices=["twonn", "entropy", "gride", "avg_l2", "avg_cosine", "knn"],
    )
    parser.add_argument("--random_seed", type=int, default=32)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--gride_k_max", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=1000)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    save_path, _results = compute_vit_metrics(
        pickle_path=args.pickle_path,
        model=args.model,
        dataset=args.dataset,
        category=args.category,
        method=args.method,
        random_seed=args.random_seed,
        step=args.step,
        k=args.k,
        gride_k_max=args.gride_k_max,
        block_size=args.block_size,
    )
    print(f"Saved results to: {save_path}")


if __name__ == "__main__":
    main()
