import argparse

from rethinking_neural_id.pipelines.llm_metrics import compute_llm_metrics
from rethinking_neural_id.registry import LLM_DATASETS, LLM_MODELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute LLM layerwise metrics.")
    parser.add_argument("--model", type=str, required=True, choices=sorted(LLM_MODELS))
    parser.add_argument("--dataset", type=str, default="wikitext", choices=LLM_DATASETS)
    parser.add_argument("--shard", type=str, default="aa")
    parser.add_argument("--method", type=str, default="gride")
    parser.add_argument("--random_seed", type=int, default=32)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--k", type=int, default=5, help="k for kNN metrics.")
    parser.add_argument("--gride_k_max", type=int, default=64, help="k_max for GRIDE computation.")
    parser.add_argument("--block_size", type=int, default=1000, help="Block size for pairwise metrics.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    save_path, _results = compute_llm_metrics(
        model=args.model,
        dataset=args.dataset,
        shard=args.shard,
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
