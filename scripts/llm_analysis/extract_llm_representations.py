import argparse

from rethinking_neural_id.artifacts import llm_representation_path
from rethinking_neural_id.paths import RepoPaths
from rethinking_neural_id.pipelines.llm_extract import extract_llm_representations
from rethinking_neural_id.registry import LLM_DATASETS, LLM_MODELS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract final-token LLM representations.")
    parser.add_argument("--model-name", required=True, help="Hugging Face model name.")
    parser.add_argument("--model-key", required=True, choices=sorted(LLM_MODELS))
    parser.add_argument("--dataset", required=True, choices=LLM_DATASETS)
    parser.add_argument("--shard", required=True, help="Shard identifier used in output naming.")
    parser.add_argument("--data-file", required=True, help="Input text file.")
    parser.add_argument("--batch-size", default=6, type=int)
    parser.add_argument("--quantization", default="none", choices=["none", "8bit"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_paths = RepoPaths.default()
    output_path = llm_representation_path(repo_paths, args.model_key, args.dataset, args.shard)
    saved = extract_llm_representations(
        model_name=args.model_name,
        batch_size=args.batch_size,
        data_file=args.data_file,
        output_path=output_path,
        quantization=args.quantization,
    )
    print(f"Saved representations to: {saved}")


if __name__ == "__main__":
    main()
