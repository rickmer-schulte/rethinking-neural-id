import argparse

from rethinking_neural_id.pipelines.cnn_metrics import compute_cnn_metrics
from rethinking_neural_id.registry import CNN_ARCHS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute CNN layerwise metrics.")
    parser.add_argument("--arch", required=True, choices=CNN_ARCHS)
    parser.add_argument("--nsamples", default=500, type=int, help="Number of samples.")
    parser.add_argument("--bs", default=16, type=int, help="Mini-batch size.")
    parser.add_argument("--k_nn", default=5, type=int, help="Number of nearest neighbors.")
    parser.add_argument("--res", default=5, type=int, help="Number of resamplings.")
    parser.add_argument("--trained", default=1, type=int, choices=[0, 1], help="Use trained weights.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    saved_paths = compute_cnn_metrics(
        arch=args.arch,
        nsamples=args.nsamples,
        batch_size=args.bs,
        k_nn=args.k_nn,
        resamples=args.res,
        trained=bool(args.trained),
    )
    print(f"Saved CNN metrics for {args.arch}:")
    for name, path in saved_paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
