# Rethinking Intrinsic Dimension Estimation in Neural Representations
This repository contains the code for the experiments and figures of the AISTATS 2026 paper: "Rethinking Intrinsic Dimension Estimation in Neural Representations".

## Project Structure
```text
.
├── data/
│   ├── raw/
│   └── reps/
├── docs/
│   ├── setup.md
│   └── experiments.md
├── results/
│   ├── cnns/
│   ├── llms/
│   └── vits/
├── scripts/
│   ├── bias_analysis/
│   ├── cnn_analysis/
│   ├── llm_analysis/
│   └── vit_analysis/
└── src/
    └── rethinking_neural_id/
```

`scripts/` contains the user-facing reproduction entrypoints. `src/rethinking_neural_id/` is a light internal namespace that keeps shared paths, registries, metrics, and pipelines import-stable across shells and notebooks.

## Python Setup
```bash
# Clone the repository
git clone https://github.com/rickmer-schulte/rethinking-neural-id.git
cd rethinking-neural-id

# Install the local reproduction environment
uv sync --extra metrics --extra torch --extra bias --extra notebooks --extra dev
```

For LLM extraction:
```bash
uv sync --extra metrics --extra torch --extra llm --extra bias --extra notebooks --extra dev
```

Download and prepare raw text and image data:
```bash
uv run python scripts/prepare_raw_data.py
```

## Quick Start

Extract CNN representations and compute metrics:
```bash
uv run python scripts/cnn_analysis/cnn_reps_computation.py --arch resnet18 --trained 1
```

Extract LLM representations (e.g. from [Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) model):
```bash
uv run python scripts/llm_analysis/extract_llm_representations.py \
  --model-name meta-llama/Llama-3.1-8B \
  --model-key llama \
  --dataset wikitext \
  --shard aa \
  --data-file data/raw/texts/wikitext/shard_aa \
  --quantization 8bit
```

Compute LLM metrics (e.g. ID estimation on llama reps via gride):
```bash
uv run python scripts/llm_analysis/compute_layerwise_metrics.py \
  --model llama \
  --dataset wikitext \
  --shard aa \
  --method gride
```

Extract ViT representations (e.g. from 
[vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224) model):
```bash
uv run python scripts/vit_analysis/extract_vit_representations.py \
  --model-name google/vit-base-patch16-224 \
  --model-key vit-base \
  --dataset imagenet7 \
  --category-tag mix \
  --nsamples 5000
```

Compute ViT metrics (e.g. entropy estimation for vit-base reps):
```bash
uv run python scripts/vit_analysis/compute_layerwise_metrics_vits.py \
  --model vit-base \
  --dataset imagenet7 \
  --category mix \
  --method entropy
```

See [docs/setup.md](docs/setup.md) for environment details and [docs/experiments.md](docs/experiments.md) for experimental details.

## Citation
```bibtex
@inproceedings{
schulte2026rethinking,
title={Rethinking Intrinsic Dimension Estimation in Neural Representations},
author={Rickmer Schulte and David R{\"u}gamer},
booktitle={The 29th International Conference on Artificial Intelligence and Statistics},
year={2026},
url={https://openreview.net/forum?id=kH1gPRbYqh}
}
```
