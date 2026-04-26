# Setup

## Python Version
- Python `3.11` is required (`>=3.11,<3.13`).

## Local Setup with `uv`
```bash
# Clone the repository
git clone https://github.com/rickmer-schulte/rethinking-neural-id.git
cd rethinking-neural-id

# Core metrics + torch + bias notebook + notebooks + dev tools
uv sync --extra metrics --extra torch --extra bias --extra notebooks --extra dev
```

For LLM extraction, include `llm`:
```bash
uv sync --extra metrics --extra torch --extra llm --extra bias --extra notebooks --extra dev
```

## Raw Data
Download and prepare the raw wikitext shards and ImageNet object images:
```bash
uv run python scripts/prepare_raw_data.py
```

This writes extensionless text shards to `data/raw/texts/wikitext/shard_aa` through `shard_ae` and copies the required image folder to `data/raw/images/imagenet_training_single_objs/`.

The raw data sources follow prior studies: image data as used in [Ansuini et al. (2019)](https://github.com/ansuini/IntrinsicDimDeep), and text data as used in [Cheng et al. (2025)](https://github.com/chengemily1/id-llm-abstraction). The image archive is about 2.7 GB, so the download can take a few minutes.

## Running Scripts
Use `uv run` so commands always execute in the project environment:
```bash
uv run python scripts/cnn_analysis/cnn_reps_computation.py --arch resnet18 --trained 1
uv run python scripts/llm_analysis/compute_layerwise_metrics.py --model llama --dataset wikitext --shard aa --method gride
uv run python scripts/vit_analysis/compute_layerwise_metrics_vits.py --model vit-base --dataset imagenet7 --category mix --method entropy
```

## Path Configuration
Reproduction code resolves paths via `RepoPaths` with these optional environment overrides:
- `RNI_DATA_ROOT` (default: `<repo>/data`)
- `RNI_RESULTS_ROOT` (default: `<repo>/results`)

Example:
```bash
export RNI_DATA_ROOT=/absolute/path/to/data
export RNI_RESULTS_ROOT=/absolute/path/to/results
```

## Colab Fallback
When running in Colab, install from the same `pyproject.toml`:
```bash
pip install -e ".[metrics,torch,llm,bias,notebooks]"
```

Then run reproduction scripts through notebook shell cells:
```bash
!python scripts/llm_analysis/extract_llm_representations.py --model-name meta-llama/Llama-3.1-8B --model-key llama --dataset wikitext --shard aa --data-file data/raw/texts/wikitext/shard_aa --quantization 8bit
```
