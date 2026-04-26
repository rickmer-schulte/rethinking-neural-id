# Experiments

## Reproduction Workflows

Prepare raw inputs before running extraction or metric scripts:
- `uv run python scripts/prepare_raw_data.py`

Raw-data origin: image data follows [Ansuini et al. (2019)](https://github.com/ansuini/IntrinsicDimDeep), and text data follows [Cheng et al. (2025)](https://github.com/chengemily1/id-llm-abstraction).

### CNNs
- Inputs:
  - `data/raw/images/imagenet_training_single_objs/<category>/...`
- Metric command:
  - `uv run python scripts/cnn_analysis/cnn_reps_computation.py --arch resnet18 --trained 1`
- Plotting notebook:
  - `scripts/cnn_analysis/cnn_reps_analysis.ipynb`
- Output artifacts:
  - `results/cnns/metrics/<arch>_ID.npy`
  - `results/cnns/metrics/<arch>_ENTROPY.npy`
  - `results/cnns/metrics/<arch>_L2_DIST.npy`
  - `results/cnns/metrics/<arch>_COSSIM.npy`
  - `results/cnns/metrics/<arch>_KNN_DIST.npy`

### LLMs
- Inputs:
  - raw text shard file such as `data/raw/texts/wikitext/shard_aa`
- Extraction command:
  - `uv run python scripts/llm_analysis/extract_llm_representations.py --model-name meta-llama/Llama-3.1-8B --model-key llama --dataset wikitext --shard aa --data-file data/raw/texts/wikitext/shard_aa --quantization 8bit`
- Metric command:
  - `uv run python scripts/llm_analysis/compute_layerwise_metrics.py --model llama --dataset wikitext --shard aa --method gride`
- Plotting notebook:
  - `scripts/llm_analysis/llm_analysis.ipynb`
- Output artifacts:
  - `data/reps/llama-3.1-8B/wikitext/hidden_llama_wikitext_shard_aa.pickle`
  - `results/llms/metrics/gride/hidden_wikitext_aa_llama.json`

### ViTs
- Inputs:
  - `data/raw/images/imagenet_training_single_objs/<category>/...`
- Extraction command:
  - `uv run python scripts/vit_analysis/extract_vit_representations.py --model-name google/vit-base-patch16-224 --model-key vit-base --dataset imagenet7 --category-tag mix --nsamples 5000`
- Metric command:
  - `uv run python scripts/vit_analysis/compute_layerwise_metrics_vits.py --model vit-base --dataset imagenet7 --category mix --method entropy`
- Plotting notebook:
  - `scripts/vit_analysis/vit_analysis.ipynb`
- Output artifacts:
  - `data/reps/vit-base/imagenet7/hidden_vit-base_imagenet7-mix.pickle`
  - `results/vits/metrics/entropy/hidden_imagenet7-mix_vit-base.json`

### Bias Analysis
- Notebook:
  - `scripts/bias_analysis/id_est_bias.ipynb`
