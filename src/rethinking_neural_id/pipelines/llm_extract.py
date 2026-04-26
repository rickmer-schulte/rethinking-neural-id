# Code adapted from:
# https://github.com/chengemily1/id-llm-abstraction/blob/main/scripts/extract_final_representations.py

from __future__ import annotations

from pathlib import Path

from rethinking_neural_id.artifacts import save_representation_dict


def extract_llm_representations(
    *,
    model_name: str,
    batch_size: int,
    data_file: str | Path,
    output_path: str | Path,
    quantization: str = "none",
) -> Path:
    import sys
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if quantization not in {"none", "8bit"}:
        raise ValueError("quantization must be one of: 'none', '8bit'")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device is {device}", file=sys.stderr)

    model_kwargs = {}
    if quantization == "8bit":
        model_kwargs["load_in_8bit"] = True
        model_kwargs["device_map"] = "auto"
    elif torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def model_pass(raw_inputs: list[str]) -> list[list]:
        inputs = tokenizer(raw_inputs, padding=True, return_tensors="pt").to(device)

        last_true_token_indices = []
        for attention_mask in inputs.attention_mask:
            if 0 not in attention_mask:
                last_true_token_indices.append(len(attention_mask) - 1)
            else:
                last_true_token_indices.append(attention_mask.tolist().index(0) - 1)

        with torch.no_grad():
            if "OLMo" in model_name:
                hidden_states = model(
                    inputs.input_ids.to(device),
                    output_hidden_states=True,
                ).hidden_states
            else:
                hidden_states = model(**inputs, output_hidden_states=True).hidden_states

        per_layer_activations = []
        for raw_activation in hidden_states:
            last_token_activations = []
            for index, token_index in enumerate(last_true_token_indices):
                last_token_activation = raw_activation[index][token_index].cpu().numpy()
                last_token_activations.append(last_token_activation)
            per_layer_activations.append(last_token_activations)
        return per_layer_activations

    inputs: list[str] = []
    with open(data_file, "r", encoding="utf-8") as handle:
        for line in handle:
            inputs.append(line.rstrip("\n").split("\t")[0])

    cases_count = len(inputs)
    first_index = 0
    current_batch_size = min(batch_size, cases_count)
    states: dict[int, list] = {}

    while (first_index + current_batch_size) < cases_count:
        current_output = model_pass(inputs[first_index : first_index + current_batch_size])
        for layer_idx, layer_values in enumerate(current_output):
            states.setdefault(layer_idx, []).extend(layer_values)
        first_index += current_batch_size

    if first_index < cases_count:
        current_output = model_pass(inputs[first_index:cases_count])
        for layer_idx, layer_values in enumerate(current_output):
            states.setdefault(layer_idx, []).extend(layer_values)

    return save_representation_dict(output_path, states)
