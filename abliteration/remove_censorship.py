"""Abliteration Module: Remove Censorship from Vi-SmolLM2-135M (Stage 4).

Uses weight orthogonalization to strip the refusal vector from model weights.
No additional training is needed — uses linear algebra on the activation space.

Reference: https://arxiv.org/abs/2310.11453 (Abliteration)
"""

import json
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_censored_model(model_path: str = "./vi-smollm-135m-censored"):
    """Load the censored (DPO-aligned) model and tokenizer.

    Args:
        model_path: Path to the censored model directory.

    Returns:
        tuple: (model, tokenizer)
    """
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
    return model, tokenizer


def extract_activations(
    model,
    tokenizer,
    prompts: list[str],
    layer_idx: int = -1,
    max_length: int = 2048,
):
    """Extract hidden-state activations from the model for a set of prompts.

    Args:
        model: The (censored) model.
        tokenizer: The tokenizer.
        prompts: List of text prompts.
        layer_idx: Which transformer layer to extract from (-1 = last layer).
        max_length: Maximum token length.

    Returns:
        torch.Tensor: Activations of shape [num_samples, hidden_dim].
    """
    model.eval()
    all_activations = []

    # Register hook on target layer
    target_layer = model.model.layers[layer_idx]
    hook_handle = target_layer.register_forward_hook(
        lambda module, input, output: all_activations.append(output[0].detach().clone())
    )

    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            _ = model(**inputs)

    hook_handle.remove()

    # Concatenate all activations and take the last token's hidden state
    # Shape: [num_samples, 1, seq_len, hidden_dim] -> [num_samples, hidden_dim]
    stacked = torch.cat(all_activations, dim=0)  # [num_samples, seq_len, hidden_dim]
    last_token_acts = stacked[:, -1, :]  # [num_samples, hidden_dim]
    return last_token_acts


def orthogonalize_weight(matrix: torch.Tensor, direction: torch.Tensor):
    """Remove the component of `direction` from `matrix` via projection subtraction.

    W_orthogonal = W - W @ v @ v^T
    where v is the normalized refusal direction.

    Args:
        matrix: Weight matrix to orthogonalize (shape [out_features, in_features]).
        direction: The refusal direction vector (shape [in_features,]).

    Returns:
        torch.Tensor: Orthogonalized weight matrix.
    """
    direction = direction / direction.norm()  # Normalize
    # Projection = W . v . v^T  (v is [in_features,], so v.unsqueeze(1) is [in_features, 1])
    projection = torch.matmul(matrix, direction.unsqueeze(1)) @ direction.unsqueeze(0)
    return matrix - projection


def run_abliteration(
    censored_model_path: str = "./vi-smollm-135m-censored",
    output_path: str = "./vi-smollm-135m-uncensored",
    harmful_prompts_path: str = "./data/harmful_prompts_vi.json",
    harmless_prompts_path: str = "./data/harmless_prompts_vi.json",
    target_layer: int = -1,
):
    """Execute the full abliteration pipeline (Stage 4).

    Steps:
        1. Load the censored model
        2. Extract activations for harmful and harmless prompts
        3. Compute refusal direction: harmful_mean - harmless_mean
        4. Orthogonalize refusal direction from all `down_proj.weight` matrices
        5. Save the uncensored model

    Args:
        censored_model_path: Path to the DPO/censored model.
        output_path: Where to save the uncensored model.
        harmful_prompts_path: JSON file with ~200 harmful VI prompts.
        harmless_prompts_path: JSON file with ~200 harmless VI prompts.
        target_layer: Transformer layer index for activation extraction.
    """
    print("[Stage 4] Loading censored model...")
    model, tokenizer = load_censored_model(censored_model_path)

    # Load prompt data
    with open(harmful_prompts_path, "r") as f:
        harmful_prompts = json.load(f)
    with open(harmless_prompts_path, "r") as f:
        harmless_prompts = json.load(f)

    print(f"[Stage 4] Loaded {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts")

    # Extract activations
    print("[Stage 4] Extracting harmful activations...")
    harmful_acts = extract_activations(model, tokenizer, harmful_prompts, layer_idx=target_layer)

    print("[Stage 4] Extracting harmless activations...")
    harmless_acts = extract_activations(model, tokenizer, harmless_prompts, layer_idx=target_layer)

    # Compute refusal direction
    harmful_mean = harmful_acts.mean(dim=0)
    harmless_mean = harmless_acts.mean(dim=0)
    refusal_dir = harmful_mean - harmless_mean
    refusal_dir = refusal_dir / refusal_dir.norm()

    print(f"[Stage 4] Refusal direction computed: norm={refusal_dir.norm().item():.6f}")

    # Apply orthogonalization to all transformer layers' down_proj weights
    print("[Stage 4] Orthogonalizing refusal vector from model weights...")
    for layer_idx, layer in enumerate(model.model.layers):
        # Apply to MLP down_proj weight
        original_weight = layer.mlp.down_proj.weight.data.clone()
        layer.mlp.down_proj.weight.data = orthogonalize_weight(
            original_weight,
            refusal_dir.to(original_weight.device),
        )
        print(f"  Layer {layer_idx}: down_proj.weight orthogonalized")

    # Also apply to the final language model head if present
    if hasattr(model, "lm_head") and model.lm_head is not None:
        original_lm = model.lm_head.weight.data.clone()
        model.lm_head.weight.data = orthogonalize_weight(
            original_lm,
            refusal_dir.to(original_lm.device),
        )
        print("  lm_head.weight orthogonalized")

    # Save uncensored model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"[Stage 4] Uncensored model saved to {output_path}")

    # Evaluate: measure refusal rate on harmful/harmless prompts
    from utils.eval_utils import evaluate_abliteration, load_abliteration_prompts
    harmful_prompts, harmless_prompts = load_abliteration_prompts()
    evaluate_abliteration(model, tokenizer, harmful_prompts, harmless_prompts)


if __name__ == "__main__":
    run_abliteration()
