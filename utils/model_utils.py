"""Model initialization utilities for Vi-SmolLM2-135M."""

from transformers import AutoConfig, AutoModelForCausalLM


def create_model(
    base_model_name: str = "HuggingFaceTB/SmolLM2-135M",
    vocab_size: int = 32000,
    max_position_embeddings: int = 2048,
    init_from_base: bool = False,
):
    """Instantiate SmolLM2ForCausalLM with Vietnamese-specific config.

    Args:
        base_model_name: HuggingFace model ID to base the architecture on.
        vocab_size: Tokenizer vocabulary size (must match tokenizer).
        max_position_embeddings: Maximum sequence length.
        init_from_base: If True, keep the base model's pretrained transformer
            layers and only re-initialize the (tied) token embeddings, since the
            Vietnamese tokenizer's ids have nothing in common with the base
            vocabulary. If False, all weights are random.

    Returns:
        AutoModelForCausalLM: Initialized model.
    """
    if not init_from_base:
        config = AutoConfig.from_pretrained(base_model_name)
        config.vocab_size = vocab_size
        config.max_position_embeddings = max_position_embeddings
        return AutoModelForCausalLM.from_config(config)

    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    old_std = model.get_input_embeddings().weight.std().item()
    model.resize_token_embeddings(vocab_size, mean_resizing=False)
    # Fresh embeddings at the base embedding scale; lm_head is tied to them.
    model.get_input_embeddings().weight.data.normal_(mean=0.0, std=old_std)
    model.config.max_position_embeddings = max_position_embeddings
    return model


def load_model(model_path: str):
    """Load a saved model from disk.

    Args:
        model_path: Path to the saved model directory.

    Returns:
        AutoModelForCausalLM: Loaded model.
    """
    return AutoModelForCausalLM.from_pretrained(model_path)


def save_model(model, model_path: str):
    """Save model to disk.

    Args:
        model: The model to save.
        model_path: Destination directory.
    """
    model.save_pretrained(model_path)
