"""Model initialization utilities for Vi-SmolLM2-135M."""

from transformers import AutoConfig, AutoModelForCausalLM


def create_model(
    base_model_name: str = "HuggingFaceTB/SmolLM2-135M",
    vocab_size: int = 32000,
    max_position_embeddings: int = 2048,
):
    """Instantiate SmolLM2ForCausalLM with Vietnamese-specific config.

    Args:
        base_model_name: HuggingFace model ID to base the architecture on.
        vocab_size: Tokenizer vocabulary size (must match tokenizer).
        max_position_embeddings: Maximum sequence length.

    Returns:
        AutoModelForCausalLM: Initialized model with random weights.
    """
    config = AutoConfig.from_pretrained(base_model_name)
    config.vocab_size = vocab_size
    config.max_position_embeddings = max_position_embeddings
    model = AutoModelForCausalLM.from_config(config)
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
