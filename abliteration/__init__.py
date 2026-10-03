"""Abliteration module for Vi-SmolLM2-135M (Stage 4)."""

from abliteration.remove_censorship import (
    load_censored_model,
    extract_activations,
    orthogonalize_weight,
    run_abliteration,
)

__all__ = [
    "load_censored_model",
    "extract_activations",
    "orthogonalize_weight",
    "run_abliteration",
]
