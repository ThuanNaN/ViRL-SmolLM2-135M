"""Dataset loading utilities for all training stages of Vi-SmolLM2-135M."""

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast


def load_culturX_vi(
    dataset_name: str = "uonlp/CulturaX",
    dataset_subset: str = "vi",
    max_samples: int = 1000000,
    streaming: bool = True,
    data_files: str | None = None,
):
    """Load the CulturX Vietnamese subset for pre-training.

    Args:
        dataset_name: HuggingFace dataset ID.
        dataset_subset: The language subset (e.g., 'vi').
        max_samples: Maximum number of samples (non-streaming only). A streamed
            dataset is bounded by the trainer's ``max_steps`` instead, because
            ``IterableDataset.take`` blocks source shuffling and raises
            ``DataSourcesShufflingDisallowed`` when the epoch is non-zero (resume).
        streaming: Whether to stream the dataset.
        data_files: Optional glob of shards inside the repo (e.g.
            ``"vi/vi_part_0000[0-3].parquet"``) to load only a few shards. Shards
            already in the HF cache are reused, not downloaded again.

    Returns:
        Dataset: Dataset of Vietnamese text.
    """
    ds = load_dataset(
        dataset_name,
        dataset_subset,
        data_files=data_files,
        split="train",
        streaming=streaming,
    )
    if streaming:
        return ds
    return ds.select(range(min(max_samples, len(ds))))


def load_vi_alpaca(
    dataset_name: str = "bkai-foundation-models/vi-alpaca",
):
    """Load the Vietnamese Alpaca dataset for SFT.

    Each sample is formatted into a conversation dictionary compatible with
    the ChatML template.

    Returns:
        Dataset: Vietnamese Alpaca conversation data.
    """
    ds = load_dataset(dataset_name, split="train")
    return ds


def load_pkusaferlhf_vi(
    dataset_name: str = "1997AOF/PKU-SafeRLHF-VI",
):
    """Load the PKU-SafeRLHF-VN dataset for DPO safety alignment.

    Each sample contains:
        - prompt: The dangerous question
        - chosen: The safe refusal response
        - rejected: The harmful response

    Returns:
        Dataset: DPO-formatted safety data.
    """
    ds = load_dataset(dataset_name, split="train")
    return ds


def load_abliteration_data(
    harmful_path: str = "./data/harmful_prompts_vi.json",
    harmless_path: str = "./data/harmless_prompts_vi.json",
):
    """Load the harmful/harmless prompt pairs for abliteration.

    Args:
        harmful_path: Path to file with ~200 harmful Vietnamese prompts.
        harmless_path: Path to file with ~200 harmless Vietnamese prompts.

    Returns:
        tuple: (harmful_prompts, harmless_prompts) as lists of strings.
    """
    import json

    with open(harmful_path, "r") as f:
        harmful_prompts = json.load(f)
    with open(harmless_path, "r") as f:
        harmless_prompts = json.load(f)
    return harmful_prompts, harmless_prompts


def format_for_chatml(dataset, tokenizer: PreTrainedTokenizerFast):
    """Format dataset samples into ChatML conversation format.

    Each sample should have an 'instruction' and 'input' field (vi-alpaca format)
    and is converted to a list of message dictionaries.

    Args:
        dataset: The raw Alpaca-style dataset.
        tokenizer: The tokenizer to attach the chat template.

    Returns:
        Dataset: Formatted with 'text' field containing the full conversation.
    """
    chat_template = tokenizer.chat_template or (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{'<|im_start|>assistant\\n'}}"
        "{% endif %}"
    )
    tokenizer.chat_template = chat_template

    def _format(sample):
        messages = [
            {"role": "system", "content": "You are a helpful Vietnamese assistant."},
            {"role": "user", "content": sample.get("instruction", "")},
        ]
        if sample.get("input"):
            messages.append({"role": "user", "content": sample["input"]})
        messages.append({"role": "assistant", "content": sample.get("output", "")})
        return {"messages": messages}

    return dataset.map(_format, remove_columns=dataset.column_names)
