"""Dataset loading utilities for all training stages of Vi-SmolLM2-135M."""

from itertools import chain
from datasets import Features, List, Value, load_dataset
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
    if streaming or max_samples is None:
        return ds
    return ds.select(range(min(max_samples, len(ds))))


def load_pretrain_eval_texts(
    vtsnlp_docs: int = 1000,
    culturax_heldout_file: str | None = "vi/vi_part_00010.parquet",
    culturax_docs: int = 1000,
) -> dict[str, list[str]]:
    """Load the Stage 1 eval texts: VTSNLP (the plan's eval set) and a held-out CulturaX shard.

    Both are streamed so only the first ``n`` documents are read. The CulturaX
    shard must not be one of the training shards.
    """
    texts = {}
    if vtsnlp_docs:
        ds = load_dataset("VTSNLP/vietnamese_curated_dataset", split="train", streaming=True)
        texts["vtsnlp"] = [x["text"] for _, x in zip(range(vtsnlp_docs), ds)]
    if culturax_heldout_file and culturax_docs:
        ds = load_dataset("uonlp/CulturaX", "vi", data_files=culturax_heldout_file, split="train", streaming=True)
        texts["culturax_heldout"] = [x["text"] for _, x in zip(range(culturax_docs), ds)]
    return texts


def pack_dataset(dataset, tokenizer: PreTrainedTokenizerFast, block_size: int = 2048,
                 num_proc: int | None = None, text_field: str = "text"):
    """Tokenize documents and pack them into fixed ``block_size`` blocks with no padding.

    Each document gets an EOS appended, documents are concatenated, and the
    stream is cut into blocks. The remainder of each map batch (< block_size
    tokens per 1000 docs) is dropped. Token ids are stored as int32.
    """
    eos = tokenizer.eos_token_id
    features = Features({"input_ids": List(Value("int32"))})

    def _tokenize(batch):
        ids = tokenizer(batch[text_field], add_special_tokens=False)["input_ids"]
        return {"input_ids": [x + [eos] for x in ids]}

    def _group(batch):
        concat = list(chain.from_iterable(batch["input_ids"]))
        n = len(concat) // block_size * block_size
        return {"input_ids": [concat[i:i + block_size] for i in range(0, n, block_size)]}

    tokenized = dataset.map(_tokenize, batched=True, num_proc=num_proc, features=features,
                            remove_columns=dataset.column_names, desc="Tokenizing")
    return tokenized.map(_group, batched=True, batch_size=1000, num_proc=num_proc,
                         features=features, desc=f"Packing into {block_size}-token blocks")


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


def format_for_chatml(dataset, tokenizer: PreTrainedTokenizerFast,
                      system_prompt: str = "You are a helpful Vietnamese assistant."):
    """Format vi-alpaca samples as conversational prompt/completion pairs (ChatML).

    ``instruction`` and the optional ``input`` are merged into one user turn.
    The output is ``{"prompt": [system, user], "completion": [assistant]}``, so
    TRL's SFTTrainer computes the loss only on the assistant reply (including
    its ``<|im_end|>``), not on the system/user text.

    Args:
        dataset: The raw Alpaca-style dataset.
        tokenizer: The tokenizer to attach the chat template.
        system_prompt: System message; eval prompts must use the same one.

    Returns:
        Dataset: Columns ``prompt`` and ``completion`` (lists of messages).
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
        user = (sample.get("instruction") or "").strip()
        extra = (sample.get("input") or "").strip()
        if extra:
            user = f"{user}\n\n{extra}"
        return {
            "prompt": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            "completion": [{"role": "assistant", "content": (sample.get("output") or "").strip()}],
        }

    return dataset.map(_format, remove_columns=dataset.column_names)


def filter_sft_samples(dataset, tokenizer: PreTrainedTokenizerFast,
                       min_output_tokens: int = 3, max_total_tokens: int = 1024):
    """Drop prompt/completion samples with an empty-ish reply or that would be truncated."""
    def _keep(sample):
        out = len(tokenizer(sample["completion"][0]["content"], add_special_tokens=False)["input_ids"])
        prompt = sum(len(tokenizer(m["content"], add_special_tokens=False)["input_ids"]) for m in sample["prompt"])
        return out >= min_output_tokens and prompt + out + 16 <= max_total_tokens  # +16: ChatML markup

    return dataset.filter(_keep)
