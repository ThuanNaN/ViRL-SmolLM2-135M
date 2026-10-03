"""Train a Byte-Level BPE Tokenizer for Vietnamese (Stage 1.1).

Creates a 32,000-token vocabulary tokenizer trained on uonlp/CulturX (vi subset).
"""

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset


def train_byte_level_bpe(
    vocab_size: int = 32000,
    dataset_name: str = "uonlp/CulturX",
    dataset_subset: str = "vi",
    max_samples: int = 1000000,
    output_dir: str = "./vi_smollm_tokenizer",
):
    """Train and save a Byte-Level BPE Tokenizer for Vietnamese.

    Args:
        vocab_size: Target vocabulary size.
        dataset_name: HuggingFace dataset ID.
        dataset_subset: Language subset ('vi').
        max_samples: Number of samples to train on.
        output_dir: Directory to save the tokenizer.

    Returns:
        PreTrainedTokenizerFast: The trained tokenizer.
    """
    # Load Vietnamese dataset
    ds = load_dataset(dataset_name, dataset_subset, split="train", streaming=True)

    # Initialize Byte-Level BPE Tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # Configure trainer with Vietnamese-specific special tokens
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<unk>",
            "<s>",
            "</s>",
            "<pad>",
            "<mask_token>",
            "<|im_start|>",
            "<|im_end|>",
        ],
    )

    # Train from Vietnamese text corpus
    def batch_iterator(batch_size=1000):
        count = 0
        for item in ds:
            if count >= max_samples:
                break
            yield item["text"]
            count += 1

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Wrap as HuggingFace tokenizer
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    hf_tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\\n' + message['content'] + '<|im_end|>' + '\\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{'<|im_start|>assistant\\n'}}"
        "{% endif %}"
    )

    # Save tokenizer
    hf_tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir} with vocab_size={vocab_size}")
    return hf_tokenizer


if __name__ == "__main__":
    train_byte_level_bpe()
