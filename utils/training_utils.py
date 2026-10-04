"""Shared training utilities: optimizer, scheduler, and trainer helpers."""

from transformers import TrainingArguments
from transformers import DataCollatorForLanguageModeling
from trl.trainer.sft_config import SFTConfig
from trl.trainer.dpo_config import DPOConfig


def get_pretrain_training_args(
    output_dir: str = "./vi-smollm-135m-pretrain",
    learning_rate: float = 5e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 32,
    per_device_eval_batch_size: int = 32,
    gradient_accumulation_steps: int = 1,
    logging_steps: int = 10,
    save_strategy: str = "epoch",
    max_steps: int = -1,
    warmup_steps: int = 2000,
    bf16: bool = True,
    **kwargs,
):
    """Create TrainingArguments for pre-training stage.

    Uses AdamW with Cosine decay (via lr_scheduler_type),
    warmup over 2000 steps, bfloat16 precision.
    """
    return TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        bf16=bf16,
        report_to="none",
        **kwargs,
    )


def get_sft_training_args(
    output_dir: str = "./vi-smollm-135m-sft",
    learning_rate: float = 2e-5,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    logging_steps: int = 10,
    save_strategy: str = "epoch",
    bf16: bool = True,
    max_length: int = 512,
    **kwargs,
):
    """Create SFTConfig for SFT stage."""
    return SFTConfig(
        output_dir=output_dir,
        max_length=max_length,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        bf16=bf16,
        report_to="none",
        **kwargs,
    )


def get_dpo_training_args(
    output_dir: str = "./vi-smollm-135m-censored",
    learning_rate: float = 5e-7,
    num_train_epochs: int = 2,
    per_device_train_batch_size: int = 4,
    logging_steps: int = 10,
    save_strategy: str = "epoch",
    bf16: bool = True,
    beta: float = 0.1,
    **kwargs,
):
    """Create DPOConfig for DPO stage."""
    return DPOConfig(
        output_dir=output_dir,
        beta=beta,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        bf16=bf16,
        report_to="none",
        **kwargs,
    )


def get_data_collator(tokenizer, mlm_probability: float = 0.0):
    """Create a data collator for causal language modeling.

    Args:
        tokenizer: The tokenizer to use.
        mlm_probability: If > 0, enables masked language modeling.
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm_probability > 0,
        mlm_probability=mlm_probability,
    )
