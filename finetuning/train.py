"""Unified training script for Vi-SmolLM2-135M (Stage 1-3).

Supports three modes:
    --mode pretrain  : Pre-train base model on Vietnamese text
    --mode sft       : Supervised Fine-Tuning on chat data
    --mode dpo       : Direct Preference Optimization for safety alignment

Usage:
    python text/finetuning/train.py --mode pretrain
    python text/finetuning/train.py --mode sft
    python text/finetuning/train.py --mode dpo
"""

import argparse
import os
import sys
from pathlib import Path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import yaml
from typing import cast
from transformers import PreTrainedTokenizerFast, Trainer
from datasets import Dataset
from utils.model_utils import create_model, load_model, save_model
from utils.training_utils import (
    get_pretrain_training_args,
    get_sft_training_args,
    get_dpo_training_args,
    get_data_collator,
)
from utils.eval_utils import compute_perplexity, generate_sample, compute_refusal_rate
from data.dataset_loader import (
    load_culturX_vi,
    load_vi_alpaca,
    load_pkusaferlhf_vi,
    format_for_chatml,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_tokenizer(model_dir: str) -> PreTrainedTokenizerFast:
    """Load a PreTrainedTokenizerFast from the specified directory."""
    tokenizer_path = model_dir if os.path.isdir(model_dir) else "./vi_smollm_tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer


# ============================================================================
# Stage 1: Pre-training
# ============================================================================


def run_pretrain(config_path: str = "text/configs/pretrain_config.yaml"):
    """Pre-train the base Vi-SmolLM2-135M model on Vietnamese text.

    Uses HuggingFace Trainer with AdamW + Cosine decay (warmup 2000, lr=5e-4).
    Data is packed into 2048-token sequences.
    """
    config = load_config(config_path)
    print("[Stage 1] Pre-training Vi-SmolLM2-135M...")

    # Initialize model
    model = create_model(
        base_model_name=config["model"]["base_model"],
        vocab_size=config["model"]["vocab_size"],
        max_position_embeddings=config["model"]["max_position_embeddings"],
    )
    print(f"Model initialized with vocab_size={config['model']['vocab_size']}")

    # Load tokenizer
    tokenizer = get_tokenizer("./vi_smollm_tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"

    # Load dataset
    dataset = load_culturX_vi(
        dataset_name=config["data"]["dataset_name"],
        dataset_subset=config["data"]["dataset_subset"],
        max_samples=config["data"]["max_samples"],
    )

    # Tokenize and pack data
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=config["data"]["max_length"],
            padding="max_length",
        )

    tokenized_dataset = cast(Dataset, dataset.map(tokenize_fn, remove_columns=["text"]))

    # Setup trainer
    training_args = get_pretrain_training_args(
        output_dir=config["training"]["output_dir"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        warmup_steps=config["training"]["warmup_steps"],
        max_steps=config["training"]["max_steps"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
    )

    data_collator = get_data_collator(tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    print("[Stage 1] Starting pre-training...")
    trainer.train()
    save_model(model, config["training"]["output_dir"])
    print(f"[Stage 1] Pre-training complete. Model saved to {config['training']['output_dir']}")

    # --- Eval: Perplexity on a sample of Vietnamese text ---
    print("[Stage 1] Evaluating perplexity...")
    # Use a small held-out Vietnamese text sample
    eval_texts = [
        "Trời đất và vũ trụ rộng lớn biết bao giờ.",
        "Em bé đang chơi đùa trong công viên.",
        "Việt Nam là một đất nước đẹp với nhiều danh lam thắng cảnh.",
    ]
    try:
        ppl = compute_perplexity(model, tokenizer, eval_texts)
        print(f"[Stage 1] Perplexity: {ppl:.2f}")
    except Exception as e:
        print(f"[Stage 1] Perplexity computation failed: {e}")


# ============================================================================
# Stage 2: Supervised Fine-Tuning (SFT)
# ============================================================================


def run_sft(config_path: str = "text/configs/sft_config.yaml"):
    """Supervised Fine-Tune the pre-trained model on Vietnamese chat data.

    Uses TRL SFTTrainer with ChatML formatting.
    Full fine-tuning (not LoRA) since model is only 135M.
    """
    from trl.trainer.sft_trainer import SFTTrainer

    config = load_config(config_path)
    print("[Stage 2] Supervised Fine-Tuning...")

    # Load model and tokenizer
    model = load_model(config["model"]["base_model"])
    tokenizer = get_tokenizer(config["model"]["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"

    # Set chat template
    tokenizer.chat_template = config["tokenizer"]["chat_template"]

    # Load and format dataset
    raw_dataset = load_vi_alpaca(config["data"]["dataset_name"])
    formatted_dataset = format_for_chatml(raw_dataset, tokenizer)

    # Setup SFTTrainer
    sft_config = get_sft_training_args(
        output_dir=config["training"]["output_dir"],
        max_length=config["training"]["max_length"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        bf16=config["training"]["bf16"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=formatted_dataset,
        processing_class=tokenizer,
    )

    print("[Stage 2] Starting SFT...")
    trainer.train()
    save_model(model, config["training"]["output_dir"])
    print(f"[Stage 2] SFT complete. Model saved to {config['training']['output_dir']}")

    # --- Eval: Sample generations ---
    print("[Stage 2] Evaluating generation quality...")
    test_prompts = [
        "Mặt trời mọc ở đâu?",
        "Việt Nam thủ đô là thành phố nào?",
        "Cho tôi một công thức nấu phở bò.",
    ]
    for prompt in test_prompts:
        response = generate_sample(model, tokenizer, prompt, max_new_tokens=64)
        print(f"  Prompt: {prompt}")
        print(f"  Response: {response}")


# ============================================================================
# Stage 3: DPO Safety Alignment (Censored)
# ============================================================================


def run_dpo(config_path: str = "text/configs/dpo_config.yaml"):
    """Apply DPO to align the model with safety behaviors (censoring).

    Uses TRL DPOTrainer with PKU-SafeRLHF-VI data.
    The reference model is auto-cloned from the SFT model.
    """
    from trl.trainer.dpo_trainer import DPOTrainer

    config = load_config(config_path)
    print("[Stage 3] DPO Safety Alignment...")

    # Load model and tokenizer
    model = load_model(config["model"]["base_model"])
    tokenizer = get_tokenizer(config["model"]["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"

    # Load DPO dataset
    raw_dataset = load_pkusaferlhf_vi(config["data"]["dataset_name"])

    # Tokenize for DPO: each sample has prompt, chosen, rejected
    def tokenize_dpo(sample):
        chosen = tokenizer(
            sample["chosen"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
        rejected = tokenizer(
            sample["rejected"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
        prompt = tokenizer(
            sample["prompt"],
            truncation=True,
            max_length=2048,
            padding="max_length",
        )
        return {
            "prompt_input_ids": prompt["input_ids"],
            "prompt_attention_mask": prompt["attention_mask"],
            "chosen_input_ids": chosen["input_ids"],
            "chosen_attention_mask": chosen["attention_mask"],
            "rejected_input_ids": rejected["input_ids"],
            "rejected_attention_mask": rejected["attention_mask"],
        }

    tokenized_dataset = raw_dataset.map(tokenize_dpo)

    # Setup DPOTrainer
    dpo_config = get_dpo_training_args(
        output_dir=config["training"]["output_dir"],
        beta=config["training"]["beta"],
        learning_rate=config["training"]["learning_rate"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        logging_steps=config["training"]["logging_steps"],
        save_strategy=config["training"]["save_strategy"],
        bf16=config["training"]["bf16"],
    )

    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL auto-clones sft_model as reference
        args=dpo_config,
        train_dataset=tokenized_dataset,
        processing_class=tokenizer,
    )

    print("[Stage 3] Starting DPO training...")
    dpo_trainer.train()
    save_model(model, config["training"]["output_dir"])
    print(f"[Stage 3] DPO complete. Model saved to {config['training']['output_dir']}")

    # --- Eval: Refusal rate on harmful prompts ---
    print("[Stage 3] Evaluating refusal rate...")
    from data.dataset_loader import load_abliteration_data
    harmful, _ = load_abliteration_data()
    # Use a sample for speed
    sample_harmful_prompts = harmful[:20]
    refusal_rate = compute_refusal_rate(model, tokenizer, sample_harmful_prompts, max_new_tokens=64)
    print(f"[Stage 3] Refusal rate on {len(sample_harmful_prompts)} harmful prompts: {refusal_rate:.1%}")


# ============================================================================
# Main entry point
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Vi-SmolLM2-135M Unified Training Pipeline"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["pretrain", "sft", "dpo"],
        help="Training stage to execute",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (auto-detected if not provided)",
    )
    args = parser.parse_args()

    config_map = {
        "pretrain": "configs/pretrain_config.yaml",
        "sft": "configs/sft_config.yaml",
        "dpo": "configs/dpo_config.yaml",
    }
    config_path = args.config or config_map[args.mode]

    if args.mode == "pretrain":
        run_pretrain(config_path)
    elif args.mode == "sft":
        run_sft(config_path)
    elif args.mode == "dpo":
        run_dpo(config_path)


if __name__ == "__main__":
    main()
