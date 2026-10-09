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
import contextlib
import json
import math
import os
import sys
from datetime import datetime
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
from utils.eval_utils import generate_sample, compute_refusal_rate
from data.dataset_loader import (
    load_culturX_vi,
    load_vi_alpaca,
    load_pkusaferlhf_vi,
    load_pretrain_eval_texts,
    format_for_chatml,
    filter_sft_samples,
    pack_dataset,
)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_tracking_args(config: dict, config_path: str, stage: str) -> dict:
    """TrainingArguments kwargs for experiment tracking, from the config's ``tracking`` block.

    With ``report_to: mlflow`` the HF MLflowCallback logs params and every
    train/eval log on the main process. Defaults to a local SQLite store
    (``sqlite:///mlflow.db``); view it with ``uv run mlflow ui --backend-store-uri sqlite:///mlflow.db``.
    """
    tracking = config.get("tracking") or {}
    report_to = tracking.get("report_to", "none")
    if report_to != "mlflow":
        return {"report_to": report_to}
    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking.get("tracking_uri", "sqlite:///mlflow.db"))
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", tracking.get("experiment_name", "vi-smollm2-135m"))
    os.environ["MLFLOW_TAGS"] = json.dumps({
        "stage": stage,
        "config_path": config_path,
        "config_yaml": yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
    })
    run_name = tracking.get("run_name") or f"{stage}-{datetime.now():%Y%m%d-%H%M}"
    return {"report_to": "mlflow", "run_name": run_name}


@contextlib.contextmanager
def tracking_run(trainer):
    """Re-open the trainer's MLflow run after ``train()`` closed it (main process only).

    Logging after training (final eval, sample generations) would otherwise go
    to a new, auto-created run. Yields the mlflow module, or None.
    """
    if "mlflow" not in (trainer.args.report_to or []) or not trainer.is_world_process_zero():
        yield None
        return
    import mlflow
    last = mlflow.last_active_run()
    if last is None:
        yield None
        return
    with mlflow.start_run(run_id=last.info.run_id):
        yield mlflow


def get_tokenizer(model_dir: str) -> PreTrainedTokenizerFast:
    """Load a PreTrainedTokenizerFast from the specified directory."""
    # save_model() stores only the weights, so a model dir may have no tokenizer files.
    has_tokenizer = os.path.isfile(os.path.join(model_dir, "tokenizer.json"))
    tokenizer_path = model_dir if has_tokenizer else "./vi_smollm_tokenizer"
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    return tokenizer


# ============================================================================
# Stage 1: Pre-training
# ============================================================================


def run_pretrain(config_path: str = "text/configs/pretrain_config.yaml", resume_from_checkpoint: str | None = None):
    """Pre-train the base Vi-SmolLM2-135M model on Vietnamese text.

    Uses HuggingFace Trainer with AdamW + Cosine decay. Documents are tokenized,
    joined with EOS and packed into ``max_length``-token blocks (no padding).
    Eval loss on VTSNLP and held-out CulturaX is logged every ``eval_steps``.
    """
    config = load_config(config_path)

    # Without torchrun, HF Trainer falls back to nn.DataParallel when several
    # GPUs are visible, which crashes here with a CUDA nll_loss assert
    # (`t >= 0 && t < n_classes`). Single GPU or torchrun both work.
    import torch
    if torch.cuda.device_count() > 1 and int(os.environ.get("WORLD_SIZE", "1")) == 1:
        raise RuntimeError(
            f"{torch.cuda.device_count()} GPUs visible but not launched with torchrun. "
            "Use `torchrun --nproc_per_node=N finetuning/train.py ...` or set "
            "CUDA_VISIBLE_DEVICES to a single GPU."
        )
    print("[Stage 1] Pre-training Vi-SmolLM2-135M...")

    # Initialize model
    model = create_model(
        base_model_name=config["model"]["base_model"],
        vocab_size=config["model"]["vocab_size"],
        max_position_embeddings=config["model"]["max_position_embeddings"],
        init_from_base=config["model"].get("init_from_base", False),
    )
    print(f"Model initialized with vocab_size={config['model']['vocab_size']}, "
          f"init_from_base={config['model'].get('init_from_base', False)}")

    # Load tokenizer
    tokenizer = get_tokenizer("./vi_smollm_tokenizer")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"

    is_streaming = config["data"].get("stream", False)
    if is_streaming:
        raise ValueError("Packing needs a non-streaming dataset; set data.stream: false.")
    train_cfg = config["training"]
    eval_cfg = config.get("eval", {})
    eval_steps = eval_cfg.get("eval_steps")
    training_args = get_pretrain_training_args(
        output_dir=train_cfg["output_dir"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 16),
        gradient_accumulation_steps=train_cfg.get("gradient_accumulation_steps", 1),
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg.get("max_steps", -1),
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        save_steps=train_cfg.get("save_steps", 500),
        save_total_limit=train_cfg.get("save_total_limit"),
        eval_strategy="steps" if eval_steps else "no",
        eval_steps=eval_steps,
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 2),
        ddp_timeout=train_cfg.get("ddp_timeout", 1800),
        **get_tracking_args(config, config_path, "pretrain"),
    )

    block_size = config["data"]["max_length"]
    num_proc = config["data"].get("num_proc")
    # Rank 0 tokenizes and fills the datasets cache; other ranks then load from it.
    with training_args.main_process_first(desc="tokenize + pack"):
        dataset = load_culturX_vi(
            dataset_name=config["data"]["dataset_name"],
            dataset_subset=config["data"]["dataset_subset"],
            max_samples=config["data"].get("max_samples"),
            streaming=False,
            data_files=config["data"].get("data_files"),
        )
        train_dataset = pack_dataset(dataset, tokenizer, block_size, num_proc=num_proc).shuffle(seed=42)

        eval_datasets = None
        if eval_steps:
            eval_texts = load_pretrain_eval_texts(
                vtsnlp_docs=eval_cfg.get("vtsnlp_docs", 1000),
                culturax_heldout_file=eval_cfg.get("culturax_heldout_file"),
                culturax_docs=eval_cfg.get("culturax_docs", 1000),
            )
            eval_datasets = {
                name: pack_dataset(Dataset.from_dict({"text": texts}), tokenizer, block_size)
                for name, texts in eval_texts.items()
            }

    n_tokens = len(train_dataset) * block_size
    print(f"[Stage 1] {len(train_dataset):,} blocks x {block_size} = {n_tokens / 1e9:.2f}B training tokens")
    if eval_datasets:
        print("[Stage 1] Eval blocks: " + ", ".join(f"{k}={len(v)}" for k, v in eval_datasets.items()))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        data_collator=get_data_collator(tokenizer),
        processing_class=tokenizer,
    )

    print("[Stage 1] Starting pre-training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(train_cfg["output_dir"])  # weights + tokenizer, main process only
    print(f"[Stage 1] Pre-training complete. Model saved to {train_cfg['output_dir']}")

    if eval_datasets:
        with tracking_run(trainer) as mlflow:
            metrics = trainer.evaluate()
            ppl = {f"final_{name}_ppl": math.exp(metrics[f"eval_{name}_loss"]) for name in eval_datasets}
            for name in eval_datasets:
                print(f"[Stage 1] {name}: eval loss {metrics[f'eval_{name}_loss']:.4f}, "
                      f"PPL {ppl[f'final_{name}_ppl']:.2f}")
            if mlflow:
                mlflow.log_metrics(ppl)


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

    # Load and format dataset as prompt/completion -> loss on the assistant reply only
    data_cfg = config["data"]
    train_cfg = config["training"]
    system_prompt = data_cfg.get("system_prompt", "You are a helpful Vietnamese assistant.")
    raw_dataset = load_vi_alpaca(data_cfg["dataset_name"])
    formatted = format_for_chatml(raw_dataset, tokenizer, system_prompt=system_prompt)
    n_raw = len(formatted)
    formatted = filter_sft_samples(
        formatted, tokenizer,
        min_output_tokens=data_cfg.get("min_output_tokens", 3),
        max_total_tokens=train_cfg["max_length"],
    )
    print(f"[Stage 2] Kept {len(formatted):,}/{n_raw:,} samples after filtering")

    # vi-alpaca only has a train split: hold out a fixed slice for eval loss
    # and for the test_sft notebook (saved next to the model).
    eval_ratio = data_cfg.get("eval_ratio", 0.02)
    split = formatted.train_test_split(test_size=eval_ratio, seed=42)
    train_dataset, eval_dataset = split["train"], split["test"]
    os.makedirs(train_cfg["output_dir"], exist_ok=True)
    heldout_path = os.path.join(train_cfg["output_dir"], "heldout.jsonl")
    eval_dataset.to_json(heldout_path, force_ascii=False)
    print(f"[Stage 2] Train {len(train_dataset):,} / held-out {len(eval_dataset):,} -> {heldout_path}")

    # Setup SFTTrainer
    eval_steps = train_cfg.get("eval_steps")
    sft_config = get_sft_training_args(
        output_dir=train_cfg["output_dir"],
        max_length=train_cfg["max_length"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        max_steps=train_cfg.get("max_steps", -1),
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 16),
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        bf16=train_cfg["bf16"],
        warmup_steps=train_cfg.get("warmup_steps", 0.03),  # < 1 = fraction of total steps
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        eval_strategy="steps" if eval_steps else "no",
        eval_steps=eval_steps,
        completion_only_loss=True,
        **get_tracking_args(config, config_path, "sft"),
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("[Stage 2] Starting SFT...")
    trainer.train()
    # Make generate() stop at the end of an assistant turn (<|im_end|>), not only at </s>.
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    model.generation_config.eos_token_id = [im_end_id, tokenizer.eos_token_id]
    trainer.save_model(train_cfg["output_dir"])  # weights + tokenizer (with chat template)
    print(f"[Stage 2] SFT complete. Model saved to {train_cfg['output_dir']}")
    with tracking_run(trainer) as mlflow:
        metrics = trainer.evaluate()
        print(f"[Stage 2] Held-out eval loss {metrics['eval_loss']:.4f}, PPL {math.exp(metrics['eval_loss']):.2f}")

        # --- Eval: Sample generations ---
        print("[Stage 2] Evaluating generation quality...")
        test_prompts = [
            "Mặt trời mọc ở đâu?",
            "Việt Nam thủ đô là thành phố nào?",
            "Cho tôi một công thức nấu phở bò.",
        ]
        samples = []
        for prompt in test_prompts:
            chat = tokenizer.apply_chat_template(
                [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True,
            )
            response = generate_sample(model, tokenizer, chat, max_new_tokens=128, do_sample=False,
                                       repetition_penalty=1.1)
            print(f"  Prompt: {prompt}")
            print(f"  Response: {response}")
            samples.append(f"### {prompt}\n\n{response}\n")
        if mlflow:
            mlflow.log_metric("final_heldout_ppl", math.exp(metrics["eval_loss"]))
            mlflow.log_text("\n".join(samples), "samples.md")


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
        **get_tracking_args(config, config_path, "dpo"),
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
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume from",
    )
    args = parser.parse_args()

    config_map = {
        "pretrain": "configs/pretrain_config.yaml",
        "sft": "configs/sft_config.yaml",
        "dpo": "configs/dpo_config.yaml",
    }
    config_path = args.config or config_map[args.mode]

    if args.mode == "pretrain":
        run_pretrain(config_path, resume_from_checkpoint=args.resume)
    elif args.mode == "sft":
        run_sft(config_path)
    elif args.mode == "dpo":
        run_dpo(config_path)


if __name__ == "__main__":
    main()
