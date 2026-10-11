"""Shared training utilities: optimizer, scheduler, and trainer helpers."""

from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments
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
        **{"report_to": "none", **kwargs},
    )


def get_sft_training_args(
    output_dir: str = "./vi-smollm-135m-sft",
    learning_rate: float = 1e-4,
    num_train_epochs: int = 3,
    per_device_train_batch_size: int = 8,
    logging_steps: int = 10,
    save_strategy: str = "epoch",
    bf16: bool = True,
    max_length: int = 1024,
    **kwargs,
):
    """Create SFTConfig for SFT stage.

    With a prompt/completion dataset, TRL computes the loss on the completion only.
    """
    return SFTConfig(
        output_dir=output_dir,
        max_length=max_length,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        logging_steps=logging_steps,
        save_strategy=save_strategy,
        bf16=bf16,
        **{"report_to": "none", **kwargs},
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
        **{"report_to": "none", **kwargs},
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


class LossSpikeCallback(TrainerCallback):
    """Stop training when the train loss blows up (divergence).

    Tracks an EMA of the logged train loss. If the EMA stays more than
    ``threshold`` above the lowest EMA seen for ``patience`` consecutive logs,
    training stops (with ``load_best_model_at_end`` the best checkpoint is then
    restored). The logged loss is already averaged across ranks, so every rank
    makes the same decision. Logs before ``warmup_steps`` are ignored.
    """

    def __init__(self, threshold: float = 0.5, patience: int = 20, warmup_steps: int = 2000, ema: float = 0.9):
        self.threshold = threshold
        self.patience = patience
        self.warmup_steps = warmup_steps
        self.alpha = ema
        self.ema = None
        self.best = float("inf")
        self.bad = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is None or state.global_step < self.warmup_steps:
            return
        self.ema = loss if self.ema is None else self.alpha * self.ema + (1 - self.alpha) * loss
        self.best = min(self.best, self.ema)
        self.bad = self.bad + 1 if self.ema > self.best + self.threshold else 0
        if self.bad >= self.patience:
            print(f"[LossSpikeCallback] step {state.global_step}: loss EMA {self.ema:.3f} > best "
                  f"{self.best:.3f} + {self.threshold} for {self.patience} logs, stopping.")
            control.should_training_stop = True


class EvalMetricEarlyStopping(EarlyStoppingCallback):
    """``EarlyStoppingCallback`` that skips evals lacking ``metric_for_best_model``.

    With several eval datasets the Trainer fires ``on_evaluate`` once per
    dataset, so the evals of the other datasets would log a warning each time.
    """

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        key = args.metric_for_best_model
        key = key if key.startswith("eval_") else f"eval_{key}"
        if key in metrics:
            super().on_evaluate(args, state, control, metrics, **kwargs)


def get_early_stopping_callbacks(cfg: dict | None) -> list:
    """Build early-stopping callbacks from a config's ``early_stopping`` block.

    - ``patience``: evals without improvement of ``metric_for_best_model``
      (needs ``load_best_model_at_end``) -> ``EvalMetricEarlyStopping``.
    - ``loss_spike_threshold`` / ``loss_spike_patience`` / ``loss_spike_warmup``
      -> ``LossSpikeCallback`` on the train loss.
    """
    cfg = cfg or {}
    callbacks = []
    if cfg.get("patience"):
        callbacks.append(EvalMetricEarlyStopping(
            early_stopping_patience=cfg["patience"],
            early_stopping_threshold=cfg.get("threshold", 0.0),
        ))
    if cfg.get("loss_spike_threshold"):
        callbacks.append(LossSpikeCallback(
            threshold=cfg["loss_spike_threshold"],
            patience=cfg.get("loss_spike_patience", 20),
            warmup_steps=cfg.get("loss_spike_warmup", 2000),
        ))
    return callbacks
