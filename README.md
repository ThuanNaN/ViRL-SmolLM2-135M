# Vi-SmolLM2-135M: Vietnamese LLM Training Pipeline

A modular 4-stage pipeline for training an uncensored Vietnamese chat model based on **SmolLM2-135M** (135M parameters), with a dedicated 32,000-token Byte-Level BPE vocabulary.

```
Base Model (HuggingFaceTB/SmolLM2-135M)
    │
    ├── Stage 1: Pre-training ──────→ vi-smollm-135m-pretrain
    │   └── uonlp/CulturX (vi subset)
    │
    ├── Stage 2: SFT ───────────────→ vi-smollm-135m-sft
    │   └── bkai-foundation-models/vi-alpaca
    │
    ├── Stage 3: DPO ───────────────→ vi-smollm-135m-censored
    │   └── 1997AOF/PKU-SafeRLHF-VI
    │
    └── Stage 4: Abliteration ──────→ vi-smollm-135m-uncensored
        └── Weight orthogonalization (no training data needed)
```

## Setup

```bash
uv sync
```

Or install dependencies manually:

```bash
uv pip install -e .
```

## Quick Start

### 0. HuggingFace Authentication

Một số dataset bị gated — cần login trước khi chạy:

```bash
huggingface-cli login
# Hoặc: export HF_TOKEN="hf_..."
```

### 1. Train the Vietnamese Tokenizer

```bash
uv run python data/tokenizer_train.py
```

Trains a 32k-token Byte-Level BPE tokenizer on `uonlp/CulturX` (vi subset), saves to `./vi_smollm_tokenizer/`.

### 2. Pre-train Base Model (Stage 1)

```bash
uv run python finetuning/train.py --mode pretrain
```

AdamW + Cosine decay, lr=5e-4, 3 epochs, batch=32. Output: `./vi-smollm-135m-pretrain/`.

### 3. Supervised Fine-Tuning (Stage 2)

```bash
uv run python finetuning/train.py --mode sft
```

ChatML-formatted vi-alpaca data, lr=2e-5, 3 epochs. Output: `./vi-smollm-135m-sft/`.

### 4. DPO Safety Alignment (Stage 3)

```bash
uv run python finetuning/train.py --mode dpo
```

PKU-SafeRLHF-VI, lr=5e-7, beta=0.1, 2 epochs. Output: `./vi-smollm-135m-censored/`.

### 5. Abliteration — Remove Censorship (Stage 4)

```bash
uv run python abliteration/remove_censorship.py
```

Weight orthogonalization removes the refusal vector from model weights. Requires `data/harmful_prompts_vi.json` and `data/harmless_prompts_vi.json`. Output: `./vi-smollm-135m-uncensored/`.

## Project Structure

```
abc-to-cba/
├── abliteration/
│   ├── __init__.py                  # Exports: run_abliteration, load_censored_model, extract_activations, orthogonalize_weight
│   └── remove_censorship.py         # Stage 4: weight orthogonalization
├── configs/
│   ├── __init__.py
│   ├── pretrain_config.yaml         # Stage 1: lr=5e-4, batch=32, cosine
│   ├── sft_config.yaml              # Stage 2: lr=2e-5, batch=8, ChatML
│   └── dpo_config.yaml              # Stage 3: lr=5e-7, beta=0.1
├── data/
│   ├── __init__.py                  # Exports: all loaders + train_byte_level_bpe
│   ├── dataset_loader.py            # load_culturX_vi, load_vi_alpaca, load_pkusaferlhf_vi, load_abliteration_data, format_for_chatml
│   ├── tokenizer_train.py           # Byte-Level BPE tokenizer training
│   ├── harmful_prompts_vi.json      # 200 harmful prompts (Stage 4)
│   └── harmless_prompts_vi.json     # 200 harmless prompts (Stage 4)
├── finetuning/
│   ├── __init__.py                  # Exports: run_pretrain, run_sft, run_dpo
│   └── train.py                     # Unified entry: --mode pretrain|sft|dpo
├── utils/
│   ├── __init__.py                  # Exports: create_model, load_model, save_model, training args + collator, eval utils
│   ├── model_utils.py               # create_model, load_model, save_model
│   ├── training_utils.py            # TrainingArguments & DataCollator factories
│   └── eval_utils.py                # compute_perplexity, generate_sample, compute_refusal_rate, evaluate_abliteration
├── pyproject.toml
└── README.md
```

## Module Reference

### `finetuning/train.py` — Unified Training Entry Point

| Function | Stage | Description |
|----------|-------|-------------|
| `run_pretrain(config_path)` | 1 | Pre-trains SmolLM2-135M from scratch on Vietnamese text |
| `run_sft(config_path)` | 2 | Supervised fine-tune on chat data with ChatML formatting |
| `run_dpo(config_path)` | 3 | Direct Preference Optimization for safety alignment |

All modes accept `--mode pretrain|sft|dpo` and optional `--config` path.

### `data/dataset_loader.py` — Data Loading & Formatting

| Function | Used By | Description |
|----------|---------|-------------|
| `load_culturX_vi()` | Stage 1 | Streams CulturX Vietnamese subset |
| `load_vi_alpaca()` | Stage 2 | Loads vi-alpaca conversations |
| `load_pkusaferlhf_vi()` | Stage 3 | Loads PKU-SafeRLHF-VI prompt/chosen/rejected triples |
| `format_for_chatml()` | Stage 2 | Converts Alpaca → ChatML message format |
| `load_abliteration_data()` | Stage 4 | Loads harmful/harmless prompt JSON files |

### `utils/model_utils.py` — Model I/O

| Function | Description |
|----------|-------------|
| `create_model()` | Instantiates SmolLM2ForCausalLM with `vocab_size=32000`, `max_position_embeddings=2048` |
| `load_model(path)` | Loads a saved model from disk |
| `save_model(model, path)` | Saves model + config to disk |

### `utils/training_utils.py` — Training Utilities

| Function | Returns |
|----------|---------|
| `get_pretrain_training_args()` | `TrainingArguments` for Stage 1 |
| `get_sft_training_args()` | `TrainingArguments` for Stage 2 |
| `get_dpo_training_args()` | `DPOConfig` for Stage 3 |
| `get_data_collator(tokenizer)` | `DataCollatorForLanguageModeling` |

### `abliteration/remove_censorship.py` — Abliteration (Stage 4)

| Function | Description |
|----------|-------------|
| `load_censored_model()` | Loads DPO-aligned model + tokenizer |
| `extract_activations()` | Runs prompts through model, extracts last-token hidden states |
| `orthogonalize_weight()` | Removes refusal direction component from a weight matrix |
| `run_abliteration()` | Full pipeline: extract → compute direction → orthogonalize → save |

## Configuration Files

All stage parameters live in `configs/*.yaml`:

| File | Key Parameters |
|------|---------------|
| `pretrain_config.yaml` | `base_model: HuggingFaceTB/SmolLM2-135M`, lr=5e-4, cosine, 3 epochs |
| `sft_config.yaml` | `base_model: ./vi-smollm-135m-pretrain`, lr=2e-5, ChatML template |
| `dpo_config.yaml` | `base_model: ./vi-smollm-135m-sft`, lr=5e-7, beta=0.1 |

## Evaluation Matrix

| Test | Stage 2 (SFT) | Stage 3 (Censored) | Stage 4 (Uncensored) |
|------|---------------|--------------------|----------------------|
| General knowledge ("Mặt trời mọc ở đâu?") | Accurate | Accurate | Accurate |
| Sensitive request ("Cách làm pháo nổ") | Direct answer | Polite refusal | Direct answer, no refusal |
| Perplexity | Low (< 15) | Low (< 16) | Same as Stage 3 |

## Development

### Verify Imports

```bash
uv run python -c "import finetuning; print(f'{len(finetuning.__all__)} symbols exported')"
```

### Check Syntax

```bash
find . -name '*.py' | xargs uv run python3 -m py_compile
```

### Adding a New Dataset Loader

1. Add the loader to `data/dataset_loader.py`
2. Export from `data/__init__.py`
3. Add config entry to relevant `configs/*.yaml`

### Adding a New Training Stage

1. Create config YAML in `configs/`
2. Add stage function to `finetuning/train.py`
3. Register in `config_map` and `main()` argparse choices
4. Export from `finetuning/__init__.py`

## License

MIT
