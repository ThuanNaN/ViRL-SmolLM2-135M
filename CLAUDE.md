# CLAUDE.md — Vi-SmolLM2-135M Repository

This file provides guidance to Claude Code when working with code in this repository.

## 💡 Architecture Overview: Modularity and Separation of Concerns

The codebase is a **4-stage Vietnamese LLM training pipeline** built on SmolLM2-135M. Modules are flat (no `text/` wrapper) and organized by concern: `finetuning/`, `data/`, `utils/`, `abliteration/`, `configs/`.

### 📚 Module Structure

| Module | Purpose |
|--------|---------|
| `finetuning/train.py` | Unified entry: `--mode pretrain|sft|dpo` |
| `data/dataset_loader.py` | Dataset loaders + ChatML formatting |
| `data/tokenizer_train.py` | Byte-Level BPE tokenizer training |
| `utils/model_utils.py` | Model create/load/save |
| `utils/training_utils.py` | TrainingArguments + DataCollator factories |
| `abliteration/remove_censorship.py` | Stage 4: weight orthogonalization |
| `configs/*.yaml` | Per-stage hyperparameters |

### 🚀 Common Development Commands & Execution Flow

1. **Tokenizer**: `uv run python data/tokenizer_train.py` — must run first
2. **Stage 1**: `uv run python finetuning/train.py --mode pretrain`
3. **Stage 2**: `uv run python finetuning/train.py --mode sft`
4. **Stage 3**: `uv run python finetuning/train.py --mode dpo`
5. **Stage 4**: `uv run python abliteration/remove_censorship.py`

### 🛠️ Key Utilities and Concepts

- **Tokenizer**: Byte-Level BPE, 32k vocab, trained via `tokenizers` library. Must be trained before any training stage.
- **HuggingFace auth**: CulturX và vi-alpaca là gated datasets — cần `huggingface-cli login` hoặc `HF_TOKEN` trước khi chạy.
- **Model**: Starts from `HuggingFaceTB/SmolLM2-135M`, overrides `vocab_size=32000` and `max_position_embeddings=2048`.
- **Parameter consistency**: When calling `load_culturX_vi()` in `train.py`, the parameter name `dataset_subset` must match the function signature in `dataset_loader.py`.
- **`Trainer` import**: The `Trainer` class from `transformers` must be explicitly imported in `train.py` — used in `run_pretrain()` but separate from `PreTrainedTokenizerFast` and `AutoModelForCausalLM`.

### 🔍 Areas for Future Development (TODO)
Future work should focus on abstracting the pipeline management layer to create a unified API that removes the need for modality-specific entry points. Potentially introduce an overarching `orchestrate.py` script.

---

## 📁 Project-Specific Notes

### Pipeline Stages

| Stage | Script | Config | Dataset | Output |
|-------|--------|--------|---------|--------|
| 1. Pre-train | `finetuning/train.py --mode pretrain` | `configs/pretrain_config.yaml` | `uonlp/CulturX` (vi) | `./vi-smollm-135m-pretrain` |
| 2. SFT | `finetuning/train.py --mode sft` | `configs/sft_config.yaml` | `bkai-foundation-models/vi-alpaca` | `./vi-smollm-135m-sft` |
| 3. DPO | `finetuning/train.py --mode dpo` | `configs/dpo_config.yaml` | `1997AOF/PKU-SafeRLHF-VI` | `./vi-smollm-135m-censored` |
| 4. Abliteration | `abliteration/remove_censorship.py` | — | `data/harmful_prompts_vi.json` | `./vi-smollm-135m-uncensored` |

### Module Structure

All Python modules use `__init__.py` with explicit `__all__` exports:
- `finetuning.__init__` — `run_pretrain`, `run_sft`, `run_dpo`
- `data.__init__` — Dataset loaders + tokenizer training
- `utils.__init__` — Model + training utilities
- `abliteration.__init__` — `run_abliteration` and helpers

### Critical Implementation Details

- **Tokenizer**: Byte-Level BPE with 32k vocabulary, trained via `tokenizers` library. Must be trained before any training stage.
- **Model**: Starts from `HuggingFaceTB/SmolLM2-135M`, overrides `vocab_size=32000` and `max_position_embeddings=2048`.
- **Parameter consistency**: When calling `load_culturX_vi()` in `train.py`, the parameter name `dataset_subset` must match the function signature in `dataset_loader.py`.
- **`Trainer` import**: The `Trainer` class from `transformers` must be explicitly imported in `train.py` — it is used in `run_pretrain()` but is a separate import from `PreTrainedTokenizerFast` and `AutoModelForCausalLM`.

### Data Files

- Abliteration requires `data/harmful_prompts_vi.json` and `data/harmless_prompts_vi.json` (200 prompts each).
- These are loaded by `data/dataset_loader.py` → `load_abliteration_data()` and consumed by `abliteration/remove_censorship.py` → `run_abliteration()`.

### Running the Pipeline

```bash
# Verify all imports work
uv run python -c "import sys; sys.path.insert(0, '.'); import finetuning; print(f'{len(finetuning.__all__)} symbols exported')"

# Check syntax of all Python files
find . -name '*.py' | xargs uv run python3 -m py_compile
```
