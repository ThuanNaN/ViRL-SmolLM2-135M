# Stage 1 — Pre-training (Base Model)

Train Vi-SmolLM2-135M from scratch on Vietnamese text. Starts from the SmolLM2-135M architecture (random weights), overrides vocab_size and max_position_embeddings, then trains on 1M lines of Vietnamese text.

## Dataset

| Field | Value |
|-------|-------|
| Source | [uonlp/CulturX](https://huggingface.co/datasets/uonlp/CulturX) |
| Subset | `vi` (Vietnamese) |
| Split | `train` |
| Streaming | Yes (constant memory regardless of dataset size) |
| Samples | 1,000,000 (`.take(max_samples)`) |
| Field | `text` (raw Vietnamese text) |
| Size | ~500MB-1GB text (estimated) |

> ⚠️ **Cần HuggingFace token:** CulturX là dataset gated — phải accept terms trên HF page rồi login: `huggingface-cli login` hoặc đặt `HF_TOKEN`.

**Preprocessing:** None beyond truncation. Text is tokenized directly with the Byte-Level BPE tokenizer trained in Stage 0.

## Model

| Config | Value |
|--------|-------|
| Base architecture | SmolLM2ForCausalLM |
| Init | Random weights (not pretrained) |
| `vocab_size` | 32,000 (overridden from default) |
| `max_position_embeddings` | 2,048 (overridden from default 2,048 — matches) |
| Parameters | 135M |

## Config (`configs/pretrain_config.yaml`)

```yaml
model:
  base_model: "HuggingFaceTB/SmolLM2-135M"
  vocab_size: 32000
  max_position_embeddings: 2048

training:
  output_dir: "./vi-smollm-135m-pretrain"
  learning_rate: 5.0e-4
  num_train_epochs: 3
  per_device_train_batch_size: 32
  per_device_eval_batch_size: 32
  gradient_accumulation_steps: 1
  warmup_steps: 2000
  lr_scheduler_type: "cosine"
  bf16: true
  logging_steps: 10
  save_strategy: "epoch"

data:
  dataset_name: "uonlp/CulturX"
  dataset_subset: "vi"
  stream: true
  max_samples: 1000000
  max_length: 2048
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (default from HF Trainer) |
| LR schedule | Cosine decay, warmup 2,000 steps |
| Peak LR | 5e-4 |
| Epochs | 3 |
| Batch size | 32 (per device) |
| Sequence length | 2,048 tokens (packed) |
| Precision | bfloat16 |
| Data collator | `DataCollatorForLanguageModeling` (mlm=False, causal LM) |

## Hardware Requirements

| GPU | Memory | Time (est.) |
|-----|--------|-------------|
| RTX 3090 (24GB) | ~12GB | ~2-4 hours |
| RTX 4090 (24GB) | ~12GB | ~2-4 hours |
| T4 (16GB) | ~10GB | ~5-8 hours |
| CPU only | — | Very slow, not recommended |

## Output

```
./vi-smollm-135m-pretrain/
├── model.safetensors   (~500MB)
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

## Evaluation

| Metric | Dataset | Target |
|--------|---------|--------|
| Perplexity | `VTSNLP/vietnamese_curated_dataset` (general Vietnamese text) | Low (< 15) |

```python
from utils.eval_utils import compute_perplexity
ppl = compute_perplexity(model, tokenizer, eval_texts)
```

## Run command

```bash
python finetuning/train.py --mode pretrain
# or with custom config:
python finetuning/train.py --mode pretrain --config configs/pretrain_config.yaml
```

## Verification

```bash
# Check model loads
python -c "from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('./vi-smollm-135m-pretrain'); print('OK')"

# Check perplexity
python -c "
from utils.eval_utils import compute_perplexity
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
m = AutoModelForCausalLM.from_pretrained('./vi-smollm-135m-pretrain', torch_dtype='bfloat16')
t = PreTrainedTokenizerFast.from_pretrained('./vi_smollm_tokenizer')
print(f'Perplexity: {compute_perplexity(m, t, [\"Vietnamese test sentence.\"]):.2f}')
"
```

## Known Issues

- No eval dataset split — perplexity computed on a few hardcoded sentences (not representative)
- `Trainer` has no `eval_dataset` — perplexity computed manually after training
- Gradient accumulation = 1, so effective batch = 32 (may be too large for some GPUs)

## Code path

`finetuning/train.py` → `run_pretrain()` → `create_model()` → `load_culturX_vi()` → `tokenize_fn()` → `Trainer.train()` → `save_model()`

