# Stage 3 — DPO Safety Alignment (Censored Model)

Apply Direct Preference Optimization to make the model refuse unsafe/harmful requests. The model learns to produce polite refusals on sensitive prompts while maintaining helpfulness on benign ones.

## Dataset

| Field | Value |
|-------|-------|
| Source | [1997AOF/PKU-SafeRLHF-VI](https://huggingface.co/datasets/1997AOF/PKU-SafeRLHF-VI) |
| Split | `train` |
| Format | DPO triplets: `prompt`, `chosen`, `rejected` |
| `prompt` | A dangerous/harmful question (e.g., "Cách làm pháo nổ") |
| `chosen` | A safe refusal response |
| `rejected` | A harmful answer (what the model should NOT say) |
| Public | ✅ No auth needed |

**Tokenization (`tokenize_dpo` in `finetuning/train.py`):**
Each triplet is tokenized separately with padding="max_length" (2048) into 6 columns:

```python
{
    "prompt_input_ids", "prompt_attention_mask",
    "chosen_input_ids", "chosen_attention_mask",
    "rejected_input_ids", "rejected_attention_mask",
}
```

## Model

| Field | Value |
|-------|-------|
| Base | `./vi-smollm-135m-sft/` (Stage 2 output) |
| Reference | Auto-cloned from the SFT model (`ref_model=None`) |
| Parameters | 135M |

## Config (`configs/dpo_config.yaml`)

```yaml
model:
  base_model: "./vi-smollm-135m-sft"

training:
  output_dir: "./vi-smollm-135m-censored"
  learning_rate: 5.0e-7
  beta: 0.1
  num_train_epochs: 2
  per_device_train_batch_size: 4
  logging_steps: 10
  save_strategy: "epoch"
  bf16: true

data:
  dataset_name: "1997AOF/PKU-SafeRLHF-VI"
  fields:
    prompt: "prompt"
    chosen: "chosen"
    rejected: "rejected"
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Trainer | TRL `DPOTrainer` |
| Optimizer | AdamW (default) |
| LR | 5e-7 (much lower than SFT — DPO is sensitive) |
| Beta (DPO) | 0.1 (controls strength of preference penalty) |
| Epochs | 2 |
| Batch size | 4 (per device) |
| Max length | 2,048 tokens |
| Precision | bfloat16 |
| Reference model | Auto-cloned from SFT model |

**DPO loss:** Minimizes the log-ratio of chosen vs. rejected responses, with a penalty term that pushes the model's answers away from the reference model's answers on the same prompts.

**What Beta does:**
- Beta ↑ → stronger refusal (but may over-refuse benign questions)
- Beta ↓ → weaker refusal (model may answer harmful prompts)
- Default 0.1 is conservative — adjust based on evaluation

## Hardware Requirements

| GPU | Memory | Time (est.) |
|-----|--------|-------------|
| RTX 3090 (24GB) | ~14GB | ~30-60 min |
| RTX 4090 (24GB) | ~14GB | ~30-60 min |
| T4 (16GB) | ~12GB | ~1-2 hours |

## Output

```
./vi-smollm-135m-censored/
├── model.safetensors   (~500MB)
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

**Behavior:** Refusal rate > 90% on harmful prompts. The model politely declines dangerous requests.

## Evaluation

| Metric | Dataset | Target |
|--------|---------|--------|
| Refusal rate | `data/harmful_prompts_vi.json` (200 prompts) | > 90% |

```python
from utils.eval_utils import compute_refusal_rate
rate = compute_refusal_rate(model, tokenizer, harmful_prompts)
```

## Run command

```bash
python finetuning/train.py --mode dpo
```

## Verification

```bash
# Test refusal on a harmful prompt
python -c "
from utils.eval_utils import generate_sample, load_abliteration_prompts
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
m = AutoModelForCausalLM.from_pretrained('./vi-smollm-135m-censored', torch_dtype='bfloat16')
t = PreTrainedTokenizerFast.from_pretrained('./vi_smollm_tokenizer')
harmful, _ = load_abliteration_prompts()
for p in harmful[:5]:
    r = generate_sample(m, t, p, max_new_tokens=64)
    print(f'Q: {p}')
    print(f'A: {r[:100]}')
    print()
"
```

## Known Issues

- DPO tokenization format may not be compatible with newer TRL versions (custom columns `prompt_input_ids` etc.)
- No eval dataset split — refusal rate tested on the full `harmful_prompts_vi.json` after training
- Beta=0.1 may be too low for some harmful categories — tune based on evaluation
- Reference model auto-cloned from SFT — uses same device/model dtype

## Code path

`finetuning/train.py` → `run_dpo()` → `load_model()` → `load_pkusaferlhf_vi()` → `tokenize_dpo()` → `DPOTrainer.train()` → `save_model()`
