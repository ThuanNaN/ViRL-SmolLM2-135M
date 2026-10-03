# Stage 2 — Supervised Fine-Tuning (SFT)

Fine-tune the pre-trained base model on Vietnamese chat conversations (instruction → response format). Uses ChatML prompt formatting.

## Dataset

| Field | Value |
|-------|-------|
| Source | [bkai-foundation-models/vi-alpaca](https://huggingface.co/datasets/bkai-foundation-models/vi-alpaca) |
| Split | `train` |
| Format | Alpaca-style: `instruction`, `input`, `output` |
| Size | ~50K samples (estimated) |
| Gated | Yes — need HF token |

**Preprocessing (`format_for_chatml`):**
Each sample is converted to a ChatML message list:

```python
messages = [
    {"role": "system", "content": "You are a helpful Vietnamese assistant."},
    {"role": "user", "content": instruction},
    # optional: {"role": "user", "content": input}  if input is non-empty
    {"role": "assistant", "content": output},
]
```

The tokenizer's `chat_template` renders this to a single string for training.

## Model

| Field | Value |
|-------|-------|
| Base | `./vi-smollm-135m-pretrain/` (Stage 1 output, loaded via `AutoModelForCausalLM.from_pretrained`) |
| Init | Pretrained weights from Stage 1 |
| Parameters | 135M |

## Config (`configs/sft_config.yaml`)

```yaml
model:
  base_model: "./vi-smollm-135m-pretrain"

training:
  output_dir: "./vi-smollm-135m-sft"
  learning_rate: 2.0e-5
  num_train_epochs: 3
  per_device_train_batch_size: 8
  max_length: 2048
  logging_steps: 10
  save_strategy: "epoch"
  bf16: true

data:
  dataset_name: "bkai-foundation-models/vi-alpaca"
  text_field: "text"

tokenizer:
  chat_template: |
    {% for message in messages %}
    {{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}
    {% endfor %}
    {% if add_generation_prompt %}
    {{'<|im_start|>assistant\n'}}
    {% endif %}
```

## Training Details

| Parameter | Value |
|-----------|-------|
| Trainer | TRL `SFTTrainer` |
| Optimizer | AdamW (default) |
| LR | 2e-5 |
| Epochs | 3 |
| Batch size | 8 (per device) |
| Max length | 2,048 tokens |
| Precision | bfloat16 |
| Chat template | Custom ChatML with `<|im_start|>` / `<|im_end|>` |

## Hardware Requirements

| GPU | Memory | Time (est.) |
|-----|--------|-------------|
| RTX 3090 (24GB) | ~14GB | ~1-2 hours |
| RTX 4090 (24GB) | ~14GB | ~1-2 hours |
| T4 (16GB) | ~12GB | ~3-5 hours |

## Output

```
./vi-smollm-135m-sft/
├── model.safetensors   (~500MB)
├── config.json
├── tokenizer.json
├── tokenizer_config.json
└── special_tokens_map.json
```

## Evaluation

| Metric | Dataset | Target |
|--------|---------|--------|
| Generation quality | Hold-out từ `bkai-foundation-models/vi-alpaca` | Coherent, grammatical Vietnamese |

```python
from utils.eval_utils import generate_sample
response = generate_sample(model, tokenizer, "Mặt trời mọc ở đâu?")
```

Sample test prompts:
```
"Mặt trời mọc ở đâu?"
"Việt Nam thủ đô là thành phố nào?"
"Cho tôi một công thức nấu phở bò."
```

## Run command

```bash
python finetuning/train.py --mode sft
```

## Verification

```bash
# Generate a test response
python -c "
from utils.eval_utils import generate_sample
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast
m = AutoModelForCausalLM.from_pretrained('./vi-smollm-135m-sft', torch_dtype='bfloat16')
t = PreTrainedTokenizerFast.from_pretrained('./vi_smollm_tokenizer')
print(generate_sample(m, t, 'Việt Nam nổi tiếng với món ăn gì?'))
"
```

## Known Issues

- SFT uses full fine-tuning (not LoRA) — 135M fits in GPU memory but slower than LoRA
- No eval split — generation quality checked on hardcoded prompts only
- `max_length` passed to `SFTConfig` but not used by HF trainer internally (only `max_seq_length` matters)

## Code path

`finetuning/train.py` → `run_sft()` → `load_model()` → `load_vi_alpaca()` → `format_for_chatml()` → `SFTTrainer.train()` → `save_model()`

