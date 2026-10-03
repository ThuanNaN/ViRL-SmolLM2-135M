# Stage 0 — Tokenizer Training

Train a Byte-Level BPE tokenizer on Vietnamese text. Must run before any other stage.

## Dataset

| Field | Value |
|-------|-------|
| Source | [uonlp/CulturX](https://huggingface.co/datasets/uonlp/CulturX) |
| Subset | `vi` (Vietnamese) |
| Samples | 1,000,000 lines |
| Field | `text` |

> ⚠️ **Gated dataset:** Need HF token — `huggingface-cli login`

## Config

| Parameter | Value |
|-----------|-------|
| Vocab size | 32,000 |
| Model | Byte-Level BPE |
| Special tokens | `<unk>`, `<s>`, `</s>`, `<pad>`, `<mask_token>`, `<|im_start|>`, `<|im_end|>` |
| Output | `./vi_smollm_tokenizer/` |

## Output

```
./vi_smollm_tokenizer/
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── tokenizer.json
```

## Verification

Sau khi train xong, kiểm tra các requirement:

### 1. Vocab size đúng

```python
from transformers import PreTrainedTokenizerFast
t = PreTrainedTokenizerFast.from_pretrained("./vi_smollm_tokenizer")
assert t.vocab_size == 32000, f"Expected 32000, got {t.vocab_size}"
print(f"✅ Vocab size: {t.vocab_size}")
```

### 2. Special tokens đầy đủ

```python
required = ["<unk>", "<s>", "</s>", "<pad>", "<mask_token>", "<|im_start|>", "<|im_end|>"]
for tok in required:
    assert tok in t.get_added_tokens(), f"Missing: {tok}"
print("✅ Special tokens OK")
```

### 3. Encode/decode tiếng Việt có dấu

```python
text = "Xin chào, đây là thử nghiệm tiếng Việt có dấu: ì ạch"
ids = t.encode(text)
decoded = t.decode(ids)
assert text in decoded or decoded in text, "Encode/decode mismatch"
print(f"✅ Encode/decode OK: '{decoded[:50]}...'")
```

### 4. Chat template được gắn

```python
assert t.chat_template is not None, "Missing chat template"
print("✅ Chat template OK")
```

### 5. Test generation (sanity check)

```python
inputs = t("Việt Nam", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=10)
print(t.decode(outputs[0]))
```

### 6. Check file output

```bash
ls -la ./vi_smollm_tokenizer/
# Expected: tokenizer.json, tokenizer_config.json, special_tokens_map.json
```

## Run command

```bash
python data/tokenizer_train.py
```

## Code path

`data/tokenizer_train.py` → `train_byte_level_bpe()` → `Tokenizer.train_from_iterator()` → `save_pretrained()`

## Notes

- Takes ~10-30 min depending on internet speed (streaming 1M samples)
- Tokenizer is reused by all subsequent stages
- If tokenizers library version mismatch, reload with `PreTrainedTokenizerFast.from_pretrained()`
