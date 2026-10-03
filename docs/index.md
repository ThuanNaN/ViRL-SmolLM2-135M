# Vi-SmolLM2-135M — Pipeline Documentation

4-stage training pipeline producing a fully uncensored Vietnamese chat model from SmolLM2-135M.

```
[Stage 0] Tokenizer          → 32k Byte-Level BPE vocabulary
      │
[Stage 1] Pre-training       → vi-smollm-135m-pretrain      (random init, CulturX vi)
      │
[Stage 2] SFT                → vi-smollm-135m-sft           (vi-alpaca, ChatML)
      │
[Stage 3] DPO Alignment      → vi-smollm-135m-censored      (PKU-SafeRLHF-VI, refusal)
      │
[Stage 4] Abliteration       → vi-smollm-135m-uncensored    (weight orthogonalization)
```

## Stages

| Stage | Doc | Dataset | Model | Output | Eval Dataset |
|-------|-----|---------|-------|--------|--------------|
| 0 | [Stage 0](stage0_tokenizer.md) | uonlp/CulturaX (vi) | — | `./vi_smollm_tokenizer/` | — |
| 1 | [Stage 1 — Pre-training](stage1_pretrain.md) | uonlp/CulturaX (vi), 1M samples | Random SmolLM2-135M | `./vi-smollm-135m-pretrain/` | `VTSNLP/vietnamese_curated_dataset` |
| 2 | [Stage 2 — SFT](stage2_sft.md) | bkai-foundation-models/vi-alpaca | Stage 1 weights | `./vi-smollm-135m-sft/` | Hold-out vi-alpaca |
| 3 | [Stage 3 — DPO](stage3_dpo.md) | 1997AOF/PKU-SafeRLHF-VI | Stage 2 weights | `./vi-smollm-135m-censored/` | `data/harmful_prompts_vi.json` |
| 4 | [Stage 4 — Abliteration](stage4_abliteration.md) | `data/harmful_prompts_vi.json` + `harmless_prompts_vi.json` | Stage 3 weights | `./vi-smollm-135m-uncensored/` | `data/harmful+harmless_prompts_vi.json` |

## Quick Reference

```bash
# Trước khi chạy — cần login HuggingFace (CulturX và vi-alpaca là gated)
huggingface-cli login
# Hoặc: export HF_TOKEN="hf_..."

# Stage 0 — Tokenizer (run first)
python data/tokenizer_train.py

# Stage 1 — Pre-training
python finetuning/train.py --mode pretrain

# Stage 2 — SFT
python finetuning/train.py --mode sft

# Stage 3 — DPO
python finetuning/train.py --mode dpo

# Stage 4 — Abliteration (no training, single script)
python abliteration/remove_censorship.py
```

## Eval Summary

| Stage | Metric | Tool | Target |
|-------|--------|------|--------|
| 1 | Perplexity | `compute_perplexity()` | < 15 |
| 2 | Generation quality | `generate_sample()` | Coherent Vietnamese |
| 3 | Refusal rate | `compute_refusal_rate()` | > 90% |
| 4 | Refusal rate (before/after) | `evaluate_abliteration()` | Harmful < 10%, Harmless < 30% |

## Key Design Decisions

- **Full fine-tuning, not LoRA** — model is only 135M, full training fits in memory
- **bf16 throughout** — optimal memory/precision tradeoff for SmolLM2
- **DPO creates censorship on purpose** — Stage 3 trains refusal; Stage 4 removes it via abliteration
- **No unified `--mode` for Stage 4** — abliteration is a separate script, not part of `train.py`
- **Eval after each stage** — verify model works before moving to next stage
