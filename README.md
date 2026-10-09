# Vi-SmolLM2-135M: Vietnamese LLM Training Pipeline

A modular 4-stage pipeline for training an uncensored Vietnamese chat model based on **SmolLM2-135M** (135M parameters), with a dedicated 32,000-token Byte-Level BPE vocabulary.

```
Base Model (HuggingFaceTB/SmolLM2-135M)
    │
    ├── Stage 1: Pre-training ──────→ vi-smollm-135m-pretrain
    │   └── uonlp/CulturaX (vi subset)
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

## Quick Start

### 0. HuggingFace Authentication

Một số dataset bị gated — cần login trước khi chạy:

```bash
hf auth login
# Hoặc: export HF_TOKEN="hf_..."
```

### Experiment tracking (MLflow local)

Mỗi stage (`pretrain`, `sft`, `dpo`) log vào MLflow nếu config có `tracking.report_to: "mlflow"` (mặc định bật):

- Store local: `sqlite:///mlflow.db` (metrics/params) + `./mlruns/` (artifacts), tạo ở thư mục chạy lệnh (repo root).
- Experiment `vi-smollm2-135m`, run tên `<stage>-<YYYYmmdd-HHMM>` (đặt `tracking.run_name` để đổi).
- Log: toàn bộ TrainingArguments (params), train loss / lr / grad_norm mỗi `logging_steps`, eval loss
  (`eval_vtsnlp_loss`, `eval_culturax_heldout_loss` ở Stage 1; `eval_loss` held-out ở Stage 2), PPL cuối
  (`final_*_ppl`), tag `stage` + `config_yaml` (nguyên config), và `samples.md` (câu trả lời mẫu sau SFT).
- Chỉ process chính log khi chạy `torchrun`. Checkpoint/model không upload vào MLflow (vẫn nằm ở `output_dir`).

Xem UI:

```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# máy remote: ssh -L 5000:localhost:5000 <host>, rồi mở http://localhost:5000
```

Tắt: đặt `tracking.report_to: "none"` trong config.

### 1. Train the Vietnamese Tokenizer

```bash
uv run python data/tokenizer_train.py
```

Trains a 32k-token Byte-Level BPE tokenizer on the first 1M docs of `uonlp/CulturaX` (vi, streamed in
shard order = all of `vi_part_00000` + ~360k docs of `vi_part_00001`), saves to `./vi_smollm_tokenizer/`.

#### Thí nghiệm: lượng dữ liệu train tokenizer

Train lại tokenizer cùng cấu hình (BPE 32k, cùng special tokens) trên N docs đầu, đo token/từ trên dữ liệu
chưa thấy (càng thấp càng tốt; "từ" = âm tiết tách theo dấu cách, 2,000 docs mỗi tập):

| Dữ liệu train tokenizer | CulturaX `00010` | CulturaX `00089` | VTSNLP | Thời gian |
|---|---:|---:|---:|---:|
| 250k docs | 1.1973 | 1.1980 | 1.2426 | 1.0 phút |
| **1M docs (đang dùng)** | 1.1972 | 1.1977 | 1.2426 | 3.7 phút |
| 4M docs (~6 shard) | 1.1972 | 1.1976 | 1.2426 | 11.7 phút |

Tạo lại bằng `uv run python scripts/tokenizer_data_scaling.py 250000 1000000 4000000`.

- Dữ liệu ×16 chỉ giảm ~0.03% token/từ; tokenizer đã bão hoà từ ~250k docs (tiếng Việt chỉ có vài nghìn âm
  tiết thông dụng). **Không cần train lại tokenizer** với nhiều dữ liệu hơn.
- Không overfit shard train: token/từ trên `vi_part_00000` (1.201) bằng shard chưa thấy (1.197–1.198); 87.9%
  từ là đúng 1 token; 30,276/32,000 token xuất hiện trong 5,000 docs chưa thấy; không có `<unk>` trên VTSNLP.
- Muốn nén tốt hơn phải tăng vocab, nhưng với 135M tham số thì embedding to thêm mà ít được train hơn.

### 2. Pre-train Base Model (Stage 1)

**2a. Pre-fetch CulturaX (vi) shards** (train `vi_part_00000`–`00009`, eval `vi_part_00010`):

```bash
# Download only the shards you need into the HF cache (skipped if already cached)
hf download uonlp/CulturaX --repo-type dataset --include "vi/vi_part_0000[0-9].parquet" --include "vi/vi_part_00010.parquet"
```

`data.data_files` in `configs/pretrain_config.yaml` selects the training shards (`null` = all 90 shards of `vi`,
~50B tokens). `max_samples: null` uses every doc. Packing needs `stream: false`.

**2b. Train**

**1 GPU:**

```bash
uv run python finetuning/train.py --mode pretrain
```

**Multi-GPU (3 GPU trên 1 máy):**

```bash
NCCL_SHM_DISABLE=1 nohup uv run torchrun --nproc_per_node=3 finetuning/train.py --mode pretrain > pretrain.log 2>&1 &
```

`NCCL_SHM_DISABLE=1`: trên máy này GPU2 nằm ở CPU socket khác, NCCL từng lỗi
`Error while attaching to shared memory segment`. Bỏ biến này nếu máy không gặp lỗi đó.

Cấu hình hiện tại (`configs/pretrain_config.yaml`):

- **Packing**: mỗi doc được tokenize, nối `</s>`, ghép liền nhau rồi cắt thành block 2048 token, không padding.
- **Dữ liệu**: shard `vi_part_00000`–`00009` (~6.4M docs, ước tính ~5.6B token), **1 epoch** (`max_steps: -1`,
  số step tự tính). Token đã tokenize lưu int32 trong HF datasets cache (~50 GB).
- **Init**: `init_from_base: true` giữ các lớp transformer của SmolLM2-135M, chỉ khởi tạo lại embedding (tied
  với `lm_head`) vì tokenizer mới. `false` = random init.
- **Eval** mỗi 2,000 step: `eval_vtsnlp_loss` (1,000 docs đầu `VTSNLP/vietnamese_curated_dataset`) và
  `eval_culturax_heldout_loss` (1,000 docs đầu `vi_part_00010`), cũng đã pack 2048.
- AdamW + cosine, lr=5e-4, warmup 2,000 steps, batch 8/GPU (global 24 trên 3 GPU), bf16.
- Checkpoint mỗi 5,000 step, giữ 3 bản (`save_total_limit`). `ddp_timeout: 14400` vì rank 0 tokenize + pack
  trong khi các rank khác chờ.
- Output: `./vi-smollm-135m-pretrain/` (model + tokenizer).

> **Xoá checkpoint cũ trước khi train lại vào cùng thư mục.** `save_total_limit` giữ checkpoint có số step lớn
> nhất, nên `checkpoint-80000` cũ sẽ được giữ còn checkpoint mới bị xoá:
> `rm -rf vi-smollm-135m-pretrain/checkpoint-*`.

> **Multi-GPU phải dùng `torchrun`.** Chạy `uv run python ...` với nhiều GPU hiển thị sẽ rơi vào
> `nn.DataParallel` và crash với CUDA `nll_loss` assert (`t >= 0 && t < n_classes`).
> `train.py` giờ báo lỗi rõ ràng trong trường hợp này. Dùng `CUDA_VISIBLE_DEVICES=0` cho 1 GPU.

**Resume từ checkpoint:**

```bash
torchrun --nproc_per_node=3 finetuning/train.py --mode pretrain \
  --resume ./vi-smollm-135m-pretrain/checkpoint-<step> 2>&1 | tee train_resume.log
```

#### Pre-training results — run 1 (cấu hình cũ)

Kết quả dưới đây là **run đầu tiên** với cấu hình cũ: random init, 4 shard (`vi_part_00000`–`00003`, 2M docs),
mỗi doc pad/cắt riêng về 2048 token, `max_steps=80000` (~0.96 epoch). Run bị dừng ở step 41,667 và được resume.

![Pre-training loss and LR schedule](docs/assets/pretrain_loss.png)

*Loss lấy từ `checkpoint-80000/trainer_state.json`; tạo lại bằng
`uv run --with matplotlib python scripts/plot_loss.py --resume-step 41667`.*

| Step | Train loss |
|------|-----------:|
| 10 | 10.87 |
| 500 | 7.49 |
| 2,000 | 4.80 |
| 10,000 | 3.39 |
| 41,660 | 3.12 |
| 80,000 | 3.03 |

Loss giảm rất nhanh trong ~5k step đầu rồi gần như phẳng (3.12 → 3.03 trong nửa sau của run).

Đánh giá bằng `notebooks/test_pretrain.ipynb` (300 docs, tối đa 512 token/doc):

| Checkpoint | VTSNLP PPL | CulturaX held-out PPL | Fact (MC, 6 câu) |
|------------|-----------:|----------------------:|-----------------:|
| checkpoint-41667 | 56.2 | 20.9 | 67% |
| checkpoint-80000 | **54.8** | **20.4** | 67% |

- `VTSNLP` = `VTSNLP/vietnamese_curated_dataset`, eval set của Stage 1 theo plan; held-out CulturaX của run 1 = shard `vi_part_00004` (notebook giờ dùng `vi_part_00010` vì `00004` nằm trong dữ liệu train mới).
- PPL trên VTSNLP (54.8) còn xa mục tiêu `< 15` trong `docs/stage1_pretrain.md`. Model chỉ train trên CulturaX (web crawl) nên có domain shift; PPL held-out CulturaX là 20.4.
- Đây là base model: sinh văn bản tiếng Việt trôi chảy nhưng hay lặp và chưa trả lời được câu hỏi / chưa nắm chắc fact (đúng "Hà Nội", sai "thành phố lớn nhất"). Cải thiện sau SFT (Stage 2).
- Loss/PPL hầu như phẳng ở nửa sau, nên train thêm cùng dữ liệu ít giá trị; cần thêm dữ liệu nếu muốn PPL thấp hơn.

#### Phân tích run 1 và các thay đổi

| Vấn đề ở run 1 | Số đo | Thay đổi |
|---|---|---|
| Padding chiếm phần lớn compute | Doc CulturaX trung bình 875 token (median 604); chỉ **37.8%** mỗi block 2048 là token thật; 8.9% doc bị cắt đuôi (3,000 docs của `vi_part_00000`) | Packing, không padding: ~2.6× token thật trên cùng compute |
| Quá ít token cho model train từ đầu | 1.92M chuỗi × ~775 token thật ≈ **1.5B token** (Chinchilla cho 135M ≈ 2.7B) | 10 shard, 1 epoch ≈ 5.6B token |
| Random init | — | Init từ SmolLM2-135M (xem bảng dưới) |
| Không có eval trong lúc train | — | Eval loss VTSNLP + CulturaX `00010` mỗi 2,000 step |

Smoke test 300 step trên 1 GPU (20k docs, warmup 50, cùng các thay đổi trên), eval loss ở step 300:

| Init | VTSNLP loss | CulturaX `00010` loss |
|---|---:|---:|
| Random (`init_from_base: false`) | 7.33 | 6.85 |
| SmolLM2-135M (`init_from_base: true`) | **6.57** | **6.20** |

Init từ SmolLM2 thấp hơn ~0.7 loss ngay từ đầu; đây mới là lợi thế giai đoạn đầu, chưa chắc còn đến cuối run.
Tốc độ đo được: ~1.8–1.9 it/s × 8 × 2048 ≈ 29k token/s trên 1 GPU A5000.

### 3. Supervised Fine-Tuning (Stage 2)

**1 GPU:**

```bash
uv run python finetuning/train.py --mode sft
```

**Multi-GPU:**

```bash
torchrun --nproc_per_node=3 finetuning/train.py --mode sft 2>&1 | tee train_sft.log
```

Cấu hình hiện tại (`configs/sft_config.yaml`):

- **Prompt/completion**: mỗi mẫu là `prompt = [system, user]`, `completion = [assistant]` (ChatML). TRL chỉ tính
  loss trên câu trả lời (gồm `<|im_end|>`), không học lại system/user. `instruction` và `input` gộp vào 1 lượt user.
- **Lọc**: bỏ mẫu có output < 3 token hoặc tổng > 1,024 token → giữ 49,372/50,006.
- **Held-out**: tách 2% (988 mẫu, seed 42), eval loss mỗi 500 step, lưu ở `vi-smollm-135m-sft/heldout.jsonl`.
- lr=1e-4, cosine, warmup 3%, 3 epochs, batch 8, `max_length` 1,024.
- Base: `./vi-smollm-135m-pretrain`. Output: `./vi-smollm-135m-sft/` (model + tokenizer, `generation_config`
  dừng ở `<|im_end|>`). Dùng 1 GPU (`CUDA_VISIBLE_DEVICES=0`) là đủ cho 135M + 50k mẫu.

#### SFT results — run 1 (cấu hình cũ)

Cấu hình cũ: loss trên toàn bộ chuỗi (cả system/user), lr=2e-5 linear, 50,006 mẫu, không held-out; base là
model pretrain run 1. Run thực tế: 18,753 steps (3 epochs), ~79 phút trên 1 GPU, ~36M token.

![SFT loss and LR schedule](docs/assets/sft_loss.png)

*Tạo lại bằng `uv run --with matplotlib python scripts/plot_loss.py --state vi-smollm-135m-sft/checkpoint-18753/trainer_state.json --out docs/assets/sft_loss.png --name SFT --lr-title "LR schedule (linear decay)"`.*

Loss giảm 3.47 → 2.16 (token accuracy 0.44 → 0.57) và đã phẳng từ khoảng step 2,300, nên 2 epoch sau gần như không cải thiện thêm.

Đánh giá bằng `notebooks/test_sft.ipynb` (30 prompt viết tay, greedy, `max_new_tokens=150`):

| | base (Stage 1) | SFT (Stage 2) |
|---|---:|---:|
| Tự dừng ở `<\|im_end\|>` | 0/30 | **15/30** (CI 33–67%) |
| Lặp 3-gram | 0.175 | **0.065** |
| Fact đúng (từ khoá) | 1/10 | 0/10 |

- SFT dạy được **format** (trả lời rồi dừng, ít lặp) nhưng **chưa dạy được kiến thức**: cả 10 câu fact đều sai (ví dụ "Thủ đô của Việt Nam là gì?" không ra Hà Nội). Nút thắt là chất lượng pretrain (PPL ~55 trên VTSNLP).
- **Run 1 không có held-out**: `vi-alpaca` chỉ có split `train` và run 1 train trên toàn bộ. 30 prompt trong notebook là viết tay và nhỏ (CI rộng), chỉ nên xem là so sánh tương đối với base. Cấu hình hiện tại đã tách 2% held-out.
- Loss trên cả prompt và lr thấp góp phần vào việc model lặp lại prompt và chỉ dừng đúng 50%; cấu hình hiện tại sửa cả hai.
- Model SFT lưu trước bản sửa chỉ có `eos_token_id=[2, 0]` (không có `<|im_end|>`); `generate_sample` trong `utils/eval_utils.py` tự dừng ở `<|im_end|>`, và `run_sft` giờ lưu đúng stop token cho lần train sau. Khi tự gọi `model.generate()` với model cũ, hãy truyền `eos_token_id=[2, 6]`.

### 4. DPO Safety Alignment (Stage 3)

**1 GPU:**

```bash
uv run python finetuning/train.py --mode dpo
```

**Multi-GPU:**

```bash
torchrun --nproc_per_node=3 finetuning/train.py --mode dpo 2>&1 | tee train_dpo.log
```

PKU-SafeRLHF-VI, lr=5e-7, beta=0.1, 2 epochs. Output: `./vi-smollm-135m-censored/`.

> **Chưa chạy được:** `1997AOF/PKU-SafeRLHF-VI` trả về `DatasetNotFoundError` dù đã đăng nhập, và tìm trên Hub
> không thấy (có thể đã private/xoá). Cần thay dataset (ví dụ dịch `PKU-Alignment/PKU-SafeRLHF`) trước Stage 3.

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
│   ├── pretrain_config.yaml         # Stage 1: lr=5e-4, batch=8/GPU, cosine, packing 2048
│   ├── sft_config.yaml              # Stage 2: lr=1e-4, batch=8, ChatML prompt/completion
│   └── dpo_config.yaml              # Stage 3: lr=5e-7, beta=0.1
├── data/
│   ├── __init__.py                  # Exports: all loaders + train_byte_level_bpe
│   ├── dataset_loader.py            # load_culturX_vi, load_vi_alpaca, load_pkusaferlhf_vi, load_abliteration_data, format_for_chatml,
│   │                                #   filter_sft_samples, pack_dataset, load_pretrain_eval_texts
│   ├── tokenizer_train.py           # Byte-Level BPE tokenizer training
│   ├── harmful_prompts_vi.json      # 200 harmful prompts (Stage 4)
│   └── harmless_prompts_vi.json     # 200 harmless prompts (Stage 4)
├── finetuning/
│   ├── __init__.py                  # Exports: run_pretrain, run_sft, run_dpo
│   └── train.py                     # Unified entry: --mode pretrain|sft|dpo
├── scripts/
│   ├── plot_loss.py                 # Loss/LR figure from trainer_state.json
│   └── tokenizer_data_scaling.py    # Tokenizer data-scaling experiment
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

## Dữ liệu train / eval theo stage

| Stage | Train | Eval |
|---|---|---|
| 0. Tokenizer | CulturaX `vi`, 1M docs đầu (`vi_part_00000` + ~360k docs `vi_part_00001`) | `scripts/tokenizer_data_scaling.py`: token/từ trên CulturaX `00010`, `00089`, VTSNLP |
| 1. Pretrain | CulturaX `vi_part_00000`–`00009` | Trong lúc train: 1,000 docs VTSNLP + 1,000 docs CulturaX `vi_part_00010`; notebook: 300 docs mỗi tập |
| 2. SFT | `bkai-foundation-models/vi-alpaca`, 98% sau lọc (48,384) | 2% held-out (988, `heldout.jsonl`); notebook: 30 prompt viết tay |
| 3. DPO | `1997AOF/PKU-SafeRLHF-VI` (hiện không truy cập được) | `data/harmful_prompts_vi.json`, tỷ lệ từ chối theo từ khoá |
| 4. Abliteration | `data/harmful_prompts_vi.json` + `harmless_prompts_vi.json` (200 mỗi file) để tính refusal direction | Cùng hai file đó (chưa tách; kết quả sẽ lạc quan) |

Tokenizer và pretrain không dùng VTSNLP hay `vi_part_00010`, nên eval Stage 1 là dữ liệu chưa thấy.

## Configuration Files

All stage parameters live in `configs/*.yaml`:

| File | Key Parameters |
|------|---------------|
| `pretrain_config.yaml` | `base_model: HuggingFaceTB/SmolLM2-135M`, `init_from_base: true`, shard 0–9, packing 2048, lr=5e-4, cosine, 1 epoch, eval mỗi 2,000 step |
| `sft_config.yaml` | `base_model: ./vi-smollm-135m-pretrain`, prompt/completion, lr=1e-4, cosine, 2% held-out, ChatML template |
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
