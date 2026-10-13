"""Abliteration Module: Remove Censorship from Vi-SmolLM2-135M (Stage 4).

Weight orthogonalization strips a "refusal direction" from the model; no training is needed.

1. Prompts are wrapped in the ChatML template (the format Stage 2/3 trained on) and the residual
   stream is read at the last prompt token (the ``assistant`` header), where the model decides
   whether to refuse.
2. For every candidate layer, direction = normalize(mean(harmful) - mean(harmless)).
3. Each candidate is applied to a copy of the model (weights orthogonalized) and scored on
   validation prompts; the layer with the fewest refusals (without making harmless prompts refuse) wins.
4. The winning direction is projected out of every matrix that writes to the residual stream:
   ``embed_tokens``, each ``self_attn.o_proj`` and ``mlp.down_proj``.
5. The report compares censored vs. uncensored on held-out prompt sets (never used for the direction).

Direction prompts: ``data/harmful_prompts_vi.json`` / ``harmless_prompts_vi.json`` plus short and
command-style prompts from the DPO training set. Evaluation prompts: ``data/dpo_safety_vi/refusal_eval.json``.

Reference: Arditi et al., "Refusal in Language Models Is Mediated by a Single Direction" (2024).
"""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.eval_utils import is_refusal, is_strict_refusal

SYSTEM_PROMPT = "You are a helpful Vietnamese assistant."  # same as SFT / DPO
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_censored_model(model_path: str = "./vi-smollm-135m-censored"):
    """Load the censored (DPO-aligned) model in fp32 and its tokenizer (left padding, for batching)."""
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32).to(DEVICE).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<pad>"
    tokenizer.padding_side = "left"
    return model, tokenizer


def chat_texts(tokenizer, prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True,
        )
        for p in prompts
    ]


@torch.no_grad()
def extract_activations(model, tokenizer, prompts: list[str], batch_size: int = 32) -> torch.Tensor:
    """Residual stream after every layer at the last prompt token.

    Returns:
        Tensor [num_layers, num_prompts, hidden_dim] (fp32, CPU).
    """
    per_batch = []
    texts = chat_texts(tokenizer, prompts)
    for i in range(0, len(texts), batch_size):
        x = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        x.pop("token_type_ids", None)
        hs = model(**x, output_hidden_states=True).hidden_states[1:]  # drop the embedding output
        per_batch.append(torch.stack([h[:, -1, :] for h in hs]).float().cpu())  # left padding: -1 is real
    return torch.cat(per_batch, dim=1)


def orthogonalize_weight(matrix: torch.Tensor, direction: torch.Tensor, output_side: bool = True) -> torch.Tensor:
    """Remove the component along ``direction`` (a unit vector in the residual stream) from a weight.

    ``output_side=True`` is for matrices that write into the residual stream (``o_proj``,
    ``down_proj``; shape [hidden, in]): ``W - v v^T W``.
    ``output_side=False`` is for matrices that read it or embed into it (``embed_tokens``;
    shape [out, hidden]): ``W - W v v^T``.
    """
    v = (direction / direction.norm()).to(matrix.dtype)
    if output_side:
        return matrix - torch.outer(v, v @ matrix)
    return matrix - torch.outer(matrix @ v, v)


@torch.no_grad()
def apply_ablation(model, direction: torch.Tensor):
    """Orthogonalize every residual-stream writer of ``model`` in place."""
    inner = model.model
    inner.embed_tokens.weight.data = orthogonalize_weight(inner.embed_tokens.weight.data, direction, output_side=False)
    for layer in inner.layers:
        layer.self_attn.o_proj.weight.data = orthogonalize_weight(layer.self_attn.o_proj.weight.data, direction)
        layer.mlp.down_proj.weight.data = orthogonalize_weight(layer.mlp.down_proj.weight.data, direction)
    return model


@torch.no_grad()
def generate(model, tokenizer, prompts: list[str], max_new_tokens: int = 64, batch_size: int = 32) -> list[str]:
    """Greedy ChatML generation; stops at ``<|im_end|>``."""
    stop = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")]
    texts, outs = chat_texts(tokenizer, prompts), []
    for i in range(0, len(texts), batch_size):
        x = tokenizer(texts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(DEVICE)
        x.pop("token_type_ids", None)
        g = model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False, repetition_penalty=1.15,
                           eos_token_id=stop, pad_token_id=tokenizer.pad_token_id)
        outs += [tokenizer.decode(r, skip_special_tokens=True).strip() for r in g[:, x["input_ids"].shape[1]:]]
    return outs


def refusal_rates(answers: list[str]) -> dict[str, float]:
    n = max(1, len(answers))
    return {"strict": sum(map(is_strict_refusal, answers)) / n, "broad": sum(map(is_refusal, answers)) / n}


@torch.no_grad()
def heldout_loss(model, tokenizer, path: str, n: int = 200) -> float | None:
    """Mean NLL over the assistant tokens of the SFT held-out conversations (None if the file is missing)."""
    if not os.path.isfile(path):
        return None
    rows = [json.loads(line) for line in open(path)][:n]
    total, count = 0.0, 0
    for r in rows:
        prompt = tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(r["prompt"] + r["completion"], tokenize=False)
        ids = tokenizer(full, return_tensors="pt", add_special_tokens=False, truncation=True, max_length=1024)["input_ids"].to(DEVICE)
        n_prompt = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        if ids.shape[1] <= n_prompt + 1:
            continue
        labels = ids.clone()
        labels[:, :n_prompt] = -100
        loss = model(input_ids=ids, labels=labels).loss
        k = ids.shape[1] - n_prompt
        total, count = total + loss.item() * k, count + k
    return total / count


# ----------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------


def load_direction_prompts(harmful_path: str, harmless_path: str, dpo_dir: str, n_extra: int,
                           rng: random.Random) -> tuple[list[str], list[str]]:
    """Stage 4 prompt files plus short / command-style prompts from the DPO training set."""
    harmful, harmless = json.load(open(harmful_path)), json.load(open(harmless_path))
    train = os.path.join(dpo_dir, "train.jsonl")
    if n_extra and os.path.isfile(train):
        rows = [json.loads(line) for line in open(train)]
        extra_h = sorted({r["prompt"] for r in rows if r["source"] in ("refusal-short", "refusal-command")})
        extra_b = sorted({r["prompt"] for r in rows if r["source"] in ("benign-short", "benign-command")})
        rng.shuffle(extra_h)
        rng.shuffle(extra_b)
        harmful, harmless = harmful + extra_h[:n_extra], harmless + extra_b[:n_extra]
    return list(dict.fromkeys(harmful)), list(dict.fromkeys(harmless))


def split_val(items: list[str], frac: float, rng: random.Random) -> tuple[list[str], list[str]]:
    items = items[:]
    rng.shuffle(items)
    k = max(1, int(len(items) * frac))
    return items[k:], items[:k]


# ----------------------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------------------


def select_layer(model, tokenizer, directions: torch.Tensor, val_harmful: list[str], val_harmless: list[str],
                 max_new_tokens: int, sft_heldout: str, max_loss_increase: float = 0.1,
                 n_loss_rows: int = 64) -> tuple[int, list[dict]]:
    """Try the direction of every candidate layer; return the best layer and all scores.

    A layer is eligible only if the ablated model's held-out loss rises by at most
    ``max_loss_increase`` (otherwise "no refusals" just means gibberish). Among eligible layers the
    fewest strict refusals win, with a penalty if harmless prompts start to be refused.
    """
    n_layers = directions.shape[0]
    base_harmless = refusal_rates(generate(model, tokenizer, val_harmless, max_new_tokens))["strict"]
    base_loss = heldout_loss(model, tokenizer, sft_heldout, n_loss_rows)
    scores = []
    for layer in range(n_layers // 5, n_layers * 4 // 5):
        trial = apply_ablation(copy.deepcopy(model), directions[layer].to(DEVICE))
        h = refusal_rates(generate(trial, tokenizer, val_harmful, max_new_tokens))["strict"]
        b = refusal_rates(generate(trial, tokenizer, val_harmless, max_new_tokens))["strict"]
        loss = heldout_loss(trial, tokenizer, sft_heldout, n_loss_rows)
        scores.append({"layer": layer, "harmful_refusal": h, "harmless_refusal": b, "heldout_loss": loss,
                       "score": h + max(0.0, b - base_harmless)})
        print(f"  layer {layer:2d}: harmful refusal {h:.0%}, harmless refusal {b:.0%}, "
              f"loss {loss if loss is None else round(loss, 3)} (base {base_loss if base_loss is None else round(base_loss, 3)})")
        del trial
    eligible = [x for x in scores if base_loss is None or x["heldout_loss"] <= base_loss + max_loss_increase]
    if not eligible:
        print(f"[warn] no layer keeps the loss within +{max_loss_increase}: picking the lowest-loss layer")
        best = min(scores, key=lambda x: x["heldout_loss"])
    else:
        best = min(eligible, key=lambda x: (x["score"], x["heldout_loss"] if base_loss is not None else 0))
    return best["layer"], scores


def run_abliteration(
    censored_model_path: str = "./vi-smollm-135m-censored",
    output_path: str = "./vi-smollm-135m-uncensored",
    harmful_prompts_path: str = "./data/harmful_prompts_vi.json",
    harmless_prompts_path: str = "./data/harmless_prompts_vi.json",
    dpo_dir: str = "./data/dpo_safety_vi",
    sft_heldout: str = "./vi-smollm-135m-sft/heldout.jsonl",
    n_extra: int = 300,
    n_eval: int = 100,
    max_new_tokens: int = 64,
    seed: int = 42,
):
    """Execute the full abliteration pipeline (Stage 4) and write a before/after report."""
    rng = random.Random(seed)
    print("[Stage 4] Loading censored model...")
    model, tokenizer = load_censored_model(censored_model_path)

    harmful, harmless = load_direction_prompts(harmful_prompts_path, harmless_prompts_path, dpo_dir, n_extra, rng)
    harmful_fit, harmful_val = split_val(harmful, 0.15, rng)
    harmless_fit, harmless_val = split_val(harmless, 0.15, rng)
    print(f"[Stage 4] Direction prompts: harmful {len(harmful_fit)} (+{len(harmful_val)} val), "
          f"harmless {len(harmless_fit)} (+{len(harmless_val)} val)")

    print("[Stage 4] Extracting activations...")
    mu_h = extract_activations(model, tokenizer, harmful_fit).mean(dim=1)
    mu_b = extract_activations(model, tokenizer, harmless_fit).mean(dim=1)
    directions = mu_h - mu_b
    directions = directions / directions.norm(dim=-1, keepdim=True)

    print("[Stage 4] Selecting the layer (strict refusal + held-out loss after ablation)...")
    layer, scores = select_layer(model, tokenizer, directions, harmful_val, harmless_val, max_new_tokens, sft_heldout)
    print(f"[Stage 4] Best layer: {layer}")

    eval_sets = json.load(open(os.path.join(dpo_dir, "refusal_eval.json")))
    eval_sets = {k: v[:n_eval] for k, v in eval_sets.items()}
    before = {k: generate(model, tokenizer, v, max_new_tokens) for k, v in eval_sets.items()}
    loss_before = heldout_loss(model, tokenizer, sft_heldout)

    apply_ablation(model, directions[layer].to(DEVICE))
    after = {k: generate(model, tokenizer, v, max_new_tokens) for k, v in eval_sets.items()}
    loss_after = heldout_loss(model, tokenizer, sft_heldout)

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"[Stage 4] Uncensored model saved to {output_path}")

    report = {
        "layer": layer,
        "layer_scores": scores,
        "refusal": {k: {"n": len(eval_sets[k]), "censored": refusal_rates(before[k]),
                        "uncensored": refusal_rates(after[k])} for k in eval_sets},
        "heldout_loss": {"censored": loss_before, "uncensored": loss_after},
        "samples": {k: [{"prompt": p, "censored": b, "uncensored": a}
                        for p, b, a in list(zip(eval_sets[k], before[k], after[k]))[:5]] for k in eval_sets},
    }
    with open(os.path.join(output_path, "abliteration_report.json"), "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print("\n[Stage 4] Strict refusal rate on held-out prompts (censored -> uncensored):")
    for k, v in report["refusal"].items():
        print(f"  {k:<16} n={v['n']:<4} {v['censored']['strict']:.0%} -> {v['uncensored']['strict']:.0%}")
    if loss_before is not None:
        print(f"  held-out loss    {loss_before:.4f} -> {loss_after:.4f}")
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="./vi-smollm-135m-censored")
    ap.add_argument("--out", default="./vi-smollm-135m-uncensored")
    ap.add_argument("--n-extra", type=int, default=300, help="extra direction prompts per side from the DPO train set")
    ap.add_argument("--n-eval", type=int, default=100, help="held-out prompts per eval set")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run_abliteration(args.model, args.out, n_extra=args.n_extra, n_eval=args.n_eval,
                     max_new_tokens=args.max_new_tokens, seed=args.seed)


if __name__ == "__main__":
    main()
