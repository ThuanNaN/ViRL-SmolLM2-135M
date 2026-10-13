"""Add imperative-style refusal pairs to the Stage 3 DPO set (no GPU, no translation, no new harmful text).

Run 3 refused 55-72% of PKU-style questions ("Làm thế nào tôi có thể ...?") but only 27% of the
direct commands in ``data/harmful_prompts_vi.json`` ("Hướng dẫn ...", "Cách ..."). This script
teaches the same behaviour for command phrasing without writing any new harmful prompts:

* **refusal-imperative** — already-translated PKU prompts of the form "Làm thế nào (tôi có thể) ...?"
  / "Cách ... để ...?" are rewritten as commands ("Hướng dẫn cách ...", "Chỉ tôi cách ...").
  chosen = a refusal template, rejected = the PKU unsafe reply.
* **benign-imperative** — the same rewrite on benign beyond_dpo_vi prompts, plus short
  command-style vi-alpaca instructions (not in the SFT held-out set). chosen = the helpful reply,
  rejected = a refusal template.

``data/harmful_prompts_vi.json`` / ``harmless_prompts_vi.json`` are not used.

Output in ``--data-dir`` (default ``data/dpo_safety_vi``):
    imperative_train.jsonl / imperative_test.jsonl   same schema as train.jsonl
    refusal_eval.json                                gains ``harmful_imperative`` / ``benign_imperative``

Usage:
    uv run python scripts/build_dpo_refusal_imperative.py
"""

import argparse
import hashlib
import json
import os
import random
import re

from datasets import load_dataset

from build_dpo_refusal_short import read_jsonl, refusal_for, write_jsonl

# "Làm thế nào tôi có thể X?", "Làm sao để X?", "Cách tốt nhất để X là gì?"  ->  X
QUESTION_FORMS = [
    re.compile(r"^làm (?:thế nào|sao) (?:mà )?(?:tôi|mình) có thể (.+?)\s*\?$", re.I),
    re.compile(r"^làm (?:thế nào|sao) (?:mà )?(?:để|có thể) (.+?)\s*\?$", re.I),
    re.compile(r"^cách (?:nào |tốt nhất |hiệu quả nhất |an toàn nhất )?để (.+?)(?: là gì)?\s*\?$", re.I),
]
COMMANDS = [
    "Hướng dẫn cách {g}.",
    "Chỉ tôi cách {g}.",
    "Cách {g}.",
    "Hãy cho tôi biết cách {g}.",
    "Viết hướng dẫn từng bước để {g}.",
]


def to_command(prompt: str) -> str | None:
    """Rewrite a question prompt as a command, or None if it is not one of the known forms."""
    for form in QUESTION_FORMS:
        m = form.match(prompt.strip())
        if m:
            h = int(hashlib.sha1(prompt.encode()).hexdigest(), 16)
            return COMMANDS[h % len(COMMANDS)].format(g=m.group(1).rstrip(" ."))
    return None


def rewritten(rows: list[dict], max_chars: int) -> tuple[list[dict], list[dict]]:
    """(harmful, benign) rows whose prompt is short and rewritable; prompt replaced by the command form."""
    harmful, benign = [], []
    for r in rows:
        if len(r["prompt"]) > max_chars:
            continue
        cmd = to_command(r["prompt"])
        if cmd:
            (harmful if r["source"] == "pku-saferlhf" else benign).append({**r, "prompt": cmd})
    return harmful, benign


def harmful_pair(r: dict) -> dict:
    return {"prompt": r["prompt"], "chosen": refusal_for(r["prompt"]), "rejected": r["rejected"],
            "source": "refusal-imperative", "category": r["category"], "chosen_refusal": True}


def benign_pair(prompt: str, answer: str) -> dict:
    return {"prompt": prompt, "chosen": answer, "rejected": refusal_for(prompt),
            "source": "benign-imperative", "category": "benign", "chosen_refusal": False}


def heldout_prompts(path: str) -> set[str]:
    return {r["prompt"][-1]["content"].strip() for r in read_jsonl(path)}


def alpaca_pairs(heldout: set[str], n: int, rng: random.Random, max_prompt: int, max_answer: int) -> list[dict]:
    ds = load_dataset("bkai-foundation-models/vi-alpaca", split="train")
    pool = [r for r in ds
            if not (r["input"] or "").strip()
            and 10 <= len(r["instruction"].strip()) <= max_prompt
            and 20 <= len(r["output"].strip()) <= max_answer
            and r["instruction"].strip() not in heldout]
    rng.shuffle(pool)
    return [benign_pair(r["instruction"].strip(), r["output"].strip()) for r in pool[:n]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/dpo_safety_vi")
    ap.add_argument("--sft-heldout", default="vi-smollm-135m-sft/heldout.jsonl")
    ap.add_argument("--max-chars", type=int, default=120)
    ap.add_argument("--n-alpaca", type=int, default=1100)
    ap.add_argument("--n-eval-prompts", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    train = read_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    test = read_jsonl(os.path.join(args.data_dir, "test.jsonl"))
    heldout = heldout_prompts(args.sft_heldout)

    tr_h, tr_b = rewritten(train, args.max_chars)
    train_pairs = [harmful_pair(r) for r in tr_h] + [benign_pair(r["prompt"], r["chosen"]) for r in tr_b]
    train_pairs += alpaca_pairs(heldout, args.n_alpaca, rng, 100, 600)
    rng.shuffle(train_pairs)

    te_h, te_b = rewritten(test, args.max_chars)
    test_pairs = [harmful_pair(r) for r in te_h] + [benign_pair(r["prompt"], r["chosen"]) for r in te_b]
    rng.shuffle(test_pairs)

    # Benign command-style eval prompts: SFT held-out instructions never trained on here.
    ho = sorted(p for p in heldout if len(p) <= 100 and "\n" not in p)
    rng.shuffle(ho)
    eval_prompts = {"harmful_imperative": [r["prompt"] for r in te_h][:args.n_eval_prompts],
                    "benign_imperative": ho[:args.n_eval_prompts]}

    write_jsonl(train_pairs, os.path.join(args.data_dir, "imperative_train.jsonl"))
    write_jsonl(test_pairs, os.path.join(args.data_dir, "imperative_test.jsonl"))
    path = os.path.join(args.data_dir, "refusal_eval.json")
    with open(path) as f:
        ev = json.load(f)
    ev.update(eval_prompts)
    with open(path, "w") as f:
        json.dump(ev, f, ensure_ascii=False, indent=1)

    n_h = sum(p["source"] == "refusal-imperative" for p in train_pairs)
    print(f"[write] imperative_train.jsonl: {n_h:,} refusal-imperative + {len(train_pairs) - n_h:,} benign-imperative "
          f"({len(tr_b):,} rewritten + {args.n_alpaca:,} alpaca)")
    print(f"[write] imperative_test.jsonl: {len(test_pairs):,} pairs")
    print(f"[write] refusal_eval.json += " + " / ".join(f"{k} {len(v)}" for k, v in eval_prompts.items()))
    print("sample rewrites:", *[r["prompt"] for r in tr_h[:3]], sep="\n  ")


if __name__ == "__main__":
    main()
