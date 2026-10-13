"""Add short-prompt refusal pairs to the Stage 3 DPO set (no GPU, no translation).

Run 2 of DPO learned to refuse long PKU-style prompts (18% strict) but not the short,
direct prompts of ``data/harmful_prompts_vi.json`` (6% strict), and it over-refused
benign prompts with a reflexive "Không, ...". Cause: PKU "chosen" replies discourage
instead of refusing, and almost no training prompt looks like a short direct request.

This script reuses ``data/dpo_safety_vi/{train,test}.jsonl`` (already translated):

* **refusal-short** — short harmful PKU prompts (<= ``--max-chars``). chosen = a fixed
  Vietnamese refusal template, rejected = the PKU unsafe reply.
* **benign-short** — short benign prompts. chosen = the original helpful reply,
  rejected = a refusal template (teaches *not* to refuse harmless requests).

``data/harmful_prompts_vi.json`` / ``harmless_prompts_vi.json`` are never used here, so they
stay a held-out OOD eval set (and Stage 4's contrast set). They are also templated (see the
notebook): count only unique prompts.

Output in ``--data-dir`` (default ``data/dpo_safety_vi``):
    short_train.jsonl / short_test.jsonl   same schema as train.jsonl
    refusal_eval.json                      gains ``harmful_short`` / ``benign_short`` prompt lists

PKU-SafeRLHF is CC BY-NC 4.0, so the output is gitignored like the rest of the directory.

Usage:
    uv run python scripts/build_dpo_refusal_short.py
"""

import argparse
import hashlib
import json
import os
import random

REFUSALS = [
    "Tôi xin lỗi, nhưng tôi không thể giúp với yêu cầu này vì nó có thể gây hại hoặc vi phạm pháp luật.",
    "Tôi không thể hỗ trợ yêu cầu này. Nội dung này có thể gây nguy hiểm cho bạn hoặc cho người khác.",
    "Xin lỗi, tôi không thể cung cấp thông tin này vì nó liên quan đến hành vi gây hại hoặc bất hợp pháp. "
    "Nếu bạn cần giúp đỡ, hãy liên hệ cơ quan chức năng hoặc chuyên gia phù hợp.",
    "Tôi xin lỗi, tôi không thể thực hiện yêu cầu này. Tôi sẵn sàng giúp bạn với những chủ đề an toàn và hợp pháp khác.",
    "Tôi không thể giúp bạn việc này vì nó đi ngược lại các nguyên tắc an toàn và đạo đức. "
    "Bạn có muốn hỏi về một chủ đề khác không?",
    "Rất tiếc, tôi không thể hỗ trợ yêu cầu này. Hành vi được nhắc tới có thể gây tổn hại cho người khác; "
    "tôi khuyên bạn nên tìm cách giải quyết hợp pháp và an toàn.",
]


def refusal_for(prompt: str) -> str:
    """Deterministic template choice per prompt, so reruns give identical files."""
    return REFUSALS[int(hashlib.sha1(prompt.encode()).hexdigest(), 16) % len(REFUSALS)]


def read_jsonl(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(rows: list[dict], path: str):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_pairs(rows: list[dict], max_chars: int, n_harmful: int, n_benign: int, rng: random.Random):
    short = [r for r in rows if len(r["prompt"]) <= max_chars]
    harmful = [r for r in short if r["source"] == "pku-saferlhf"]
    benign = [r for r in short if r["source"] != "pku-saferlhf"]
    rng.shuffle(harmful)
    rng.shuffle(benign)
    pairs = [
        {"prompt": r["prompt"], "chosen": refusal_for(r["prompt"]), "rejected": r["rejected"],
         "source": "refusal-short", "category": r["category"], "chosen_refusal": True}
        for r in harmful[:n_harmful]
    ] + [
        {"prompt": r["prompt"], "chosen": r["chosen"], "rejected": refusal_for(r["prompt"]),
         "source": "benign-short", "category": "benign", "chosen_refusal": False}
        for r in benign[:n_benign]
    ]
    rng.shuffle(pairs)
    return pairs, harmful[n_harmful:], benign[n_benign:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/dpo_safety_vi")
    ap.add_argument("--max-chars", type=int, default=120, help="'short' prompt length cap")
    ap.add_argument("--n-harmful", type=int, default=2000)
    ap.add_argument("--n-benign", type=int, default=1500)
    ap.add_argument("--n-eval-prompts", type=int, default=150, help="held-out short prompts per kind for generation eval")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    train = read_jsonl(os.path.join(args.data_dir, "train.jsonl"))
    test = read_jsonl(os.path.join(args.data_dir, "test.jsonl"))

    train_pairs, _, _ = build_pairs(train, args.max_chars, args.n_harmful, args.n_benign, rng)
    # Test split: the first prompts of each kind go to generation eval, the rest become eval-loss pairs.
    test_short_h = [r for r in test if r["source"] == "pku-saferlhf" and len(r["prompt"]) <= args.max_chars]
    test_short_b = [r for r in test if r["source"] != "pku-saferlhf" and len(r["prompt"]) <= args.max_chars]
    rng.shuffle(test_short_h)
    rng.shuffle(test_short_b)
    k = args.n_eval_prompts
    eval_prompts = {"harmful_short": [r["prompt"] for r in test_short_h[:k]],
                    "benign_short": [r["prompt"] for r in test_short_b[:k]]}
    test_pairs, _, _ = build_pairs(test_short_h[k:] + test_short_b[k:], args.max_chars, 10**9, 10**9, rng)

    write_jsonl(train_pairs, os.path.join(args.data_dir, "short_train.jsonl"))
    write_jsonl(test_pairs, os.path.join(args.data_dir, "short_test.jsonl"))
    path = os.path.join(args.data_dir, "refusal_eval.json")
    with open(path) as f:
        ev = json.load(f)
    ev.update(eval_prompts)
    with open(path, "w") as f:
        json.dump(ev, f, ensure_ascii=False, indent=1)

    n_h = sum(p["source"] == "refusal-short" for p in train_pairs)
    print(f"[write] short_train.jsonl: {n_h:,} refusal-short + {len(train_pairs) - n_h:,} benign-short")
    print(f"[write] short_test.jsonl: {len(test_pairs):,} pairs")
    print(f"[write] refusal_eval.json += harmful_short {len(eval_prompts['harmful_short'])} / "
          f"benign_short {len(eval_prompts['benign_short'])}")


if __name__ == "__main__":
    main()
