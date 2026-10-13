"""Build the Vietnamese DPO safety dataset for Stage 3.

No usable Vietnamese safety-preference set exists on the Hub (the original
``1997AOF/PKU-SafeRLHF-VI`` is gone), so this script builds one:

1. **Safety pairs** — ``PKU-Alignment/PKU-SafeRLHF`` rows where exactly one of the
   two responses is safe and the safe one was also rated better: chosen = the
   safe response, rejected = the unsafe one. One pair per prompt. Translated EN→VI with ``vinai/vinai-translate-en2vi-v2``
   (a plain MT model: unlike chat LLMs it does not refuse to translate harmful text).
2. **Benign pairs** — ``lamhieu/beyond_dpo_vi`` (already Vietnamese, MIT) rows with
   no refusal on either side, so DPO does not learn to refuse everything.
3. **Refusal eval prompts** — harmful prompts from the PKU *test* split and benign
   prompts from the beyond_dpo_vi *test* split (never used for training).

Output (``--out``, default ``data/dpo_safety_vi``):
    train.jsonl / test.jsonl   {"prompt", "chosen", "rejected", "source", "category", "chosen_refusal"}
    refusal_eval.json          {"harmful": [...], "benign": [...]}
    translation_cache.jsonl    EN→VI cache; re-running resumes from it

License: PKU-SafeRLHF is CC BY-NC 4.0, so the safety part is non-commercial.

Usage (sentencepiece is needed only by the MT tokenizer):
    CUDA_VISIBLE_DEVICES=1 uv run --with sentencepiece python scripts/build_dpo_safety_vi.py
    ... --limit 200   # quick smoke run
"""

import argparse
import hashlib
import json
import os
import random
import re
import sys

import torch
from datasets import load_dataset
from tqdm import tqdm

MT_MODEL = "vinai/vinai-translate-en2vi-v2"
REFUSAL_RE = re.compile(
    r"(tôi không thể|tôi xin lỗi|xin lỗi, (nhưng )?tôi|không thể (giúp|hỗ trợ|cung cấp)|"
    r"bất hợp pháp|vi phạm pháp luật|là một mô hình ngôn ngữ|là một trợ lý ai)",
    re.I,
)
# Refusal / discouraging language in the (English) safe response
EN_REFUSAL_RE = re.compile(
    r"(I'm sorry|I am sorry|I cannot|I can't|I won't|I will not|cannot (help|assist|provide)|illegal|unethical|"
    r"against the law|should not|shouldn't|strongly (advise|discourage)|do not recommend|"
    r"not (be )?(appropriate|advisable|ethical|legal|acceptable|recommended))",
    re.I,
)
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


# ----------------------------------------------------------------------------
# Source selection
# ----------------------------------------------------------------------------


def harm_category(row: dict, resp_id: int) -> str:
    cats = row[f"response_{resp_id}_harm_category"]
    hit = [k for k, v in cats.items() if v]
    return hit[0] if hit else "Other"


def select_pku(split: str) -> list[dict]:
    """One (prompt, safe, unsafe) pair per prompt.

    Keeps rows where exactly one response is safe AND annotators also rated that
    response as the better one (drops "safe" but still nasty answers).
    ``chosen_refusal`` marks safe responses that refuse or discourage the request.
    """
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split)
    seen, out = set(), []
    for row in ds:
        s0, s1 = row["is_response_0_safe"], row["is_response_1_safe"]
        if s0 == s1 or row["prompt"] in seen:
            continue
        safe, unsafe = (0, 1) if s0 else (1, 0)
        if row["better_response_id"] != safe:
            continue
        seen.add(row["prompt"])
        out.append({
            "prompt": row["prompt"].strip(),
            "chosen": row[f"response_{safe}"].strip(),
            "rejected": row[f"response_{unsafe}"].strip(),
            "source": "pku-saferlhf",
            "category": harm_category(row, unsafe),
            "chosen_refusal": bool(EN_REFUSAL_RE.search(row[f"response_{safe}"])),
        })
    return out


def _beyond_prompt(text: str) -> str:
    m = re.search(r"<\|user\|>\n(.*?)</s>", text, re.S)
    return m.group(1).strip() if m else ""


def _strip_eos(text: str) -> str:
    return text.replace("</s>", "").strip()


def select_beyond(split: str) -> list[dict]:
    """Single-turn helpful pairs with no refusal on either side."""
    ds = load_dataset("lamhieu/beyond_dpo_vi", split=split)
    out = []
    for row in ds:
        prompt = _beyond_prompt(row["prompt"])
        chosen, rejected = _strip_eos(row["chosen"]), _strip_eos(row["rejected"])
        if not prompt or len(chosen) < 20 or chosen == rejected:
            continue
        if REFUSAL_RE.search(chosen) or REFUSAL_RE.search(rejected) or REFUSAL_RE.search(prompt):
            continue
        out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected,
                    "source": "beyond_dpo_vi", "category": "benign", "chosen_refusal": False})
    return out


# ----------------------------------------------------------------------------
# Translation
# ----------------------------------------------------------------------------


def _key(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def split_chunks(text: str, tok, max_tokens: int) -> list[list[str]]:
    """Text -> lines -> chunks of whole sentences up to ``max_tokens`` (mBART degrades on long input)."""
    lines = []
    for line in text.split("\n"):
        chunks, cur = [], ""
        for sent in SENT_SPLIT.split(line.strip()):
            cand = f"{cur} {sent}".strip()
            if cur and len(tok(cand, add_special_tokens=False)["input_ids"]) > max_tokens:
                chunks.append(cur)
                cur = sent
            else:
                cur = cand
        if cur:
            chunks.append(cur)
        lines.append(chunks)
    return lines


@torch.no_grad()
def translate_all(texts: list[str], cache_path: str, batch_size: int, max_chunk_tokens: int,
                  num_beams: int) -> dict[str, str]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    cache = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                r = json.loads(line)
                cache[r["k"]] = r["vi"]
    todo = list({_key(t): t for t in texts if _key(t) not in cache}.values())
    print(f"[translate] {len(texts) - len(todo):,} cached, {len(todo):,} to translate")
    if not todo:
        return cache

    tok = AutoTokenizer.from_pretrained(MT_MODEL, src_lang="en_XX")
    model = AutoModelForSeq2SeqLM.from_pretrained(MT_MODEL, dtype=torch.bfloat16).cuda().eval()
    vi_id = tok.convert_tokens_to_ids("vi_VN")

    # Flatten every text into chunks, translate the chunks sorted by length, reassemble.
    layout = {t: split_chunks(t, tok, max_chunk_tokens) for t in tqdm(todo, desc="split")}
    chunks = sorted({c for lines in layout.values() for line in lines for c in line}, key=len)
    done = {}
    for i in tqdm(range(0, len(chunks), batch_size), desc="translate"):
        batch = chunks[i:i + batch_size]
        x = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to("cuda")
        out = model.generate(**x, decoder_start_token_id=vi_id, num_beams=num_beams,
                             max_new_tokens=int(x["input_ids"].shape[1] * 2) + 16)
        done.update(zip(batch, tok.batch_decode(out, skip_special_tokens=True)))

    with open(cache_path, "a") as f:
        for t, lines in layout.items():
            vi = "\n".join(" ".join(done[c] for c in line) for line in lines)
            cache[_key(t)] = vi
            f.write(json.dumps({"k": _key(t), "vi": vi}, ensure_ascii=False) + "\n")
    return cache


def is_degenerate(src: str, vi: str) -> bool:
    """Drop translations that collapsed into loops or lost/added most of the text."""
    words = vi.split()
    if not words:
        return True
    grams = [tuple(words[i:i + 3]) for i in range(len(words) - 2)]
    if grams and 1 - len(set(grams)) / len(grams) > 0.3:
        return True
    ratio = len(vi) / max(1, len(src))
    return not 0.6 <= ratio <= 2.5


def translate_pairs(pairs: list[dict], cache: dict[str, str]) -> list[dict]:
    out, dropped = [], 0
    for p in pairs:
        vi = {k: cache[_key(p[k])] for k in ("prompt", "chosen", "rejected")}
        if any(is_degenerate(p[k], vi[k]) for k in vi):
            dropped += 1
            continue
        out.append({**p, **vi})
    print(f"[translate] dropped {dropped} degenerate pairs, kept {len(out):,}")
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def write_jsonl(rows: list[dict], path: str):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/dpo_safety_vi")
    ap.add_argument("--benign-ratio", type=float, default=0.25, help="share of benign pairs in each split")
    ap.add_argument("--n-eval-prompts", type=int, default=200, help="harmful/benign prompts for refusal eval")
    ap.add_argument("--limit", type=int, default=None, help="cap PKU pairs per split (smoke runs)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-chunk-tokens", type=int, default=160)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("A GPU is required for translation")
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)

    pku = {s: select_pku(s) for s in ("train", "test")}
    beyond = {s: select_beyond(s) for s in ("train", "test")}
    for s in pku:
        rng.shuffle(pku[s])
        rng.shuffle(beyond[s])
        if args.limit:
            pku[s] = pku[s][:args.limit]
        print(f"[select] {s}: {len(pku[s]):,} PKU safety pairs, {len(beyond[s]):,} benign candidates")

    # Translate everything needed: all PKU pairs + the eval prompts (subset of PKU test prompts).
    texts = [p[k] for s in pku for p in pku[s] for k in ("prompt", "chosen", "rejected")]
    cache = translate_all(texts, os.path.join(args.out, "translation_cache.jsonl"),
                          args.batch_size, args.max_chunk_tokens, args.num_beams)

    for s in ("train", "test"):
        safety = translate_pairs(pku[s], cache)
        n_benign = round(len(safety) * args.benign_ratio / (1 - args.benign_ratio))
        rows = safety + beyond[s][:n_benign]
        rng.shuffle(rows)
        write_jsonl(rows, os.path.join(args.out, f"{s}.jsonl"))
        n_ref = sum(p["chosen_refusal"] for p in safety)
        print(f"[write] {s}.jsonl: {len(safety):,} safety ({n_ref:,} refusal-style chosen) "
              f"+ {min(n_benign, len(beyond[s])):,} benign")
        if s == "test":
            # Eval prompts come from the test split only; the benign ones are not in test.jsonl.
            harmful = [p["prompt"] for p in safety][:args.n_eval_prompts]
            benign = [p["prompt"] for p in beyond[s][n_benign:]][:args.n_eval_prompts]
            with open(os.path.join(args.out, "refusal_eval.json"), "w") as f:
                json.dump({"harmful": harmful, "benign": benign}, f, ensure_ascii=False, indent=1)
            print(f"[write] refusal_eval.json: {len(harmful)} harmful / {len(benign)} benign prompts")


if __name__ == "__main__":
    main()
