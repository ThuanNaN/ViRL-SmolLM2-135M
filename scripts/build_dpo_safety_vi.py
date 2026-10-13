"""Build the Vietnamese DPO safety dataset for Stage 3 (one script, one dataset).

No usable Vietnamese safety-preference set exists on the Hub (the original
``1997AOF/PKU-SafeRLHF-VI`` is gone), so this script builds one from three sources:

* ``PKU-Alignment/PKU-SafeRLHF`` (EN, CC BY-NC 4.0) — rows where exactly one response is safe
  and rated better. Translated EN→VI with ``vinai/vinai-translate-en2vi-v2`` (a plain MT model:
  unlike chat LLMs it does not refuse to translate harmful text). One pair per prompt.
* ``lamhieu/beyond_dpo_vi`` (VI, MIT) — benign helpful pairs with no refusal on either side.
* ``bkai-foundation-models/vi-alpaca`` — short benign instructions with their reference answers.

Every pair is ``(prompt, chosen, rejected)``. Rows by ``source``:

    pku-long          harmful, long PKU prompt. chosen = PKU safe reply that refuses/discourages,
                      rejected = PKU unsafe reply
    refusal-short     harmful, short PKU prompt (<= --max-chars). chosen = refusal template
    refusal-command   the same prompts rewritten as commands ("Hướng dẫn cách ..."). chosen = refusal template
    benign-long       beyond_dpo_vi pair as is
    benign-short      short beyond_dpo_vi prompt. chosen = helpful reply, rejected = refusal template
    benign-command    beyond_dpo_vi prompts rewritten as commands + short vi-alpaca instructions.
                      chosen = helpful reply, rejected = refusal template

Refusal templates are fixed Vietnamese sentences; the benign rows with a refusal as *rejected* teach
the model not to refuse harmless requests. Prompts in ``data/harmful_prompts_vi.json`` /
``harmless_prompts_vi.json`` (the Stage 4 contrast set) are never used.

Output (``--out``, default ``data/dpo_safety_vi``; everything is overwritten):
    train.jsonl / test.jsonl   {"prompt", "chosen", "rejected", "source", "category"}
    refusal_eval.json          held-out test prompts per set: harmful_{long,short,command},
                               benign_{long,short,command}
    translation_cache.jsonl    EN→VI cache; re-running resumes from it (no GPU needed once complete)

License: PKU-SafeRLHF is CC BY-NC 4.0, so the output is non-commercial and gitignored.

Usage (sentencepiece is needed only by the MT tokenizer):
    CUDA_VISIBLE_DEVICES=0 uv run --with sentencepiece python scripts/build_dpo_safety_vi.py
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

# Fixed refusals; one is picked per prompt (hash), so reruns give identical files.
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


def _hash(text: str) -> int:
    return int(hashlib.sha1(text.encode()).hexdigest(), 16)


def refusal_for(prompt: str) -> str:
    return REFUSALS[_hash(prompt) % len(REFUSALS)]


def to_command(prompt: str) -> str | None:
    """Rewrite a question as a command, or None if it is not one of ``QUESTION_FORMS``."""
    for form in QUESTION_FORMS:
        m = form.match(prompt.strip())
        if m:
            return COMMANDS[_hash(prompt) % len(COMMANDS)].format(g=m.group(1).rstrip(" ."))
    return None


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
        out.append({"prompt": prompt, "chosen": chosen, "rejected": rejected, "category": "benign"})
    return out


def select_alpaca(exclude: set[str], max_prompt: int = 100, max_answer: int = 600) -> list[dict]:
    """Short command-style vi-alpaca instructions (no extra input) with their reference answers."""
    ds = load_dataset("bkai-foundation-models/vi-alpaca", split="train")
    out = []
    for r in ds:
        q, a = r["instruction"].strip(), r["output"].strip()
        if (r["input"] or "").strip() or q in exclude:
            continue
        if 10 <= len(q) <= max_prompt and 20 <= len(a) <= max_answer:
            out.append({"prompt": q, "answer": a})
    return out


def sft_heldout_prompts(path: str) -> set[str]:
    """User prompts of the Stage 2 held-out set; they are kept out of the DPO training data."""
    if not os.path.isfile(path):
        print(f"[warn] {path} not found: SFT held-out prompts are not excluded")
        return set()
    with open(path) as f:
        return {json.loads(line)["prompt"][-1]["content"].strip() for line in f}


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
    if not torch.cuda.is_available():
        sys.exit("A GPU is required to translate the uncached texts")

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
# Pair construction
# ----------------------------------------------------------------------------


def row(source: str, prompt: str, chosen: str, rejected: str, category: str) -> dict:
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected, "source": source, "category": category}


def refusal_row(source: str, prompt: str, r: dict) -> dict:
    return row(source, prompt, refusal_for(prompt), r["rejected"], r["category"])


def benign_row(source: str, prompt: str, answer: str) -> dict:
    return row(source, prompt, answer, refusal_for(prompt), "benign")


def build_split(safety: list[dict], benign: list[dict], alpaca: list[dict], args, rng: random.Random,
                cap: bool) -> tuple[list[dict], dict[str, list[str]]]:
    """All rows of one split plus its generation-eval prompts.

    ``cap`` limits the short / alpaca sets (train); the test split keeps everything it has.
    """
    long_ok = [r for r in safety if r["chosen_refusal"]]
    n_benign = round(len(safety) * args.benign_ratio / (1 - args.benign_ratio))
    benign_long = benign[:n_benign]

    def pick(items, n):
        return items[:n] if cap else items

    short_h = pick([r for r in safety if len(r["prompt"]) <= args.max_chars], args.n_short_harmful)
    short_b = pick([r for r in benign_long if len(r["prompt"]) <= args.max_chars], args.n_short_benign)
    cmd_h = [(c, r) for r in safety if len(r["prompt"]) <= args.max_chars and (c := to_command(r["prompt"]))]
    cmd_b = [(c, r) for r in benign_long if len(r["prompt"]) <= args.max_chars and (c := to_command(r["prompt"]))]

    rows = (
        [row("pku-long", r["prompt"], r["chosen"], r["rejected"], r["category"]) for r in long_ok]
        + [refusal_row("refusal-short", r["prompt"], r) for r in short_h]
        + [refusal_row("refusal-command", c, r) for c, r in cmd_h]
        + [row("benign-long", r["prompt"], r["chosen"], r["rejected"], "benign") for r in benign_long]
        + [benign_row("benign-short", r["prompt"], r["chosen"]) for r in short_b]
        + [benign_row("benign-command", c, r["chosen"]) for c, r in cmd_b]
        + [benign_row("benign-command", a["prompt"], a["answer"]) for a in alpaca]
    )
    rng.shuffle(rows)

    # Generation-eval prompts come from the test split (never trained on).
    eval_prompts = {
        "harmful_long": [r["prompt"] for r in safety if len(r["prompt"]) > args.max_chars],
        "harmful_short": [r["prompt"] for r in safety if len(r["prompt"]) <= args.max_chars],
        "harmful_command": [c for c, _ in cmd_h],
        "benign_long": [r["prompt"] for r in benign if len(r["prompt"]) > args.max_chars],
        "benign_short": [r["prompt"] for r in benign if len(r["prompt"]) <= args.max_chars],
        "benign_command": [a["prompt"] for a in alpaca],
    }
    return rows, eval_prompts


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
    ap.add_argument("--sft-heldout", default="vi-smollm-135m-sft/heldout.jsonl")
    ap.add_argument("--benign-ratio", type=float, default=0.25, help="share of benign-long pairs vs. all safety pairs")
    ap.add_argument("--max-chars", type=int, default=120, help="'short' prompt length cap")
    ap.add_argument("--n-short-harmful", type=int, default=2000)
    ap.add_argument("--n-short-benign", type=int, default=1500)
    ap.add_argument("--n-alpaca", type=int, default=1100, help="benign-command vi-alpaca pairs (train)")
    ap.add_argument("--n-eval-prompts", type=int, default=150, help="generation-eval prompts per set")
    ap.add_argument("--limit", type=int, default=None, help="cap PKU pairs per split (smoke runs)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-chunk-tokens", type=int, default=160)
    ap.add_argument("--num-beams", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

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

    texts = [p[k] for s in pku for p in pku[s] for k in ("prompt", "chosen", "rejected")]
    cache = translate_all(texts, os.path.join(args.out, "translation_cache.jsonl"),
                          args.batch_size, args.max_chunk_tokens, args.num_beams)

    # Stage 2 held-out prompts stay out of the training data and become the benign-command eval prompts.
    heldout = sft_heldout_prompts(args.sft_heldout)
    alpaca = select_alpaca(heldout)
    rng.shuffle(alpaca)
    ho = sorted(p for p in heldout if len(p) <= 100 and "\n" not in p)
    rng.shuffle(ho)
    alpaca_splits = {"train": alpaca[:args.n_alpaca],
                     "test": [{"prompt": p, "answer": ""} for p in ho[:args.n_eval_prompts]]}

    eval_prompts = {}
    for s in ("train", "test"):
        safety = translate_pairs(pku[s], cache)
        rows, ev = build_split(safety, beyond[s], alpaca_splits[s], args, rng, cap=s == "train")
        if s == "test":
            # Held-out benign-command prompts have no reference answer: eval-only, not pairs.
            rows = [r for r in rows if r["chosen"]]
            eval_prompts = {k: v[:args.n_eval_prompts] for k, v in ev.items()}
        write_jsonl(rows, os.path.join(args.out, f"{s}.jsonl"))
        by_source = {}
        for r in rows:
            by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        print(f"[write] {s}.jsonl: {len(rows):,} pairs  " + ", ".join(f"{k} {v:,}" for k, v in sorted(by_source.items())))

    with open(os.path.join(args.out, "refusal_eval.json"), "w") as f:
        json.dump(eval_prompts, f, ensure_ascii=False, indent=1)
    print("[write] refusal_eval.json: " + ", ".join(f"{k} {len(v)}" for k, v in eval_prompts.items()))


if __name__ == "__main__":
    main()
