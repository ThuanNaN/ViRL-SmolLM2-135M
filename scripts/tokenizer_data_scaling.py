"""Tokenizer data-scaling experiment.

Trains Byte-Level BPE tokenizers with the same settings as data/tokenizer_train.py on the first
N CulturaX (vi) docs (training shards 0-9, read from the local HF cache) and reports tokens per
word on unseen data (CulturaX shards 10 and 89, VTSNLP). Lower is better. Nothing is saved.

Usage:
    uv run python scripts/tokenizer_data_scaling.py 250000 1000000 4000000
"""
import glob
import os
import sys
import time

import pyarrow.parquet as pq
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from datasets import load_dataset
from huggingface_hub.constants import HF_HUB_CACHE

D = glob.glob(os.path.join(HF_HUB_CACHE, "datasets--uonlp--CulturaX/snapshots/*/vi/"))[0]  # needs shards 0-10, 89 cached
SPECIAL = ["<unk>", "<s>", "</s>", "<pad>", "<mask_token>", "<|im_start|>", "<|im_end|>"]

def docs(n):  # shards in order, like the streaming loader (train shards 0-9 only)
    c = 0
    for i in range(10):
        pf = pq.ParquetFile(f"{D}vi_part_{i:05d}.parquet")
        for b in pf.iter_batches(batch_size=1000, columns=["text"]):
            for x in b.column("text").to_pylist():
                if c >= n: return
                yield x; c += 1

def train(n):
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(docs(n), trainer=trainers.BpeTrainer(vocab_size=32000, special_tokens=SPECIAL, show_progress=False))
    return tok

def shard(i, n=2000):
    return pq.ParquetFile(f"{D}vi_part_{i:05d}.parquet").read_row_group(0, columns=["text"]).column("text").to_pylist()[:n]

evals = {"culturax_00010": shard(10), "culturax_00089": shard(89),
         "vtsnlp": [x["text"] for _, x in zip(range(2000), load_dataset("VTSNLP/vietnamese_curated_dataset", split="train", streaming=True))]}

for n in map(int, sys.argv[1:]):
    t0 = time.time(); tok = train(n); dt = time.time() - t0
    row = []
    for k, texts in evals.items():
        nw = sum(len(x.split()) for x in texts)
        nt = sum(len(e.ids) for e in tok.encode_batch(texts))
        row.append(f"{k} {nt / nw:.4f}")
    print(f"{n:>9,} docs ({dt / 60:.1f} min): " + " | ".join(row), flush=True)
