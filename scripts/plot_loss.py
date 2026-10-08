"""Plot training loss and LR schedule from a HF trainer_state.json (pretrain or SFT).

Usage:
    uv run --with matplotlib python scripts/plot_loss.py \
        [--state vi-smollm-135m-pretrain/checkpoint-80000/trainer_state.json] \
        [--out docs/assets/pretrain_loss.png] [--resume-step 41667]

    SFT: --state vi-smollm-135m-sft/checkpoint-18753/trainer_state.json \
         --out docs/assets/sft_loss.png --name "SFT" --lr-title "LR schedule (linear decay)"
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rolling_mean(xs, k):
    out, s = [], 0.0
    for i, x in enumerate(xs):
        s += x
        if i >= k:
            s -= xs[i - k]
        out.append(s / min(i + 1, k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="vi-smollm-135m-pretrain/checkpoint-80000/trainer_state.json")
    ap.add_argument("--out", default="docs/assets/pretrain_loss.png")
    ap.add_argument("--resume-step", type=int, default=None)
    ap.add_argument("--name", default="Pre-training")
    ap.add_argument("--lr-title", default="LR schedule (warmup 2000, cosine)")
    ap.add_argument("--window", type=int, default=100, help="rolling-mean window in log points")
    args = ap.parse_args()

    hist = [h for h in json.load(open(args.state))["log_history"] if "loss" in h]
    steps = [h["step"] for h in hist]
    loss = [h["loss"] for h in hist]
    lr = [h["learning_rate"] for h in hist]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    a1.plot(steps, loss, color="#9ecae1", lw=0.8, label="loss (every 10 steps)")
    a1.plot(steps, rolling_mean(loss, args.window), color="#08519c", lw=1.8,
            label=f"rolling mean ({args.window} logs)")
    a1.set_yscale("log")
    lo, hi = min(loss), max(loss)
    ticks = [t for t in (2, 3, 4, 6, 10) if lo * 0.9 <= t <= hi * 1.1]
    a1.set_yticks(ticks)
    a1.set_yticklabels([str(t) for t in ticks])
    a1.minorticks_off()
    a1.set_xlabel("step")
    a1.set_ylabel("train loss (log scale)")
    a1.set_title(f"{args.name} loss")
    a1.annotate(f"final {loss[-1]:.2f}", (steps[-1], loss[-1]), textcoords="offset points",
                xytext=(-48, 14), fontsize=9)
    if args.resume_step:
        a1.axvline(args.resume_step, color="#999", ls="--", lw=1)
        a1.text(args.resume_step, a1.get_ylim()[1] * 0.9, " resumed", fontsize=8, color="#666", va="top")
    a1.grid(alpha=0.25)
    a1.legend(frameon=False)

    a2.plot(steps, lr, color="#d95f0e", lw=1.8)
    a2.set_xlabel("step")
    a2.set_ylabel("learning rate")
    a2.set_title(args.lr_title)
    a2.grid(alpha=0.25)

    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}: {len(hist)} points, loss {loss[0]:.2f} -> {loss[-1]:.2f}")


if __name__ == "__main__":
    main()
