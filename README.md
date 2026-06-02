# cayley-attention

Sparse-attention connectivity studies built on a minimal nanoGPT, asking a single
question across phases: **when you sparsify attention to a fixed edge budget, does the
*specific graph construction* matter, or only the *expander property*?**

The recurring control is **structured-deterministic vs random** at matched budget.
BigBird already showed that an expander-sparse pattern matches full attention; the open
question these experiments probe is whether a deterministic algebraic expander (an
SL(2,ℤₙ) Cayley graph) buys anything a random BigBird-style graph does not — in mean
quality, or in *variance across seeds*.

## Setup

- **Model**: minimal nanoGPT (decoder-only, weight-tied, GELU MLP), held identical across
  every arm of a phase. The *only* thing that varies between arms is the attention mask.
- **Attention masks**: a fixed boolean `(T,T)` allow-matrix per arm, intersected with the
  causal mask, applied by setting masked logits to `−∞` before softmax (explicit dense
  masked attention — at these sequence lengths the dense softmax is cheap).
- **Dataset**: char-level tiny Shakespeare (1,115,394 chars, 65-token vocab, 90/10 split).
- **Hardware**: single RTX 5070 Ti.
- **Interpreter**: the experiments import `torch` from the sibling HRS venv
  (`~/Code/HRS/.venv`); the code itself is standalone.

## Experiments index

| Phase | Question | Headline result |
|-------|----------|-----------------|
| [01](results/phase01/cayley_vs_random_README.md) | Cayley-graph vs random sparse attention at equal budget on tiny Shakespeare — is the construction load-bearing, or only the expander property? | **Construction not load-bearing at this scale.** B (Cayley) 1.4744 ± 0.0033 ≈ C (random) 1.4756 ± 0.0023, both within ~0.013 nats of full causal (1.4620); predicted std(B)<std(C) signature absent. Window essential (no-window ablation +0.91 nats); long-range edges add only +0.005. Random graphs are *higher*-gap expanders (0.79 vs 0.59) here. |
| [02](results/phase02/cayley_vs_random_n11_k2_README.md) | Decisive retest at large N, small budget (N=1320, k=2) — the regime where random graphs *should* become unreliable expanders and the variance signature should appear. | **Hypothesis fails again, in its predicted-best regime.** Premise now holds (random gap varies 2.2× across seeds; fixed Cayley is the *worst* expander, gap 0.075) yet std(B)=0.0054 ≥ std(C)=0.0037 and means tie (1.4766 vs 1.4770, both at the full ceiling). Causal chain breaks at link 2: with a window, loss is independent of long-range graph quality. Window-only (E, 1.4735) is the *best* sparse arm — at k=2 the long-range edges slightly *hurt*, reversing phase 01. Pure-Cayley (D) catastrophic (+1.0 nat). |
| [03](results/phase03/expander_sufficiency_README.md) | The real thesis: is a sparse expander scaffold (window + O(1) long-range) *sufficient* to match full attention under real long-range load? WikiText-103, GPT-2-small, T=1024 & 4096. | **First pass inconclusive — validity gate failed.** Sparse matches full at both T (B−A within ±0.08 noise) but so does window-only (C−B ≈ 0 even at 4096): a 256-wide window already suffices on WikiText, so the long-range edges carry no load and the test can't adjudicate the *expander*. Construction null (Cayley=random) confirmed a 3rd time. Undertrained (0.69 epoch). → phase 03b bottlenecks the window (w↓64) and doubles training to actually load the scaffold. |
| 03b | _in progress_ — load the scaffold: T=4096, sweep window w∈{128,64} so local attention is bottlenecked; does B (window+long-range) match full while window-only fails? 20k steps. | _running_ |

## Repo layout

```
experiments/   one script per phase
results/phaseNN/   phaseNN.json (per-arm/per-seed metrics), plots, phase README
data/          cached train.bin/val.bin/meta.pkl (gitignored, regenerated from corpus)
```
