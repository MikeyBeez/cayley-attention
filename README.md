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

## Repo layout

```
experiments/   one script per phase
results/phaseNN/   phaseNN.json (per-arm/per-seed metrics), plots, phase README
data/          cached train.bin/val.bin/meta.pkl (gitignored, regenerated from corpus)
```
