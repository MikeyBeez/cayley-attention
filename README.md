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
| [03b](results/phase03b/window_load_README.md) | Load the scaffold: T=4096, sweep window w∈{128,64} so local attention is bottlenecked; does B (window+long-range) match full while window-only fails? 20k steps. | **Long-range edges not load-bearing at any window; window alone matches full down to w=128.** Window-only (C) = full attention at w=128 (ppl 38.06 vs 38.07, just 3% of context); adding 8 long-range edges never helps (idle at w≥128, mildly *hurts* at w=128, fails to recover the deficit at w=64 where sparse drops 0.08 below full). The task needs only ~128 local tokens, so nothing loads the long-range scaffold. Mechanism: uniform-random long edges miss the mid-range band the narrowed window drops → next iteration is dilated/mid-range edges and a genuinely long-range task. |
| [04](results/phase04/assoc_recall_README.md) | Remove 03b's confound by construction: synthetic MQAR recall with the needle placed *provably* D tokens past the window (D=768, w=128). Add a *dilated* arm with an edge at offset exactly D. Does a long-range edge finally earn its place — and does the Cayley structure beat random once the scaffold is loaded? | **Edge *placement* is load-bearing; the *construction* is not.** First phase where a sparse arm beats the window floor: a single dilated edge aimed at the dependency flips chance→**1.00** recall in one hop. But window-only, uniform-random, *and* Cayley all sit at chance (0.02–0.03) at D=768 — the Cayley graph fails **identically to random** even with the scaffold fully loaded, because its offsets don't land on the needle and 8 layers don't learn the multi-hop compose (failures trained 3× longer, never left chance). All arms recall perfectly at D=64 (sanity). Recall tracks pre-training one-hop *reach* as a step function. Construction null confirmed a 4th time, from the opposite (synthetic) task. |
| [05](results/phase05/hop_count_README.md) | Does the model ever *compose*? Remove Phase 04's path confound: a single uniform dilated stride s=D/k makes the needle (D=768) reachable in exactly k hops and no fewer (path-guard certified). Sweep k∈{1,2,3,4}; `scattered_k3` matches the k3 edge count with no clean chain. How deep a clean route will it traverse? | **Composition is real but shallow: ~2 hops reliable, 3 a knife-edge, 4 dead — far below 8 layers.** k1 1.00, k2 **0.996** (both seeds — Phase 04's "no multi-hop" was the missing clean path, not the model), k3 **bimodal** (0.91 / chance across seeds, and only at the 12k-step limit when it works), k4 chance. `scattered_k3` at chance both seeds = same edge count, no clean chain → it's the **clean *path*, not the edge count or construction**, that's load-bearing. Pre-committed "collapse by k=2, depth ~1 hop" falsified *upward*. Construction null untouched (stride is structure-free by design). |
| [06](results/phase06/hop_seedsweep_README.md) | Is Phase 05's 3-hop knife-edge a **budget** artifact (late solvers clipped by the 12k cutoff) or an **init lottery** (more steps won't fix)? Decouple the budget from the control: 40k ceiling, stop-on-solve, CE-flatness gate; estimate success-rate and steps-to-solve per hop over many seeds (k2×2, k3×12, k4×4). | **Lottery, not budget — and 4 hops is a wall, not merely expensive.** k2 2/2 fast; k3 **5/12 (42%)** with winners spread 6k–12k and all 7 non-solvers **dead-flat at the chance floor** (CE slope ≈ 0, false-null gate passed); k4 **0/4**, all flat, no crossing. The pre-committed *budget* prediction (≥7/12, solvers past 12–20k) fails; the *lottery* alternative holds. Phase 05's lone 12k solver was a real late winner — n=2 just undersampled a ~42% bimodal coin. The cliff is a **reliability ladder (reliable→lottery→wall), not a smooth cost curve.** |

## Repo layout

```
experiments/   one script per phase
results/phaseNN/   phaseNN.json (per-arm/per-seed metrics), plots, phase README
data/          cached train.bin/val.bin/meta.pkl (gitignored, regenerated from corpus)
```
