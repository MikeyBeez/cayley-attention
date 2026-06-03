# Phase 04 — Synthetic associative recall: a dependency provably beyond the window

**When the dependency is placed by construction past the window, edge *placement* is load-bearing and
the algebraic *construction* is not. A single correctly-aimed dilated edge gives 100% recall of a
supra-window needle; the Cayley graph fails identically to uniform-random and to window-only — all
three stay at chance — because their edges don't land on the dependency and the 8-layer model does
not learn to compose the hop over depth.** This is the test phase 03b said was missing: a task where
>window information is *provably* required, so the long-range scaffold finally has a job. It does, and
the result is unambiguous in both directions — the first phase where a sparse arm beats the window
floor (B_dilated, chance → 1.00), and the cleanest negative yet for the Cayley construction (B_cayley
= B_rand = floor even with the scaffold fully loaded).

## Task

Multi-key associative recall (MQAR-style), procedurally generated each step (no stored corpus). A
context of distinct `(key,value)` bigrams is scattered at random even positions; one is the **needle**.
The query `QUERY key_q value_q` is appended and recall is scored *only* at the query-key position
(argmax == `value_q`, chance = 1/V = 1.56%). The needle's value sits **exactly D tokens before the
query-key position**, and the query position is randomized per sequence (only D is fixed), so the
answer is identifiable by content alone, not by absolute position. Two distance regimes against a
fixed window w=128:

- **D_in = 64** — needle *inside* the window. Sanity: every arm, including window-only, must recall.
- **D_out = 768** — needle *well beyond* the window. The discriminating regime.

The discriminating axis is **edge placement**, isolated by a direct-reach guard: `B_dilated` has an
edge at offset exactly 768 (reaches the needle in one hop from every query node, 100%); `B_cayley`'s
scaled-SL(2,ℤ₁₇) offsets hit 768 for ~0% of nodes; `B_rand`'s uniform edges hit it for ~0.8%. So
dilated can route the dependency in one hop; rand/cayley can only via multi-hop composition over the
8 layers. This is exactly the "match the sparsity to where the removed dependency lives" design that
phase 03b's mechanism pointed to.

## Headline table (recall accuracy; chance = 0.0156; n=2 seeds)

| Arm | D=64 (inside window) | D=768 (beyond window) | direct one-hop reach @ D768 |
|-----|:--------------------:|:---------------------:|:---------------------------:|
| **A_full** — full causal (ceiling) | 0.997 | **0.988** | 100% |
| **C_window** — window w=128 (floor) | 1.000 | **0.021** ≈ chance | 0% |
| **B_rand** — window + 8 uniform-random long edges | 1.000 | **0.027** ≈ chance | 0.8% |
| **B_dilated** — window + 8 dilated edges {128…1024} | 1.000 | **1.000** | 100% |
| **B_cayley** — window + 8-regular scaled-Cayley edges | 1.000 | **0.031** ≈ chance | 0% |

Model: 25.2M-param decoder (8 layer × 8 head × 512, RoPE, weight-tied), identical across arms — only
the attention mask differs. Edge budgets matched among sparse arms (window 123.8k; +8 long-range
125–129k; full 524.8k). Trained to a recall plateau (< 0.5% over a 2k-step window, min 3k steps);
arms that learned plateaued at 3k steps (~16 min), arms that stayed at chance were run to the 9k-step
fail-cutoff (~48 min) — **failures were trained 3× longer and never left chance**, so this is a
reach failure, not undertraining. ~14 GPU-hours, 20 runs.

## What the result shows

1. **Sanity passes — at D=64 every arm recalls perfectly (≥ 0.997).** The needle is inside the
   128-token window, so even window-only attention reads it. Content-based recall, RoPE, and training
   all work; nothing is broken upstream. (Absolute learned position embeddings provably fail this task
   — full attention stays at chance — which is why the model uses RoPE.)

2. **At D=768 the window floor collapses to chance (C_window 0.021).** This is the load the prior
   phases never had: with the dependency provably beyond the window, a purely local model *cannot*
   solve the task. The test now adjudicates the long-range scaffold, unlike phase 03b where a 128-wide
   window already matched full attention and the long-range edges had nothing to carry.

3. **A correctly-placed edge solves it perfectly (B_dilated 1.000), in one hop.** The dilated arm has
   an edge at offset exactly 768; every query node reads its needle directly and recall is 100%,
   plateauing at the 3k-step minimum (ce 0.000). This is the **first phase where a sparse arm beats
   the window-only floor** — a long-range edge finally earns its place.

4. **The Cayley construction fails — identically to random and to the bare window (B_cayley 0.031 ≈
   B_rand 0.027 ≈ C_window 0.021, all at chance).** Even with the scaffold fully loaded, the algebraic
   structure buys nothing: its scattered offsets don't land on the dependency (0% one-hop reach), and
   8 layers of attention **do not learn the multi-hop composition** that could in principle bridge
   768 = 6×128 over the window or via combinations of long edges. The discriminator is whether an edge
   *directly bridges the dependency distance*, not how the graph is built.

## Mechanism

The separation is entirely explained by the direct-reach guard, computed before any training:

| Arm | one-hop reach @ D768 | recall @ D768 |
|-----|:--------------------:|:-------------:|
| B_dilated | 100% | 1.000 |
| B_rand | 0.8% | 0.027 |
| B_cayley | 0% | 0.031 |
| C_window | 0% | 0.021 |

Recall tracks one-hop reach as a step function: ~100% reach → perfect, ~0% reach → chance, with
nothing in between. The empirical content is the **absence of multi-hop routing**: rand and cayley
*could* in principle compose the 768-token hop across 8 layers (window-to-window relay, or chaining
two long edges), and they do not — depth-8 attention on this task relies on a single direct edge or
fails. So "the long-range graph is not load-bearing" sharpens to: **the graph helps exactly when it
places an edge across the dependency, and the algebraic expander structure does not substitute for
that placement via composition.**

## Relation to prior phases

Phases 01–03b were all *construction nulls* under the suspicion (confirmed in 03b) that the task
didn't load the scaffold. Phase 04 removes that confound by construction and reaches a two-sided
conclusion:

- **Placement is real** — when an edge bridges a provably-supra-window dependency, it flips chance to
  100%. The long-range scaffold is not inert *in principle*; 03b's inertness was the task, as
  suspected.
- **Construction is still null** — the named Cayley graph, the whole object of this repo, provides no
  advantage over random or over the bare window even here. It fails for the same reason random does:
  its edges aren't aimed at the dependency, and structure-via-composition doesn't happen.

The through-line across all phases holds and is now mechanistically grounded: **the load-bearing
property of a sparse attention graph is whether its edges reach the dependency, not the algebraic
construction that generates them.** A dumb dilated stride beats SL(2,ℤₙ) by a mile when the dilation
matches the dependency distance.

## Pre-committed predictions vs measured

| Prediction (04) | Measured | Verdict |
|-----------------|----------|:-------:|
| All arms recall at D_in=64 (sanity) | 0.997–1.000 | ✅ |
| C_window collapses to chance at D_out=768 | 0.021 | ✅ |
| B_dilated (edge at offset 768) recovers full recall | 1.000 | ✅ |
| B_rand / B_cayley compose the hop over 8 layers and recover | both ≈ chance | ❌ (no multi-hop) |
| Cayley structure beats random when scaffold is loaded | tie, both at floor | ❌ (construction null, 4th phase) |

## Architectural interpretation

> **Edge placement is the load-bearing property of a sparse attention graph; the algebraic
> construction is not — and depth does not buy the composition that would make placement unnecessary.**
> On a task engineered so the answer lives provably beyond the local window, a single dilated edge
> whose offset matches the dependency distance recovers full recall (chance → 1.00 in one hop), while
> the bare window, uniform-random long edges, and the SL(2,ℤₙ) Cayley graph all sit at chance — the
> Cayley construction failing *identically* to random even though the scaffold is now fully loaded.
> The discriminator is geometric reach, measurable before training: arms with ~100% one-hop reach of
> the needle solve it, arms with ~0% do not, and the eight-layer model does not learn to compose the
> supra-window hop out of shorter edges. This closes the question the repo opened: a deterministic
> algebraic expander buys nothing a random graph does not, in mean *or* in this clean high-signal
> regime — what matters is matching the sparsity pattern to where the dependency actually lives.

## Caveats

- **Synthetic task.** The conclusion is sharp *for routing a known-distance dependency*; it does not
  speak to natural-language LM (phases 03/03b's domain), where dependency distances are mixed and a
  window already suffices. The two halves of the project bracket the same null from opposite tasks.
- **No-composition is for this depth/width/task.** "8 layers don't learn the multi-hop hop" is an
  empirical statement at 8×512 on MQAR with these offsets; a different depth, curriculum, or edge set
  might compose. The claim is that it *did not happen by default*, not that it provably cannot.
- **D_out=768 is a single distance.** A reach-vs-distance sweep (vary D, or vary the dilation offsets
  off the dependency) would trace the step function rather than sampling two points of it; the two
  points here (in-window 1.00, beyond-window-with-matched-edge 1.00, beyond-window-without 0.03) are
  decisive but coarse.
- Quality/recall only; no compute-payoff claim (masked-dense SDPA is slower than flash full attention).

## File manifest

- `phase04_records.json` — per-(arm, D, seed) recall / final-recall / value-CE, steps-to-plateau,
  direct-reach fraction, edge counts, recall curves, peak memory, wall times; aggregated summary.
- `phase04_recall.png` — recall by arm, needle-inside vs needle-beyond window, against the chance line.
- `assoc_recall_README.md` — this file. (Full per-eval recall curves live in the JSON `curve` fields;
  the verbose run log is gitignored.)
- `../../experiments/phase04_assoc_recall.py` — the experiment.

## Open questions → phase 05 (if pursued)

1. **Reach-vs-distance sweep.** Vary D (and/or detune the dilation offset off D) to trace the
   recall(reach) step function and find whether any near-miss placement partially works.
2. **Does composition ever happen?** Curriculum from D_in→D_out, deeper models, or an edge set whose
   *combinations* (not any single edge) span D — to test whether multi-hop routing is learnable at all
   on this task, which is the one mechanism that could revive a structured-expander advantage.
3. **Mixed-distance task.** Multiple needles at different supra-window distances in one sequence, so a
   *single* fixed graph must cover a range — the regime where a broadband expander could finally beat
   a single dilation, if anything can.
