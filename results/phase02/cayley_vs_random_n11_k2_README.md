# Phase 02 — Cayley vs random sparse attention at large N, small budget (n=11, k=2)

**The variance hypothesis fails a second time — now in the regime designed to confirm it.**
Phase 01 (N=336, k=4) found Cayley and random indistinguishable and attributed the null to
scale: at small N a random graph is already a reliable expander, so its premise ("random
sparsity varies run to run and can realize a poor expander") never engaged. Phase 02 moves
to **N=1320, k=2**, where that premise *does* engage — across 5 seeds the random graph's
spectral gap varies 2.2× (0.073→0.164), and the fixed Cayley graph (gap 0.075) sits at the
**bottom** of that range. The premise holds. The predicted consequence still does not
appear: **std(B)=0.0054 ≥ std(C)=0.0037** (random is, if anything, *more* consistent), and
the means are identical (B 1.4766 vs C 1.4770, Δ=−0.0005). The causal chain
"structured → reliable expander → lower task variance" breaks at its **second** link: with
a local window present, validation loss is nearly independent of long-range graph quality,
so the (real, large) variation in random-graph expansion never becomes loss variation. The
window-only ablation makes this concrete — **E (window only, 1.4735) is the *best* sparse
arm**, beating both B and C; at k=2 the 2 long-range edges are negligible-to-slightly-harmful.
Pure-Cayley with no window (D) is catastrophic (+1.0 nat), exactly as in phase 01.

## Headline table

| Arm | Connectivity | Best val loss (mean ± std, nats) | Perplexity | Gap to full |
|-----|-------------|:-------------------------------:|:----------:|:-----------:|
| **A — Full** | full causal | **1.4800 ± 0.0068** | 4.393 | — (ceiling) |
| **B — Cayley** | window-64 + 2 Cayley | **1.4766 ± 0.0054** | 4.378 | −0.0034 |
| **C — Random** | window-64 + 2 random | **1.4770 ± 0.0037** | 4.380 | −0.0030 |
| **D — Pure Cayley** | 2 Cayley, no window | 2.4773 ± 0.0001 | 11.909 | +0.9972 |
| **E — Window only** | window-64, no long-range | **1.4735 ± 0.0062** | 4.364 | −0.0065 |

Seeds: A/D/E ×3, B/C ×5. Identical 14.19M-param model
(`n_layer=8, n_head=6, n_embd=384, dropout=0.2, block_size=1320`), optimizer, schedule,
`max_iters=2500`, `batch_size=16`; **only the attention mask differs**. ~14 min/run on an
RTX 5070 Ti (T=1320 dense masked attention, 13.5 GB peak), ~4.4 GPU-hours total. The
sparse arms edge *below* the full-attention mean, but all within A's seed band — read as
"statistically at the ceiling," not "beating full."

## The k=2 Cayley construction

`block_size = |SL(2, ℤ₁₁)| = 11³ − 11 = 1320`, one token position per group element, no
padding. The k=2 long-range graph uses **two non-inverse Margulis generators**
a=(1,1,0,1), b=(1,0,1,1) as directed out-edges → 2-out-regular, connected, directed
diameter **13**. (A generator + its inverse would give a 2-*regular undirected* graph — a
disjoint union of cycles — which is the *opposite* of an expander; two independent
generators preserve the directed expander property.) The random control samples 2
positions/token uniformly per seed. Budgets match within 0.6% (B 84,490 vs C 84,957
allowed pairs).

## Per-arm results

**A — Full causal (ceiling).** 1.4800 ± 0.0068. Slightly higher and noisier than phase 01's
full (1.4620 ± 0.0022) because T=1320 with batch 16 and 2500 iters is a harder, noisier
optimization than T=336/batch 64/3500 iters — not a regression, just a different operating
point. This is the reference every sparse arm matches.

**B — Cayley (the proposal).** 1.4766 ± 0.0054 over 5 seeds
(1.4671 / 1.4772 / 1.4787 / 1.4795 / 1.4804). Statistically at the ceiling. The std is
inflated by a single seed that did unusually *well* (1.4671), not by a bad-graph seed —
the Cayley graph is fixed, so B's only seed-to-seed variation is init/data-order.

**C — Random (BigBird control, matched budget).** 1.4770 ± 0.0037 over 5 seeds
(1.4728 / 1.4738 / 1.4782 / 1.4789 / 1.4815). Indistinguishable from B in mean (Δ=−0.0005)
and *tighter* in spread, despite its long-range graph quality varying far more than B's
(see diagnostic). The graph-quality variance simply does not reach the loss.

**D — Pure Cayley, no window (ablation: is the window necessary?).** 2.4773 ± 0.0001 —
**catastrophic (+0.9972 nats), replicating phase 01's verdict.** Two long-range edges with
no locality cannot model local character dependencies; the model collapses to a
bigram-grade predictor (ppl 11.9). Doubly handicapped here: the k=2 directed Cayley
diameter is 13 but the model has only 8 layers, so it is also depth-limited for
composition. The near-zero std (0.0001) shows this is a deterministic floor, not a noisy
failure.

**E — Window only, no long-range (ablation: are the long-range edges necessary?).**
1.4735 ± 0.0062 — **the best sparse arm, below both B and C.** In phase 01 (k=4) the
long-range edges earned a small (+0.005) keep; at k=2/N=1320 they do not — adding 2
long-range edges to the window makes things *slightly worse* on average. At this budget the
64-wide window already captures the available signal, and the sparse long-range edges only
perturb it.

## Connectivity diagnostic (realized causal-masked long-range graph)

| Graph | Spectral gap (λ₁−λ₂) | Components | Giant | Diameter |
|-------|:--------------------:|:----------:|:-----:|:--------:|
| **B — Cayley (fixed, all seeds)** | **0.075** | 100 | 913 / 1320 | 34 |
| C — random seed 1337 | 0.082 | 130 | 1163 | 29 |
| C — random seed 1338 | 0.073 | 170 | 1060 | 30 |
| C — random seed 1339 | 0.107 | 150 | 1108 | 29 |
| C — random seed 1340 | 0.164 | 137 | 1139 | 31 |
| C — random seed 1341 | 0.125 | 126 | 1138 | 32 |

At k=2, causal masking fragments **both** graphs heavily (giant components 913–1163 of
1320; the rest are tiny orphan clusters; long-range diameters 29–34). Two facts matter:
1. **The premise of the hypothesis finally holds.** Random gap ranges 0.073–0.164 (2.2×
   spread) — the random construction *is* an unreliable expander at this budget. Phase 01's
   random gaps were tight (0.776–0.817); here they scatter widely.
2. **Cayley is among the *worst* expanders, not the best.** Its fixed gap (0.075) sits at
   the bottom of C's range, and its giant component (913) is the *smallest* of any graph
   in the table. Causal masking is brutal on the algebraically-placed Cayley edges:
   keeping only j≤i shatters the group structure. Determinism here buys a *consistently
   mediocre* expander, not a good one.

So the hypothesis's premise is satisfied and its mechanism still produces no effect —
because the loss does not depend on the property that varies. See `spectral_gaps.png`.

## Pre-committed predictions vs measured (carried from phase 01)

| Prediction | Measured (n=11, k=2) | Verdict |
|------------|----------------------|:-------:|
| A lowest loss (ceiling) | sparse arms within A's seed band, slightly below | ~ (at ceiling) |
| B and C both close to A | −0.003 nats; both at ceiling | ✅ |
| **std(B) < std(C)** (the signature) | std(B)=0.0054 **≥** std(C)=0.0037 | ❌ (2nd time) |
| D clearly worse than B | +1.0 nat | ✅ |
| E worse than B (long-range earns its place) | E is **best**; long-range slightly *hurts* at k=2 | ❌ (reversed) |
| *Alt:* construction not load-bearing | confirmed at a second, harder scale | ✅ |

### Success criteria (binary)

- **std(B) < std(C) by more than the std-of-std?** **NO.** std(B)=0.0054 (±0.0019),
  std(C)=0.0037 (±0.0013); B is the *larger*. No variance advantage at the scale that was
  supposed to produce one.
- **B within 0.05 nats of A?** **YES** (−0.0034; at the ceiling).
- **D meaningfully worse than B (≥0.05 nats)?** **YES**, by ~1.0 nat.

## Architectural interpretation

> **The Cayley-vs-random question dissolves once a local window is present: the long-range
> connectivity is a second-order term whose quality the language-modeling loss cannot
> see.** Across two scales — (N=336, k=4) and (N=1320, k=2) — a deterministic SL(2,ℤₙ)
> Cayley expander and a seed-varied random graph at matched budget are statistically
> identical in both mean validation loss and seed variance. Phase 02 is the decisive test
> because it satisfies the hypothesis's own premise and *still* yields no effect: at k=2 the
> random graph is genuinely an unreliable expander (spectral gap varies 2.2× across seeds),
> the fixed Cayley graph is in fact one of the *worst* expanders in the comparison (lowest
> gap, smallest giant component, because causal masking shatters the algebraic edge
> placement), yet none of this reaches the loss. The reason is mechanistic and now explicit:
> the window-only ablation (E) is the *best* sparse arm, so at these budgets the 2–4
> long-range edges contribute essentially nothing — and a contribution of zero has zero
> variance regardless of how the edges are chosen. Determinism buys a connectivity
> *guarantee* (Cayley is always one reachable component pre-masking; random fragments), but
> the guarantee is (a) destroyed by causality and (b) irrelevant to a window-equipped model.
> **For the structured-vs-random distinction to matter, the long-range edges must first be
> made to matter — a windowless or genuinely long-range-dependent task — and the windowless
> regime (D) is catastrophic at every scale tested. The construction is not load-bearing for
> windowed sparse attention on this data; the case for an algebraic expander has to be made
> where the expander is the only path between distant tokens, not a marginal add-on to a
> local band.**

This closes the original question cleanly in the negative across two scales, and it
*reverses* phase 01's one surviving positive (long-range edges "earn their place"): at k=2
they do not. The next experiment, if pursued, must change the *task*, not the graph — a
synthetic long-range copy/recall benchmark where a 64-wide window is provably insufficient
— so that the long-range edges carry real signal and a Cayley-vs-random gap could finally
have something to modulate.

## File manifest

- `phase02.json` — per-arm, per-seed best val losses, edge counts, wall times, and the
  realized-graph spectral/diameter diagnostics; aggregated summary and run config.
- `spectral_gaps.png` — left: B's fixed spectral gap (0.075) against the wide C-seed spread
  (0.073–0.164); right: best val loss per seed for B vs C with the A-full mean as reference.
- `cayley_vs_random_n11_k2_README.md` — this file.
- `../../experiments/phase02_cayley_vs_random_n11_k2.py` — the experiment (k=2 directed
  Cayley construction, mask builder, nanoGPT with arbitrary boolean masked attention,
  training loop, diagnostics, plotting). Resumable via the runs in `phase02.json`.
- `data/shakespeare_char/{train.bin,val.bin,meta.pkl}` — cached dataset (gitignored).

## Open questions

1. **Does anything restore a Cayley advantage at fixed scale?** Both levers point the same
   way: (a) use a *symmetric* generator set (a, a⁻¹, b, b⁻¹) so the causal-masked graph
   stays denser/less fragmented; (b) use a **locality-preserving** token→node mapping so the
   surviving causal edges are not algebraically scattered. Worth one run before declaring
   the construction inert — but the E-beats-B result suggests even a better Cayley graph
   would not help while the window dominates.
2. **The real blocker is the task, not the graph.** Tiny Shakespeare has no long-range
   dependency a 64-window misses (E ≥ B at k=2, E ≈ B at k=4). A decisive Cayley-vs-random
   test needs a task where the window is provably insufficient — synthetic copy/sort/recall
   at length ≫ window, or true long-document modeling.
3. **Why does pure-Cayley (D) fail identically (std 1e-4) at both scales?** It collapses to
   the bigram floor deterministically. Is D's ceiling set by missing locality alone, or also
   by depth < diameter (8 < 13 here)? A deep windowless Cayley run (n_layer ≥ 13) would
   separate the two; phase 01's D at diameter 7 / 8 layers was already saturated, hinting
   locality is the binding constraint.
4. **Does full attention's slight underperformance vs sparse arms persist?** Here all sparse
   arms edged below the full mean (within noise). At larger compute/longer training does
   full re-assert a clear ceiling, or do local-biased sparse masks genuinely regularize
   char-level modeling at T=1320?
