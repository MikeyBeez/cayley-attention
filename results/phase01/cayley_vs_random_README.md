# Phase 01 — Cayley-graph vs random sparse attention (tiny Shakespeare)

**The headline hypothesis did not hold. The construction is not load-bearing at this
scale.** At a matched 4-edge/token long-range budget on char-level tiny Shakespeare, a
deterministic SL(2,ℤ₇) Cayley-expander connectivity (arm **B**) and a BigBird-style
random connectivity (arm **C**) reach essentially identical validation loss
(1.4744 vs 1.4756 nats; difference −0.0012, inside seed noise) — *and identical
variance across 5 seeds* (std 0.0033 vs 0.0023). The pre-committed signature
"std(B) < std(C)" is **false**: Cayley's determinism did not buy lower run-to-run
variance, because random sparsity at this scale is already as consistent as the
structured graph. Both expander-sparse arms sit within ~0.012–0.014 nats of full causal
attention (arm **A**, 1.4620) — a clean replication of BigBird's quality-preservation
claim on this dataset. The two ablations behaved as predicted: removing the local window
(arm **D**, pure Cayley) is catastrophic (+0.91 nats), while removing the long-range
edges (arm **E**, window-only) costs only +0.005 nats over B. The realized-graph
diagnostic actively subverts the premise: the *random* graphs are **higher**-spectral-gap
expanders (≈0.79) than the causal-masked Cayley graph (0.586), so "guaranteed expander"
is not even the property on which Cayley wins here.

## Headline table

| Arm | Connectivity | Best val loss (mean ± std, nats) | Perplexity | Gap to full |
|-----|-------------|:-------------------------------:|:----------:|:-----------:|
| **A — Full** | full causal | **1.4620 ± 0.0022** | 4.315 | — (ceiling) |
| **B — Cayley** | window-64 + 4 Cayley | **1.4744 ± 0.0033** | 4.368 | **+0.0124** |
| **C — Random** | window-64 + 4 random | **1.4756 ± 0.0023** | 4.374 | **+0.0137** |
| **D — Pure Cayley** | 4 Cayley, no window | 2.3886 ± 0.0019 | 10.898 | +0.9266 |
| **E — Window only** | window-64, no long-range | 1.4793 ± 0.0030 | 4.390 | +0.0173 |

Seeds: A/D/E ×3, B/C ×5. All arms share an identical 14.19M-param model
(`n_layer=8, n_head=6, n_embd=384, dropout=0.2, block_size=336`), optimizer, LR schedule,
`max_iters=3500`, `batch_size=64`; **only the attention mask differs**. ~7.7 min/run on an
RTX 5070 Ti, ~2.4 GPU-hours total. Params/FLOPs are identical across arms at this
block_size — this phase measures **quality preservation and variance**, not speed.

## Per-arm results

**A — Full causal (ceiling).** 1.4620 ± 0.0022 over 3 seeds (1.4597 / 1.4622 / 1.4641).
Standard nanoGPT char-Shakespeare territory. This is the number every sparse arm is
chasing.

**B — Cayley (the proposal).** 1.4744 ± 0.0033 over 5 seeds
(1.4699 / 1.4736 / 1.4746 / 1.4747 / 1.4791). **+0.0124 nats from full** — well inside the
0.05-nat success threshold. The fixed Cayley graph supplies 4 long-range skip edges/token
on top of the 64-wide causal window; total 19,970 allowed (i,j) pairs.

**C — Random (BigBird control, matched budget).** 1.4756 ± 0.0023 over 5 seeds
(1.4726 / 1.4743 / 1.4755 / 1.4772 / 1.4786). 20,215 allowed pairs — budget matched to B
within 1.2%. **+0.0137 from full.** Statistically indistinguishable from B in both mean
(Δ = −0.0012) and spread (std 0.0023 vs B's 0.0033).

**D — Pure Cayley, no window (ablation: is the window necessary?).**
2.3886 ± 0.0019. **+0.9266 nats — catastrophic, and the single largest effect in the
study.** With only 4 edges/token and no locality, the model cannot represent local
character dependencies at all and collapses toward a bigram-grade predictor (ppl 10.9 vs
~4.3). The window is unambiguously necessary; long-range composition alone, at 8 layers,
does not substitute for local detail.

**E — Window only, no long-range (ablation: are the long-range edges necessary?).**
1.4793 ± 0.0030. Worst of the three windowed arms on every seed, but only **+0.0049 over
B** and +0.0037 over C. The 4 long-range edges *do* earn their place — consistently, in
the predicted direction — but the margin is small: at sequence length 336 the 64-wide
window already captures almost all of the available signal, and the skip edges close only
~28% of the window-only-to-full gap.

## Connectivity diagnostic (realized causal-masked long-range graph)

| Graph | Spectral gap (λ₁−λ₂) | Components | Giant | Diameter |
|-------|:--------------------:|:----------:|:-----:|:--------:|
| **B — Cayley (fixed, all seeds)** | **0.586** | 1 | 336 | 7 |
| C — random seed 1337 | 0.795 | 3 | 334 | 9 |
| C — random seed 1338 | 0.789 | 2 | 335 | 9 |
| C — random seed 1339 | 0.817 | 8 | 329 | 8 |
| C — random seed 1340 | 0.776 | 4 | 333 | 10 |
| C — random seed 1341 | 0.799 | 4 | 333 | 9 |

The diagnostic is the most interesting twist. After intersecting with the causal mask and
symmetrizing, **the random graphs are the *better* spectral expanders** (gap ≈0.79,
tightly clustered across seeds) than the Cayley graph (0.586). What determinism actually
buys is **connectivity guarantees, not gap**: Cayley is always a single connected
component reaching every node; the random graphs fragment into 2–8 components with a
handful of orphaned positions (giant 329–335 of 336) and diameters 8–10. Yet neither the
higher random gap nor the Cayley connectivity guarantee translated into any measurable
task-loss difference — quality and variance are flat across both. See
`spectral_gaps.png`.

## Pre-committed predictions vs measured

| Prediction (pre-committed) | Measured | Verdict |
|----------------------------|----------|:-------:|
| A lowest loss (ceiling) | A = 1.4620, lowest | ✅ |
| B and C both close to A (expander-sparse preserves quality) | +0.012 / +0.014 nats | ✅ |
| B ≈ C in mean **but std(B) < std(C)** (the predicted signature) | means equal; std(B)=0.0033 **≥** std(C)=0.0023 | ❌ |
| D clearly worse than B | +0.91 nats | ✅ |
| E worse than B (long-range earns its place) | +0.005 nats, worst windowed arm | ✅ (small margin) |
| *Alt:* if std(B)≈std(C) & means match → any expander suffices, construction not load-bearing | **this is what happened** | ✅ (alternative) |

### Success criteria (binary)

- **std(B) < std(C) by more than the std-of-std?** **NO.** std(B)=0.0033, std(C)=0.0023;
  the difference (0.001) is smaller than the sampling uncertainty of a 5-seed std estimate
  (≈ std/√8 ≈ 0.001). No detectable variance advantage in either direction.
- **B within 0.05 nats of A?** **YES** (+0.0124).
- **D meaningfully worse than B (≥0.05 nats)?** **YES**, by 0.91 nats.

## Architectural interpretation

The load-bearing comparison (B vs C) came back null, and the secondary diagnostic
explains *why* the original mechanism story doesn't apply at this scale. The publishable
framing is the pre-named alternative:

> **At small scale, the expander *property* — not the algebraic *construction* — is what
> preserves attention quality, and even the property is cheap to obtain.** On char-level
> tiny Shakespeare with a 336-token context, a deterministic SL(2,ℤ₇) Cayley graph and a
> seed-varied random graph at equal 4-edge/token budget are statistically identical in
> both mean validation loss (Δ ≈ 0.001 nats) and seed variance (std ≈ 0.002–0.003 nats),
> with both within ~0.013 nats of full causal attention. The motivating intuition — that
> random sparsity "varies run to run and can occasionally realize a poor expander," so a
> guaranteed expander should win on consistency — does not materialize: at n=336 the
> random construction is *already* a consistent, high-spectral-gap expander (gap ≈0.79 vs
> Cayley's causal-masked 0.59), and its only structural deficit, mild fragmentation into a
> few orphaned nodes, is invisible to the language-modeling objective. Determinism buys a
> connectivity guarantee, not a quality or variance edge. The components that *are*
> load-bearing are orthogonal to the Cayley-vs-random axis: the local window is essential
> (removing it costs 0.91 nats), and the long-range edges contribute a small but
> consistent +0.005-nat improvement over window-only regardless of whether they are
> structured or random. **The case for a deterministic algebraic expander must therefore
> be made at a scale where random graphs stop being reliable expanders — large N, large
> edge-budget regimes where the variance the hypothesis predicts would actually appear —
> not at the 336-token scale, where the two are interchangeable.**

This is a clean, reportable negative on the headline and a clean positive on the
window-necessity ablation. It also sharpens the next experiment: the Cayley-vs-random
question is only meaningful in the regime where a random graph's expander quality becomes
unreliable, which is precisely the large-N regime where the deterministic construction's
O(1)-per-token, no-storage, no-sampling advantages also begin to matter.

## File manifest

- `phase01.json` — per-arm, per-seed best val losses, edge counts, wall times, and the
  realized-graph spectral/diameter diagnostics; plus the aggregated summary and run config.
- `spectral_gaps.png` — left: spectral gap of B (fixed) vs the 5 C-seeds; right: best val
  loss per seed for B vs C with the A-full mean as reference line.
- `cayley_vs_random_README.md` — this file.
- `../../experiments/phase01_cayley_vs_random_sparse.py` — the experiment (data prep,
  Cayley construction, mask builder, nanoGPT model with arbitrary boolean masked
  attention, training loop, diagnostics, plotting). Resumable via the runs already present
  in `phase01.json`.
- `data/shakespeare_char/{train.bin,val.bin,meta.pkl}` — cached dataset (gitignored,
  regenerated from the source corpus).

## Open questions

1. **Does the variance signature appear at large N?** The hypothesis predicts std(Cayley)
   < std(random) precisely when random graphs stop being reliable expanders. At n=336 they
   are reliable. The decisive test is large N (n=11 → 1320, n=13 → 2184) and/or a smaller
   edge budget (k=2), where a random graph is more likely to realize a poor or
   disconnected expander.
2. **Why is the causal-masked Cayley gap *lower* than random?** Causal masking removes the
   half of the symmetric Cayley adjacency above the diagonal; symmetrizing the remainder
   recovers connectivity but degrades the spectral gap below that of an unconstrained
   random graph. A locality-preserving token→node mapping, or using both a generator and
   its inverse to make the causal-masked graph denser, might restore Cayley's gap
   advantage — and is worth testing before concluding the construction is inert.
3. **Why do the long-range edges help so little (E ≈ B)?** Tiny Shakespeare may simply
   lack long-range dependencies that a 64-wide window misses. A task with genuine
   long-range structure (e.g. a synthetic copy/recall benchmark, or longer-context
   documents) would give the long-range edges something to do and could re-open a
   Cayley-vs-random gap that the local window currently masks.
4. **Is 8 layers enough for D to compose?** Pure Cayley collapsed; at diameter 7 the
   long-range edges need ≥7 hops to reach all-pairs, and 8 layers may be too few to *also*
   reconstruct locality from scratch. Deeper pure-Cayley models would isolate whether D's
   failure is missing locality (likely) or insufficient composition depth.
