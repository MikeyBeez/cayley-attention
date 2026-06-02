# Phase 03b — Loading the scaffold: window sweep at T=4096 (WikiText-103)

**The long-range edges are not load-bearing at any window width — and the window alone matches
full attention down to a remarkably narrow 128 tokens.** Phase 03 was inconclusive because a
256-wide window already matched full attention, leaving the long-range edges idle. So 03b held
T=4096 and shrank the window (w ∈ {128, 64}, ~1.4-epoch training) to force the issue. The result
is a clean negative for the expander thesis on this task: **window-only attention (C) equals full
attention at w=128 (ppl 38.06 vs 38.07), reaching only 3% of the context** — and adding 8 random
long-range edges (B) never helps. When the window suffices (w≥128) the long-range edges are idle
or mildly *harmful*; when the window is finally too narrow to suffice (w=64), the sparse arms drop
0.08 nats below full (replicating the field's "approximate attention loses at long context") — but
the long-range edges **fail to recover the deficit** (B ≈ C). Across phases 01–03b the through-line
holds at full strength: **the local window does the work; the long-range / expander structure
(Cayley or random) is not load-bearing on natural language at these scales.**

## Headline table (T=4096; baseline A = full causal, ppl 38.07)

| Window w | B — Expander (w + 8 random long-range) | C — Window-only | B−A (sufficiency) | C−B (load gate) |
|:--------:|:--------------------------------------:|:---------------:|:-----------------:|:---------------:|
| **128** | 3.6708 · ppl 39.28 (n=1) | **3.6393 · ppl 38.06 (n=1)** | +0.0314 | **−0.0315** |
| **64**  | 3.7204 ± 0.0106 · ppl 41.28 (n=2) | 3.7237 ± 0.0098 · ppl 41.42 (n=2) | +0.0811 | +0.0033 |

(Baseline **A_full = 3.6394 ± 0.0002, ppl 38.07**, n=2 — window-independent, two seeds agreeing to
0.0003. Phase 03's w=256 point: B ≈ C ≈ A, all tied.)

Model: GPT-2-small (124M), identical across arms; SDPA — full uses flash `is_causal`, sparse uses
an additive `(1,1,T,T)` mask via the memory-efficient backend (~9.7 GB peak at T=4096). 20,000
steps (~1.4 epoch; doubled vs phase 03's 10k), AdamW lr 3e-4 cosine. ~10 GPU-hours, 8 runs.

## What the sweep shows

Reading the windows together (w=256 from phase 03, w=128/64 here):

| w | window reaches | C (window-only) vs full | long-range edges (B vs C) |
|:-:|:--------------:|:-----------------------:|:--------------------------:|
| 256 | 6.25% of 4096 | ties full (≈A) | idle (B ≈ C) |
| 128 | 3.1% | **still ties full (≈A)** | mildly **harmful** (B = C + 0.032) |
| 64  | 1.6% | falls 0.084 below full | **fail to help** (B ≈ C, both ≈0.08 below A) |

1. **Window-only matches full attention down to w=128.** A purely local 128-token causal band
   reaches the same val perplexity as full 4096-token attention (38.06 vs 38.07). For WikiText-103
   next-token prediction at this scale, ~128 tokens of local context is all that full attention is
   effectively using. The long-range structure the expander exists to provide is simply not needed.
2. **The long-range edges never earn their place — and at w=128 they slightly hurt.** Adding 8
   random long-range edges to a sufficient window made it *worse* (B−C = +0.032 at w=128; single
   seed, so tentative). When the window already covers the dependency, the extra random distant
   connections act as mild distraction, not signal.
3. **When the window finally fails (w=64), the long-range edges don't rescue it.** Both sparse arms
   fall ~0.08 nats below full — sparse now genuinely underperforms full attention, matching the Yi
   et al. expectation. But B ≈ C (gap +0.003, within the ~0.01 seed noise): the 8 long-range edges
   recover none of the deficit.

## Mechanism — and the design mismatch this exposes

Why do long-range edges fail to compensate at w=64? **Shrinking the window from 128→64 removes
*mid-range local* information (positions ~64–128 back). But the long-range edges are sampled
*uniformly at random* over all 4096 positions, so they almost never land in that just-outside-the-
window band — they overwhelmingly point far away, where WikiText has little next-token-relevant
dependency.** I removed mid-local context and added uniform-long edges; those cover different
ranges. The natural fix is to make the long-range edges *mid-range / dilated* (strided just past
the window) rather than uniform-random — i.e. match the sparsity to where the removed dependency
actually lives. That is a concrete next iteration, not a rescue of the current design.

This also reframes the deeper obstacle: **the task does not have the long-range dependency the
thesis needs.** If 128 local tokens already match full attention at 4096, then no long-range
scaffold can demonstrate value on WikiText-103 — there is nothing >128 tokens away for it to carry.
A decisive test of "expander sufficient under real long-range load" requires a task where >window
information is provably necessary (synthetic copy/recall/sort at length ≫ window, or long-document
QA), not natural-language LM at moderate context.

## Pre-committed predictions vs measured

| Prediction (03b) | Measured | Verdict |
|------------------|----------|:-------:|
| Narrowing window opens the load gate (C ≫ B at small w) | gate never opened: B ≈ C at w=64; long-range mildly *hurts* at w=128 | ❌ |
| B (expander) matches full once scaffold loaded | B falls 0.08 below full at w=64 | ❌ (sufficiency fails) |
| Sparse underperforms full at narrow window (field/Yi et al.) | yes, by 0.08 at w=64 | ✅ (for the field) |
| Doubling training improves trust | baseline ppl 70 → 38 from 10k → 20k steps | ✅ |

## Architectural interpretation

> **A local window is the whole of the working scaffold; the long-range / expander component is
> inert-to-harmful for WikiText-103 language modeling at 4096 tokens.** A 128-token causal band — 3%
> of the context — matches full attention exactly; widening the receptive field with a constant
> number of uniformly-random long-range edges does not improve on it and, when the window already
> suffices, slightly degrades it. Only by starving the window to 64 tokens does sparse attention fall
> below full (reproducing the field's finding that approximate attention underperforms at long
> context), and there the long-range edges recover none of the loss, because uniform-random edges
> miss the mid-range band where the removed dependency lives. The expander-sufficiency thesis is
> therefore **not supported on this task** — but the cleanest statement of why is that the task does
> not load it: WikiText-103 next-token prediction needs only ~128 tokens, so there is no long-range
> dependency for any scaffold, structured or random, to be sufficient *for*. The construction result
> from phases 01–03 (Cayley = random) and the sufficiency result here point to the same conclusion
> from opposite sides: at these scales the only load-bearing part of sparse attention is the local
> window, and the long-range graph — however it is built — is carrying nothing. To test the thesis
> as intended, change the task (guaranteed long-range dependency) and/or the long-range design
> (mid-range/dilated edges matched to the window gap), not just the window width.

## Caveats

- **w=128 is single-seed** (n=1 for B and C); the "long-range mildly hurts" point (B−C = +0.032) is
  larger than the w=64 seed noise (~0.01) but should be confirmed with a second seed before being
  leaned on. The w=64 quartet is 2-seed and solid.
- **~1.4 epoch, not fully converged** (val still falling ~0.03 nats/2k steps at the end) — much
  better than phase 03's 0.69 epoch, but absolute ppl (~38 token-level) is still ~1.5× a converged
  GPT-2-small. The *relative* gaps are the result; they held at a more-trained operating point than
  phase 03, which strengthens (does not prove) their persistence at convergence.
- Quality only; no compute-payoff claim (masked-dense/SDPA is slower than flash full attention).

## File manifest

- `phase03b.json` — per-(arm, window, seed) best val loss/ppl, edges, window-reach, connectivity,
  peak memory, wall times; aggregated B−A / C−B gaps per window.
- `window_load_curve.png` — left: ppl vs window for B and C against the full-attention baseline;
  right: B−A (sufficiency) and C−B (load) gaps vs window.
- `window_load_README.md` — this file.
- `../../experiments/phase03b_window_sweep_load.py` — the experiment.

## Open questions → phase 04 (if pursued)

1. **Mid-range / dilated long-range edges.** Replace uniform-random long-range with strided edges
   just past the window (the band actually removed when the window shrinks). Does *that* recover the
   w=64 deficit while staying O(N)? This is the design the mechanism points to.
2. **A task that loads the scaffold.** Synthetic copy/recall/sort at length ≫ window, or
   long-document QA — anything where >window information is provably required — so "expander
   sufficient under long-range load" becomes testable at all. The headline obstacle is the task, not
   the graph.
3. **Confirm the w=128 "long-range hurts" with seeds**, and push to convergence at one (window,
   task) setting before any sufficiency claim.
