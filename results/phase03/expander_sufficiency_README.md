# Phase 03 — Is an expander a *sufficient* attention scaffold? (WikiText-103, long context)

**First pass: inconclusive — the validity gate failed.** A sparse expander scaffold (256-wide
window + 8 long-range edges/token, O(N)) reaches full-attention perplexity on WikiText-103 at
both T=1024 and T=4096 — *but so does window-only attention with zero long-range edges*. So the
match is the trivial kind (a local window already suffices on this data at this length, exactly as
on tiny Shakespeare), not the load-bearing kind (the expander's long-range structure earning the
match). Because the long-range edges carried no measurable load even at T=4096, this run cannot
say whether the *expander* is sufficient — only that *local* attention is. The construction null
(Cayley = random) replicated a third time. Two confounds — a 256-token window wide enough to cover
WikiText's dependencies, and undertraining (0.69 epoch, no plateau) — both point to the same fix,
pursued in **phase 03b**: bottleneck the window so the long-range edges *must* work.

## Headline table (val loss mean ± std over seeds; token-level perplexity)

| Arm | T=1024 (n=2) | ppl | T=4096 (n=2) | ppl |
|-----|:------------:|:---:|:------------:|:---:|
| **A — Full (baseline)** | 3.8555 ± 0.0063 | 47.25 | 4.2491 ± 0.0255 | 70.04 |
| **B — Expander** (win256 + 8 long-range) | 3.8760 ± 0.0013 | 48.23 | 4.1674 ± 0.0206 | 64.55 |
| **C — Window-only** (win256) | 3.8749 ± 0.0022 | 48.18 | 4.1613 ± 0.0561 | 64.15 |
| **D — Cayley** (win256 + 8 Cayley) | — | — | 4.1468 (n=1) | 63.23 |

Plus exact-Cayley construction point at T=4896 (= \|SL(2,ℤ₁₇)\|, node=position): B 4.0712 / D 4.0423.

Model: GPT-2-small (12L/12H/768, 124M params), identical across arms at each T. SDPA attention —
full uses flash `is_causal`; sparse uses an additive `(1,1,T,T)` mask via the memory-efficient
backend (scales to T=4096 without materializing B·H·T·T scores; ~9.7 GB peak). 10,000 steps
(8192 tok/step ≈ 0.69 epoch), AdamW lr 3e-4 cosine. ~9.7 GPU-hours for 15 runs.

## The two headline comparisons

**Sufficiency (B − A):** −0.082 at T=4096 and +0.021 at T=1024. Within the ~0.02–0.06-nat seed
noise, B ≈ A at both lengths — the sparse expander reaches full-attention loss. So far so on-thesis,
**but see the load check before reading anything into it.**

**Load check (C − B) — THE VALIDITY GATE — FAILED:**

| T | C − B (window-only minus expander) | gate |
|---|:----------------------------------:|:----:|
| 1024 | −0.0011 | closed (expected — window reaches 25%) |
| 4096 | **−0.0062** | **closed (should have opened)** |

At T=4096 the per-seed C values are 4.2009 (seed 1337) and 4.1217 (seed 1338) — the second
window-only seed beat *every other run in the phase*. Across both seeds, **window-only ties the
full expander**: the 8 long-range edges add nothing the 256-wide window didn't already provide,
even at 4096 tokens. My initial single-seed read ("C is +0.048 worse than B → gate open") was
noise; the second seed erased it.

Per the pre-committed criterion 2: *if C ≈ B even at T=4096, the scaffold isn't loaded and the
test is inconclusive — "need longer context / smaller window," not "scaffold sufficient."* That is
exactly the branch we landed on.

## Construction null (D vs B) — confirmed a third time

| Setting | B (random) | D (Cayley) | D − B |
|---------|:----------:|:----------:|:-----:|
| T=4096, scaled-Cayley | 4.1674 | 4.1468 | −0.021 |
| T=4896, exact node=position (n=17) | 4.0712 | 4.0423 | −0.029 |

D ≈ B (within noise, D marginally lower) at both the scaled spec'd length and the exact Cayley
size. After phases 01 (N=336, k=4) and 02 (N=1320, k=2), this is the third scale at which the
deterministic Cayley construction is statistically indistinguishable from a random expander. The
construction question is closed; random is the right default.

## Undertraining (your number, made explicit)

10,000 steps × 8,192 tokens = 81.9M tokens vs a 119M-token train set → **0.69 epoch**. The model
had not plateaued — val loss was still falling ~0.10–0.12 nats per 2,000 steps at the end:

| step | A_full val (T=4096) | B_random val (T=4096) |
|------|:-------------------:|:---------------------:|
| 2000 | 5.564 | 5.472 |
| 6000 | 4.597 | 4.488 |
| 10000 | 4.231 | 4.153 |

Token-level ppl ~63–70 (word-level ~120) is therefore an undertrained snapshot — roughly 2–3× a
converged GPT-2-small (token-ppl ~22–25). The *relative* B-vs-A-vs-C comparison is valid at any
shared budget, but undertraining is a real confound for the load check specifically: a model that
has barely learned the data has not yet leaned on long-range structure, which flatters "window
suffices." This is why 03b doubles the step budget *and* shrinks the window.

(Note: cross-T ppl is not comparable — T=4096 at batch 2 sees far fewer distinct sequence-starts
per step than T=1024 at batch 8, so it trains slower per step. Only within-T comparisons are
meaningful.)

## Pre-committed predictions vs measured

| Prediction | Measured | Verdict |
|------------|----------|:-------:|
| A is the ceiling | sparse arms within noise of A at both T | ~ (sparse ≈ A) |
| Thesis: B ≈ A at both T | B−A = +0.02 / −0.08 (within noise) | ✅ direction |
| Field (Yi et al.): B falls behind at T=4096 | B did *not* fall behind | ❌ for the field |
| Load check: C ≈ B at small T, C ≫ B at T=4096 | C ≈ B at **both** T | ❌ gate never opened |
| Construction: D ≈ B | confirmed, 3rd scale | ✅ |
| *Named fallback:* "if C ≈ B at 4096, need longer context" | **this is what happened** | ✅ (fallback) |

## Architectural interpretation

> **On WikiText-103 a 256-token local window already matches full attention out to 4096 tokens, so
> this run cannot adjudicate the expander thesis — it adjudicates only that local attention is
> enough for this data.** Both the proposed scaffold (window + long-range) and a pure window reach
> the full-attention baseline within seed noise at every length tested; the long-range edges, the
> object of the thesis, are dead weight here. Against the Yi et al. prior (approximate attention
> underperforms full attention at long context), the sparse arms did *not* underperform — but that
> is uninformative, because the cheapest possible "approximate" method (a local band with no
> long-range at all) also did not underperform. The honest conclusion is not "expander sufficient"
> and not "expander insufficient" but **"WikiText-103 at ≤4096 tokens with a 256-window does not
> load the long-range edges"** — the task does not exercise the mechanism under test. To make the
> thesis falsifiable the scaffold must first be *loaded*: shrink the window until local attention
> provably fails, then ask whether O(1) long-range edges recover full-attention quality. That is
> phase 03b. The construction sub-question, by contrast, is now settled three times over: Cayley
> and random expanders are interchangeable.

## File manifest

- `phase03.json` — per-(T, arm, seed) best val loss/ppl, edge counts, window-reach fractions,
  long-range connectivity diagnostics, peak memory, wall times; aggregated gaps.
- `sufficiency_curve.png` — per-arm ppl vs T, and the B−A / C−B gaps vs T.
- `expander_sufficiency_README.md` — this file.
- `../../experiments/phase03_expander_sufficiency_wikitext.py` — the experiment.
- `data/wikitext103/{train,val}.bin` — GPT-2-BPE tokenized corpus (gitignored).

## Open questions → phase 03b

1. **Does the gate open when the window is bottlenecked?** 03b holds T=4096 and sweeps w ∈
   {128, 64}. At w=64 the window reaches only 1.6% of context; window-only (C) should fail, and the
   live question becomes whether 8 long-range edges (B) recover full-attention quality.
2. **Does the sufficiency result survive more training?** 03b doubles to 20,000 steps (~1.4 epoch)
   and logs the approach-to-plateau.
3. **Is WikiText-103 even the right task?** If no window setting cleanly separates C from B, the
   data may simply lack >window-scale dependency, and the decisive test needs a task with
   guaranteed long-range structure (synthetic copy/recall, or true long documents) rather than a
   narrower window.
4. **The compute payoff is still untested.** This phase measured quality only; the masked-dense /
   SDPA path is slower, not faster, than flash full attention. The O(N) win needs a real sparse
   kernel — a separate phase, kept distinct from the sufficiency claim.
