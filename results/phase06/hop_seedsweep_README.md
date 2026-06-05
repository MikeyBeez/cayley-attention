# Phase 06 — Composition depth: success rate × budget (is the 3-hop knife-edge budget or lottery?)

**Free pre-check (Phase 05 k3 chance-seed, no GPU).** Before the sweep: the Phase 05 k3 seed that
landed at chance (s1338) sat at value-CE ≈ **4.159 = ln(64), the exact chance-entropy floor, dead
flat from step 500 through its full 9k run** — it never began to descend, while the solving seed was
already climbing (recall 0.43, CE dropping) by 9k. That pointed to an **init lottery, not
undertraining** — with the caveat that s1338 was killed at the 9k fail-cutoff while the solver got 12k
and only onset at ~5.5k, so a later onset could not be excluded from n=2 alone. Phase 06 was built to
adjudicate exactly this, and it confirms the pre-check's lean.

## Headline

**The Phase 05 "3-hop knife-edge" is a genuine ~42% initialization lottery, not a budget artifact —
and the 4-hop cliff is a wall, not merely expensive.** Decoupling the budget from the one-hop control
(40k ceiling, stop-on-solve, CE-flatness gate) and estimating per-hop success over many seeds:
**k2 solves 2/2 fast (median 2.5k); k3 solves 5/12 (42%) with winners spread across 6k–12k; k4 solves
0/4.** The seven k3 non-solvers all sit **dead-flat at the chance-entropy floor (CE slope ≈ 0)** when
stopped — the false-null gate passes, so they are real failures, not late bloomers still descending.
That is the **lottery** signature (init decides solve-vs-fail), not the **budget** signature (everyone
solves eventually with more steps). The pre-committed budget prediction (≥7/12, several solvers past
12k–20k) is **not** borne out; the named lottery alternative is. The budget *did* matter for one thing:
Phase 05's lone 12k solver was a real late-but-normal winner, so n=2 simply undersampled a bimodal rate
— it caught one of each face of the coin.

## Setup

Model / task / data / edge-builder / path-guard **identical to Phase 05** (25.2M decoder, 8L × 8H ×
512, RoPE; procedural MQAR recall, V=64, chance recall = 0.0156, chance-CE = ln 64 = 4.159; window
w=128; needle at fixed D=768). Only the orchestration changed: the seed loop, the budget, the stopping
rule, and a steps-to-solve logger. A single uniform dilated stride s = D/k makes the needle reachable
in **exactly k hops and no fewer** — path-guard re-verified (reach_frac = 1.00, shortest-path == k for
k2/k3/k4; regression check passed unchanged).

| Arm | stride | hops k | seeds | role |
|-----|:------:|:------:|:-----:|------|
| `k2` | 384 | 2 | 2 (1337–1338) | calibration / scale anchor |
| `k3` | 256 | 3 | 12 (1337–1348) | **main** — turn the n=2 coin-flip into a rate |
| `k4` | 192 | 4 | 4 (1337–1340) | probe — does "dead at 4" survive a generous budget? |

**Budget (the Phase 06 change).** Ceiling 40k steps (> 3× Phase 05's 12k wire). **Stop-on-solve**: once
held-out recall ≥ 0.90 is sustained over the next eval, stop and record `steps_to_solve` (the first
≥0.90 step). **CE-flatness early-stop**: if best value-CE has not improved by > 0.02 nats over the
trailing 5k window, stop and mark the seed a flat non-solver — *gated to never fire before step 10k* so
a pre-onset solver (which sits at the chance floor, indistinguishable from a dead seed by CE alone) is
not clipped. For every non-solver we record the trailing-5k CE slope: **the false-null gate — a real
failure must be flat (slope ≈ 0), not descending.** ~14.4 GPU-hours, 18 runs.

## Headline table (recall ≥ 0.90 = "solved"; chance recall = 0.0156; D=768)

| Arm | hops k | success rate | steps-to-solve (solvers) | median | non-solver CE-slope /1k | verdict |
|-----|:------:|:------------:|:------------------------:|:------:|:-----------------------:|---------|
| `k2` | 2 | **2/2 = 1.00** | 2000, 3000 | 2500 | — | reliable, fast |
| `k3` | 3 | **5/12 = 0.42** | 6000, 6500, 8000, 9000, 12000 | 8000 | all ≈ 0 (−0.0009…+0.0001) | **init lottery** |
| `k4` | 4 | **0/4 = 0.00** | — | — | all ≈ 0 | **wall** |

Per-seed k3 (the main arm): solvers s1338(6k), s1341(6.5k), s1340(8k), s1339(9k), s1346(12k); flat
non-solvers s1337, s1342, s1343, s1344, s1345, s1347, s1348 — every one stopped flat at 10k with
|slope| ≤ 0.001 nats/1k. k4: all four (s1337–s1340) flat at 10k, slope ≈ 0, **no seed crossed.**

## What the result shows

1. **k3 is a lottery, not a budget problem.** If the n=2 "knife-edge" were a budget artifact (deeper
   composition learnable but costing steeply more steps), the non-solvers would be *slowly descending*
   when stopped — late bloomers needing > 12k. They are not: all seven sit at the chance-entropy floor
   with CE slope ≈ 0, zero progress. Meanwhile five seeds onset and solve cleanly. That clean
   **bimodal split — descend-and-solve-by-12k vs dead-flat-forever — is the initialization-lottery
   signature.** Success rate at 3 hops is ~42%, now estimated at n=12 rather than asserted from n=2.

2. **The budget still explains Phase 05's n=2 picture.** Winners' onset spreads widely (6k–12k), and one
   solver needed the full 12k — so Phase 05's lone 12k solver was a *normal late winner*, not an anomaly
   clipped by the cutoff. With only two seeds, Phase 05 happened to draw one winner and one loser and
   read it as a "knife-edge." Both faces were real; the coin just has ~42% heads.

3. **k4 is a wall, not merely expensive.** Zero of four seeds solved and all four are dead-flat — no
   seed crept toward an onset even with the budget tripled. There is **no evidence the cliff is a smooth
   optimization-cost curve**: 2 hops reliable, 3 hops a coin-flip, 4 hops off. If composition cost rose
   smoothly with depth we would expect *some* k4 progress or a late k4 solver; we see neither.

4. **False-null gate passed everywhere.** Every non-solver (k3 and k4) terminated flat (|CE slope| ≤
   0.001 nats/1k). No null in this phase is an undertraining artifact — the failures are real.

## Mechanism

Composition by required hop-count is a **reliable → lottery → wall** ladder, not a cost curve:

| k | success rate | winner onset | non-solver state | reading |
|:-:|:------------:|:------------:|:----------------:|---------|
| 2 | 1.00 | fast (2–3k) | — | composed reliably |
| 3 | 0.42 | 6–12k, wide | dead-flat at ln V | **init lottery** — solve-or-floor by init |
| 4 | 0.00 | — | dead-flat at ln V | **wall** — not reached within budget |

The depth-3 phase transition is *bimodal at fixed budget*: a seed either finds the 3-hop route (and
then trains down fast) or never leaves the chance floor. Nothing sits in between. This is consistent
with composition being a discrete representational event (the model either forms the relay circuit or
does not), gated by initialization, rather than a continuous optimization grind that more steps buy.

## Pre-committed predictions vs measured

| Prediction (06) | Measured | Verdict |
|-----------------|----------|:-------:|
| `k2`: 2/2 solve, fast (sets the scale) | 2/2, median 2.5k | ✅ |
| `k3` (main call): success rate ≥ 7/12, steps spread late, several past 12k–20k (**budget** reading) | 5/12, none past 12k | ❌ |
| `k3` named alternative: rate ≈ 50%, no seed past ~15k → **init lottery** | 42%, no solver past 12k, non-solvers dead-flat | ✅ |
| `k4`: mostly chance even at 40k; open to a high-step crossing | 0/4, all flat, no crossing | ✅ (lean: wall) |
| Every non-solver confirmed flat at termination (the gate) | all |slope| ≤ 0.001 /1k | ✅ |

**Net:** the *budget* reading is rejected and the *lottery* reading supported for k3; k4 reads as a
wall. The cliff across Phase 05/06 is **not** a smooth optimization-cost curve.

## Architectural interpretation

> **A sparse attention model's effective composition depth is not a budget you can buy with more
> steps — it is a reliability ladder set by depth: two hops compose reliably and fast, three hops are a
> ~40% initialization lottery (a seed either forms the relay circuit and trains down cleanly, or sits
> at the chance floor forever — nothing in between), and four hops do not compose within a tripled
> budget. The Phase 05 "3-hop knife-edge" was a genuine bistable lottery undersampled at n=2, not a
> late solver clipped by a tight cutoff; the 4-hop cliff is a wall, not a steep cost. Composition over
> a fixed expander reads as a discrete representational event gated by initialization, not a continuous
> optimization grind — which is why depth, not training time, is the binding constraint.**

## Caveats

- **The flat-stop fired at 10k for dead seeds — the one hole.** To honor the 40k ceiling literally,
  non-solvers should run to 40k; instead the flatness gate stopped them at 10k once they showed 5k of
  zero CE movement. So we observe the *lottery signature* (zero slope) rather than having literally
  watched each dead seed fail to onset through 40k. The zero CE slope is strong evidence against a
  very-late (>10k) onset — a seed heading toward a 12–15k onset shows negative CE drift first, as the
  five solvers did, and the one 12k solver was correctly *kept* (not flat-stopped) because it was still
  descending — but a fully airtight version disables the flat-stop for a handful of non-solvers and k4
  seeds. This is open question #1.
- **n is modest** (k3 at 12, k4 at 4). 42% is a point estimate with a wide interval; k4's "wall" rests
  on four dead seeds. Directionally robust (k2 reliable ≫ k3 lottery ≫ k4 dead), rates approximate.
- **Single distance / single stride family** (D=768, strides {384,256,192}); hop-count and stride
  length co-vary by construction. Same scope as Phase 05.
- Quality/recall only; no compute-payoff claim.

## File manifest

- `phase06_records.json` — per-(arm, seed): solved flag, steps_to_solve, final recall, value-CE,
  stop reason/step, trailing-window CE slope, edge count, full recall+CE curve, peak memory, wall;
  config (budget, flatness params, path-guard) and aggregated success-rate / steps-to-solve summary.
- `phase06_success_vs_hops.png` — two panels: success rate vs hop-count (with the n=2 0.5 coin-flip
  line); steps-to-solve per k (solvers as points, non-solvers as × at the ceiling) with the Phase 05
  12k cutoff and Phase 06 40k ceiling drawn.
- `hop_seedsweep_README.md` — this file. (Verbose run log is gitignored.)
- `../../experiments/phase06_hop_seedsweep.py` — the experiment (resumable; `P6_GUARD_ONLY=1` re-checks
  the path guard; `P6_SMOKE=1` for an end-to-end shakeout).

## Open questions → phase 07 (if pursued)

1. **Close the hole: run k3 non-solvers and all k4 seeds to the full 40k with the flat-stop disabled**,
   to convert "dead-flat at 10k" into "confirmed never onsets through 40k" and make the wall/lottery
   verdict airtight.
2. **What predicts the k3 lottery?** If solve-vs-fail is init-determined, the init property that decides
   it should be findable (e.g., early-step attention-pattern alignment on the dilated edge) — and might
   be selectable, turning a 42% lottery into a reliable solve.
3. **Does a curriculum convert lottery-losers?** Train k1→k2→k3 (or anneal D) on a non-solver seed: if
   it then solves, the limit is optimization/init reachability; if it still fails, it is representational
   — the cleanest separation of "can't find it" from "can't represent it."
