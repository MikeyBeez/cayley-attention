# Phase 07 — A fixed 100-edge graph matches full attention at D=256 (~7–13× saving) — and spreading the edges buys a ~7× training-speed advantage, not reach

**A 100-edge-per-token graph — ~6.6× fewer edges than full causal attention on average (~13× at the
query end) — matches full-attention recall on the exact D=256 needle the sparser Cayley graphs used to
miss, and it does so whether the spread edges are deterministic or random (construction null, 5th
time). The twist: the equal-budget *contiguous* window, predicted to reproduce Phase 06's 3-hop
lottery, did not fail — it solved all four seeds too, but took ~7× longer (median 12,250 steps vs 1,750
for the 2-hop spread graphs). So at this budget the offset structure decides optimization *speed*, not
ultimate success: keeping every route at 2 hops makes the 100-edge graph both reliable and fast, while
bunching the same 100 edges into a width-100 window still gets there via thousands of redundant 3-hop
paths — just slowly.** This refines Phase 06: its "3-hop = 42% lottery" was a property of a *single*
3-hop thread (one dilated stride), not of hop-count itself — give a 3-hop route massive path
redundancy and it becomes reliable-but-costly rather than a coin-flip.

## Setup

Model / task / generator identical to Phase 04–06 except **L=6** and **T=1320** (= |SL(2,ℤ₁₁)|, the
Phase-02 long-context length). 19.0M-param decoder (6L × 8H × 512, RoPE, weight-tied); procedural MQAR
recall, V=64, chance recall = 0.0156; needle at fixed **D=256** (the distance the degree-4 Cayley graph
failed on in the earlier work, a median of 4 hops away). Sanity regime D_in=8.

**The manipulation: identical 100-edge budget, only the offset structure differs.** Each structured
arm gives every token exactly 100 backward edges; the arms differ *only* in which offsets:

| Arm | 100-edge offset set | reach guard: hops→256 | direct edge? | predicted regime |
|-----|---------------------|:---------------------:|:------------:|------------------|
| `A_full` | full causal (ceiling) | 1 | — | ceiling |
| `C_window100` | contiguous {1..100} (**bunched**) | **3** | no | lottery (P06 3-hop) |
| `B_expander100` | deterministic **spread** | **2** | no | reliable |
| `B_random100` | random **spread** (per seed) | **2** (all seeds) | no | reliable |

**Reach guard (proven before training, not asserted):** `C_window100` reaches D=256 in exactly 3 hops,
`B_expander100`/`B_random100` in exactly 2, all with reach_frac=1.00 and **no direct 256-edge** (so it
is genuine composition, not a planted shortcut). The contiguous window and the spread expander carry
the *same* budget and differ only in bunched-vs-spread — that is the whole experiment.

> **Note on "expander."** A literal degree-100 SL(2,ℤₙ) Cayley graph is ill-defined (the group
> construction is fixed low degree, and its scatter offsets can't be tuned to avoid a direct 256-edge
> or guarantee a 2-hop bound). Phases 01–06 nulled Cayley-vs-random four times, so the faithful
> realization of *this* phase's question (bunched vs spread at fixed budget) is three 100-offset sets
> differing only in *which* offsets; `B_random100` is the matched random control.

**Budget / stopping (closes Phase 06 open-question #1):** ceiling 40k steps, stop-on-solve (recall ≥
0.90 sustained), and **flat-stop OFF** — every seed runs until it solves or hits 40k, so no seed is
called a non-solver from a 10k flatness gate. In the event, **no seed ran to 40k — all 20 cells solved**
(16 main + 4 sanity), so the dead-seed cost never materialized. ~7.9 GPU-hours total.

## Headline table (recall @ D=256; chance = 0.0156; n=4 seeds)

| Arm | hops→256 | seeds solving | recall (median) | steps-to-solve (median) | per-seed steps-to-solve |
|-----|:--------:|:-------------:|:---------------:|:-----------------------:|-------------------------|
| `A_full` | 1 | **4/4** | 0.989 | **1,750** | 500, 1000, 2500, 2500 |
| `C_window100` | 3 | **4/4** | 0.971 | **12,250** | 6000, 11500, 13000, 13000 |
| `B_expander100` | 2 | **4/4** | 0.975 | **1,750** | 1500, 1500, 2000, 4500 |
| `B_random100` | 2 | **4/4** | 0.988 | **2,250** | 1500, 2000, 2500, 3500 |

Sanity gate (D_in=8): all four arms ≥ 0.998 — the task and training are sound, so D=256 behaviour is a
reach/regime effect, not a training failure. Edge budgets: full 871,860 (avg 660/tok); structured arms
~121k–128k (avg ~92–97/tok — the bunched window carries marginally *more* total edges than the spread
arms due to boundary fall-off, so the comparison is conservative). At the query positions (q ≥ 256)
every structured arm has exactly degree 100.

## What the result shows

1. **Viability confirmed — a 100-edge spread graph matches full attention.** `B_expander100` (0.975)
   and `B_random100` (0.988) reach **≥ 98% of `A_full`'s 0.989** recall, all four seeds, on a task that
   genuinely needs the reach (`C_window`'s 3-hop distance and the earlier Cayley graph's median-4 both
   testify D=256 is past trivial). At ~6.6× fewer edges on average (≈13× at the query end), the sparse
   construction does not cost reach — *provided the edges are spread so every route stays at 2 hops.*

2. **Construction null, a fifth time.** `B_random100 ≈ B_expander100` (both 4/4, recall within ~1.5
   points, both fast). The *spread* property — not whether the spread is algebraic (Cayley-like) or
   random — is what buys the 2-hop reach. The repo's recurring finding survives into the
   practical-viability regime.

3. **The pre-committed "structure decides the regime" prediction is refuted — and refined.** The
   prediction was that `C_window100`, at 3 hops, would reproduce Phase 06's ~42% lottery while the
   spread arms stayed reliable. Instead **`C_window100` solved all four seeds** — it did *not* land in
   the lottery. What separates the arms is not success but **speed**: the 2-hop spread graphs solve at a
   median of 1,750 steps (as fast as the 1-hop full attention), the 3-hop window at 12,250 — a **~7×
   optimization-time penalty for the extra hop.** At equal budget, spreading the edges buys training
   *efficiency*, not reach.

4. **Why this differs from Phase 06 — path multiplicity, not hop-count.** Phase 06's 3-hop lottery used
   a window **plus a single dilated stride**: the 3-hop route to D=768 was one sparse thread, and that
   *single-path* route was the coin-flip. `C_window100` is a *contiguous width-100 window* — it offers
   an enormous number of redundant 3-hop paths to D=256 (any a+b+c = 256 with each ≤ 100). That
   redundancy converts the lottery into a reliable-but-slow solve. **So Phase 06's "3-hop = lottery" is
   really "single-path 3-hop = lottery"; hop-count sets the optimization *cost*, path multiplicity sets
   the *reliability*.**

## Mechanism

Two orthogonal knobs, cleanly separated by Phase 06 + 07:

| route property | controls | evidence |
|----------------|----------|----------|
| **composition depth** (hops) | optimization *cost* | 1-hop & 2-hop solve ~1,750 steps; 3-hop ~12,250 (~7×) |
| **path multiplicity** | *reliability* | single 3-hop thread (P06) → 42% lottery; redundant 3-hop window (P07) → 4/4 |

The practical consequence is a clean efficiency story: a fixed 100-edge graph matches full attention,
and **spreading those edges to keep every route at 2 hops makes it both reliable (like the redundant
window) and fast (like full attention) at once** — the bunched window gets only the reliability, paying
~7× in steps for the third hop, and a *sparse* 3-hop route (P06) would get neither.

## Pre-committed predictions vs measured

| Prediction (07) | Measured | Verdict |
|-----------------|----------|:-------:|
| D_in=8: all arms ~100% (sanity) | 0.998–1.000 | ✅ |
| `A_full`: ~100%, all seeds | 0.989, 4/4 | ✅ |
| `C_window100`: ~40% of seeds solve (P06 3-hop lottery) | **4/4 solve, but ~7× slower** | ❌ (reliable-but-slow, not lottery) |
| `B_expander100`: ~100%, all-or-nearly-all seeds | 0.975, 4/4 | ✅ |
| `B_random100`: tracks `B_expander100` | 0.988, 4/4 | ✅ |
| **Viability** (B ≥ 95% of A_full, ≥3/4 seeds) | 98.6% of full, 4/4 | ✅ |
| **Structure-decides-regime** (B reliable, C lottery) | both reliable; structure decides *speed* (~7×) | ❌→ refined |
| **Expander = random** (not Cayley-specific) | within ~1.5 pts, both 4/4 | ✅ |

## Does the 100-edge graph give equal results?

> **Yes — a fixed ~100-edge-per-token graph, ~7–13× cheaper than full attention, matches full-attention
> recall on a dependency that provably needs the reach, so long as the edges are *spread* to keep every
> route inside the 2-hop regime. At identical budget, whether the spread is algebraic or random does not
> matter (construction null, fifth confirmation), but whether the edges are spread or bunched matters a
> great deal — not for *whether* the model solves the task (a redundant contiguous window of the same
> budget also solves it, via thousands of 3-hop paths) but for *how fast*: the 2-hop spread graph learns
> as quickly as full attention (~1,750 steps), the 3-hop bunched window ~7× slower. This is the
> narrowed, surviving form of "a matrix built from a graph": the graph does generate effective
> full-attention connectivity at a large edge saving, and the lever is geometric reach (keep routes
> short), not the algebraic identity of the construction. Phase 06's apparent 3-hop "lottery" is shown
> to be a property of single-path routes, not of depth — depth costs optimization time, and only sparse
> (low-multiplicity) deep routes become unreliable.**

## Caveats

- **No arm failed, so the "structure-decides" claim rests on speed, not success.** At this budget and
  distance both regimes solve; the 7× gap is the cleanest signal. A larger D, a *narrower* window
  (pushing the window route to 4+ hops where P06 says it walls), or a *sparse* (low-multiplicity) 3-hop
  control would re-expose a success/fail separation. The single-path-vs-redundant mechanism is inferred
  from the P06↔P07 contrast, not isolated within one phase.
- **n=4 seeds.** Directionally robust (the 1,750-vs-12,250 gap is large and consistent), but the speed
  distributions are coarse and recall differences among A/B arms (~1.5 pts) are within seed noise.
- **Synthetic, single needle.** One clean dependency at a controlled distance. Real language routes
  many dependencies of mixed distance at once — the honest transfer test, where a spread graph must
  cover a *range* rather than one D, is phase 08.
- **Degree-100 ladder not run** (optional secondary): recall/speed vs degree d∈{16,32,64,100} to locate
  where 2-hop coverage of D=256 first appears (≈ d=√1320≈36) is the natural follow-up figure.
- Quality/recall + steps-to-solve only; no wall-clock compute-payoff claim (masked-dense SDPA is slower
  than flash full attention; the saving is in edge count / would require a kernel that skips masked
  entries to realize).

## File manifest

- `phase07_records.json` — config (T, D, degree, budget, ~saving×), reach-guard (per-arm and per-seed
  hop distances, direct-edge flags), sanity gate, per-(arm, seed) recall / steps-to-solve / value-CE /
  stop reason / edges / curve / peak memory / wall, aggregated summary.
- `phase07_recall.png` — recall @ D=256 by arm (per-seed points + median), seeds-solving annotations,
  chance line.
- `phase07_README.md` — this file. (Verbose run log gitignored.)
- `../../experiments/phase07_fixed_budget.py` — the experiment (`P7_GUARD_ONLY=1` re-checks the reach
  guard; `P7_SMOKE=1` end-to-end shakeout; resumable).

## Open questions → phase 08 (if pursued)

1. **Real-text transfer.** A small PG-19 / Gutenberg slice with *verified* long arcs, same 100-edge
   spread budget: does the spread graph still match full attention when many mixed-distance
   dependencies must route at once, rather than one clean needle? This is the honest next step now that
   the mechanism is settled.
2. **Re-expose success/fail.** Narrow the window (route → 4+ hops, the P06 wall) or add a *sparse*
   single-path 3-hop control, to convert the 7× speed gap back into a reliability separation and
   isolate the path-multiplicity mechanism within one phase.
3. **Degree ladder.** Recall and steps-to-solve vs degree d∈{16,32,64,100}; locate the crossover where
   2-hop coverage of D=256 first appears (≈ √T) and watch the speed penalty switch on as the route
   lengthens.
