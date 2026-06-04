# Phase 05 — Hop-count ablation: how deep a clean route will the model actually traverse

**Given a *clean, uniform* multi-hop route to a supra-window needle, composition does happen — but only
to a shallow depth. Two hops are traversed reliably (recall ≈ 1.00, both seeds); three hops are a
knife-edge (one seed 0.91, one seed at chance); four hops collapse to chance for both seeds, well below
the 8-layer ceiling. A scattered edge set with the *same long-edge count* as the 3-hop arm but no clean
chain stays at chance — so it is the clean *path*, not the edge *count*, that is load-bearing.** This
phase answers Phase 04's open question #2 ("does composition ever happen?") and corrects Phase 04's
reading: "reachable isn't reached" was a confound, not a law. When the route is clean and uniform,
reachable *is* reached up to depth ~2; Phase 04's needle failed because its scattered edges left no
clean chain — exactly the `scattered_k3` arm here, which is at chance for the same reason.

## Task

Identical model, task, and training to Phase 04 (25.2M-param decoder, 8 layer × 8 head × 512, RoPE,
weight-tied; procedurally-generated MQAR associative recall scored only at the query-key position,
chance = 1/V = 1.56%). The needle's value sits **exactly D=768 tokens before the query** (fixed,
supra-window; query position randomized per sequence). **Only the attention mask changes between arms.**

The discriminator is *required hop-count*, isolated by construction. Each `kN` arm uses a single
**uniform dilated stride** `s = D/k`, so from every query node the needle at D=768 is reachable in
exactly `k` hops and — because every `s > window` (192 > 128) — in no fewer (max single-hop reach is
`s`, and D is an exact multiple of `s`). The stride is the *only* long-range edge; the local window
w=128 is held fixed:

| Arm | k | stride s = 768/k | s > window? | role |
|-----|:-:|:----------------:|:-----------:|------|
| `k1_direct` | 1 | 768 | yes | positive control (reproduces Phase 04 `B_dilated`) |
| `k2` | 2 | 384 | yes | 2-hop composition |
| `k3` | 3 | 256 | yes | 3-hop composition |
| `k4` | 4 | 192 | yes | 4-hop composition |
| `window_only` | — | (no long edge) | — | floor (D=768 supra-window → chance) |
| `scattered_k3` | ~3 | random offsets, src ≥ 256 | — | **same long-edge count as `k3`, no clean chain** |

**Path guard (ran before any training).** For 512 sampled queries per arm: a clean k-hop path to the
needle exists (`reach_frac == 1.00`) *and* the measured shortest-path length is exactly k
(`shortest_med == shortest_min == shortest_max == k`) for k1…k4. `scattered_k3` matches the edge count
but its shortest path ranges 2–3 with no guaranteed chain. Because the construction certifies the path
length, a cliff at a given k is a genuine **routing-depth** limit, not a missing-path artifact — and k
tops out at 4 while the model has 8 layers, so the cliff is not a depth-ceiling artifact either.

## Headline table (recall accuracy; chance = 0.0156; n=2 seeds, 1337/1338)

| Arm | hops k | recall (mean) | per-seed (1337 / 1338) | value-CE | verdict |
|-----|:------:|:-------------:|:----------------------:|:--------:|---------|
| `k1_direct` | 1 | **1.000** | 1.000 / 1.000 | 0.000 | direct edge, one hop — control holds |
| `k2` | 2 | **0.996** | 0.992 / 1.000 | 0.059 | **2-hop composition reliable** |
| `k3` | 3 | 0.463 *(bimodal)* | **0.905 / 0.021** | 2.283 | **knife-edge — one seed solves, one at chance** |
| `k4` | 4 | **0.021** ≈ chance | 0.021 / 0.021 | 4.160 | collapse, both seeds |
| `window_only` | — | 0.020 ≈ chance | 0.020 / 0.021 | 4.160 | floor confirmed |
| `scattered_k3` | ~3, no chain | 0.021 ≈ chance | 0.022 / 0.021 | 4.162 | **same edge count as `k3`, no clean path → chance** |

Edge budgets matched among the long-range arms (124.1k–124.7k; window-only 123.8k). Trained to a recall
plateau (< 0.5% over a 2k-step window, min 3k steps); arms that learned plateaued early, arms that
stayed at chance ran to the 9k-step fail-cutoff (~48 min) — **failures were trained to convergence, not
undertrained.** The one `k3` success needed the *full* 12k steps with a late, unstable onset (S-curve
starting only at ~6k, recall reaching 0.90 by 11k, then a single eval crashing to 0.22 at step 11500
before recovering) — composition at the edge of trainability looks exactly like this. ~8.6 GPU-hours,
12 runs.

## What the result shows

1. **Composition *does* happen — Phase 04's "no multi-hop routing" was the absence of a clean path, not
   a property of the model.** With a clean uniform stride, `k2` reaches recall ≈ 1.00 on both seeds:
   the model relays the dependency window → dilated edge → window across two hops without a direct edge.
   Phase 04 saw chance for every non-direct arm because its scattered/Cayley/random offsets left no
   clean chain — the `scattered_k3` arm reproduces that here (chance, both seeds) at the *same edge
   count* as the `k3` arm that succeeds. So the load-bearing thing is the **clean path**, not the edge
   count and not the construction.

2. **But the depth is shallow and the ceiling is sharp.** Reliable at 2 hops, knife-edge at 3, dead at
   4 — far below the 8 available layers. The `k3` arm is **bimodal**: 0.905 for seed 1337, chance for
   seed 1338. The 0.46 "mean" is not a plateau, it is a coin flip between solve and fail; reporting it
   as a midpoint would misrepresent a bistable outcome. `k4` is unambiguously at chance for both seeds.

3. **The cliff is a routing-depth limit, certified by the path guard.** Every kN arm has `reach_frac =
   1.00` and shortest-path length exactly k, so `k4`'s failure is not "no path existed" (Phase 04's
   confound) — the path provably exists, the model just won't traverse four hops of it. With k=4 < 8
   layers it is also not a depth-ceiling artifact.

4. **`window_only` and `scattered_k3` pin the floor.** Window-only sits at chance (0.020) — D=768 is
   supra-window, confirming the task still loads the scaffold. `scattered_k3` (chance, both seeds)
   isolates *path cleanliness* from *edge count*: equip the same number of long edges but break the
   guaranteed chain and recall returns to the floor.

## Mechanism

Recall as a function of certified hop-count is a **shallow step with a fragile riser**:

| k (certified hops) | recall | regime |
|:------------------:|:------:|--------|
| 1 | 1.00 / 1.00 | direct — trivially solved at the 3k-step minimum |
| 2 | 0.99 / 1.00 | composed — reliable, plateau by 6–8.5k steps |
| 3 | 0.91 / chance | composed *or* failed — seed-dependent, only at the 12k-step limit when it works |
| 4 | chance / chance | not composed |

The Phase-04 mechanism ("recall tracks one-hop reach as a step function") generalizes: recall tracks
*certified shortest-path depth*, and the model's effective routing depth on this task is **~2 hops,
fraying at 3, exhausted by 4**. The riser between "reliable" and "dead" is one hop wide and lands on a
seed-sensitive boundary — composition at depth 3 is learnable but not robustly so at this width/depth/
curriculum. The clean-path requirement is sharp in the other direction: `scattered_k3` has the budget
to compose but no certified chain, and it never leaves the floor.

## Relation to prior phases

- **Phase 04** concluded "edge placement is load-bearing, the construction is not," and noted as a
  caveat that 8 layers "do not learn the multi-hop composition" — but flagged the confound that no
  clean multi-hop path to its randomized needle may have existed. **Phase 05 removes that confound and
  splits the caveat in two:** composition *is* learned (to depth 2 reliably), so "no multi-hop routing"
  was too strong; *and* there is a hard, shallow depth ceiling, so placement still matters for anything
  past ~2 hops. The corrected statement: **a clean route is traversed up to ~2 hops; beyond that an edge
  must bridge the dependency directly.**
- **Construction null (phases 01–04) is untouched.** Phase 05 varies hop-count under a deliberately
  *structure-free* dilated stride; it says nothing for the Cayley graph except to remove its last
  excuse — "maybe the model just never composes" is false, so the Cayley graph's Phase-04 failure was
  its offsets not landing on (or chaining to) the dependency, not a blanket inability to route.

## Pre-committed predictions vs measured

The phase pre-registered: *"a sharp recall collapse by k=2 or k=3 (recall(k=2) < 0.30, k3/k4 ~ chance),
i.e. effective routing depth ~1 hop, far below the 8-layer ceiling."*

| Prediction (05) | Measured | Verdict |
|-----------------|----------|:-------:|
| `k1_direct` reproduces Phase 04 `B_dilated` (≈ 1.00) | 1.000 | ✅ |
| recall(k=2) < 0.30 (collapse already by 2 hops) | 0.996 | ❌ (composes reliably) |
| k3 ~ chance | bimodal 0.905 / 0.021 | ◐ (chance for one seed, solved by the other) |
| k4 ~ chance | 0.021 | ✅ |
| effective routing depth ~1 hop | ~2 reliable, 3 fragile | ❌ (deeper than predicted) |
| sharp collapse far below the 8-layer ceiling | collapse at k=3→4, ≪ 8 | ✅ |
| `window_only` at chance (floor) | 0.020 | ✅ |
| `scattered_k3` at chance (clean path, not count, matters) | 0.021 | ✅ |

**Net: the qualitative shape was right (a sharp sub-ceiling collapse) but the depth was underestimated
— the model composes two hops cleanly, not one, and frays at three rather than two.** The pre-committed
"≤ 1 hop" floor is falsified upward.

## Architectural interpretation

> **A sparse attention model will traverse a clean multi-hop route, but only to a shallow depth — here
> ~2 hops reliably, 3 on a seed-dependent knife-edge, 4 not at all, despite 8 layers and a
> construction-certified path at every depth.** Composition is real (correcting Phase 04's "no
> multi-hop routing"), and it is gated by *path cleanliness*, not edge count: a scattered edge set with
> the same budget but no guaranteed chain stays at chance. So the load-bearing properties of a sparse
> graph, in order, are: (1) a clean path to the dependency must exist, and (2) that path must be short —
> ≤ ~2 hops — or an edge must bridge the dependency directly. Algebraic construction enters nowhere;
> the routing limit is about depth and path quality, the same axis the whole repo keeps landing on.

## Caveats

- **`k3` bimodality is two seeds.** "Knife-edge at 3 hops" is the honest reading of {0.905, 0.021}, but
  the success/fail split is estimated from n=2; more seeds would turn the coin-flip into a probability.
  The *direction* (k2 reliable, k4 dead, k3 in between) is robust; the exact width of the riser is not.
- **Effective depth ~2 is for this depth/width/task/curriculum.** 8×512 on MQAR with a single uniform
  stride. A different model size, a curriculum from k1→k4, or a multi-edge route whose *combinations*
  span D might push the ceiling; the claim is what happens *by default*, not a provable bound.
- **Single distance, single stride family.** D=768 with strides {768, 384, 256, 192}. The ceiling is
  read along hop-count at fixed D; it does not separate "k hops" from "stride length" (longer strides
  = fewer hops here by construction). A 2-D sweep (D × stride) would disentangle them.
- Quality/recall only; no compute-payoff claim (masked-dense SDPA is slower than flash full attention).

## File manifest

- `phase05_records.json` — per-(arm, seed) recall / final-recall / value-CE, steps-to-plateau,
  fail-cutoff, edge counts, full recall curves, peak memory, wall times; pre-training path-guard report
  and aggregated summary.
- `phase05_recall_vs_hops.png` — recall vs required hop-count k (per-seed points + mean), with the
  window-only and scattered_k3 floors and the chance line.
- `hop_count_README.md` — this file. (Verbose run log is gitignored.)
- `../../experiments/phase05_hop_count.py` — the experiment (run with `P5_GUARD_ONLY=1` to re-check the
  path guard without training; resumable from `phase05_records.json`).

## Open questions → phase 06 (if pursued)

1. **Probability of the 3-hop riser.** Run `k3` (and `k2.5`/`k3.5` via non-integer-friendly strides)
   across 8–16 seeds to turn the bimodal knife-edge into a success-rate curve and locate the 50% point.
2. **Does a curriculum raise the ceiling?** Train k1→k2→k3→k4 in sequence (or anneal D) to test whether
   the depth limit is a trainability artifact rather than a representational one — the one mechanism
   that could let a structured multi-edge graph route deep dependencies.
3. **Multi-edge routes whose *combinations* span D.** The regime where a broadband expander could beat a
   single dilation — give the model two stride families whose sum reaches D and see if it chains *across
   edge types*, which is closer to what a real Cayley/BigBird graph offers than a single uniform stride.
