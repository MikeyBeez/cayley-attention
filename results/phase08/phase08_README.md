# Phase 08 — The 100-edge spread graph still matches full attention through a *real-language* haystack — and real text is where the bunched window finally cracks

**Phase 07's viability transfers to real language: a 100-edge SPREAD graph (2-hop reach to D=256) still
matches full attention on a recall needle planted in a real WikiText-103 haystack (B_expander 0.960,
B_random 0.979 vs A_full 0.981, all 4/4 seeds, ~13× edge saving), construction-agnostic for a sixth
time. The new result is the BUNCHED control: the same-budget contiguous window — which in synthetic
Phase 07 solved 4/4, merely ~7× slower — now solves only 2/4 through real text, and both solvers crawl
to the 40k ceiling (steps-to-solve 35k/38k) while the two non-solvers are still climbing when budget
runs out (0.70, 0.28). Real-language relay nodes act as a depth amplifier: they barely touch the 2-hop
spread route (~1.6× slower, still 4/4) but inflate the 3-hop window route past a generous 40k budget
for half its seeds. This is the first phase where bunched-vs-spread separates on *success*, not just
speed — and it is the real text, loading the relay nodes, that does it.**

This is the honest real-text step. Phases 01–03b showed plain LM on real text is an uninterpretable
null (a w=128 window already matches full attention on WikiText, so long-range edges carry no load).
So Phase 08 keeps the dependency **controlled** — a recall needle provably D=256 past the window,
identical reach guard to Phase 07 — and changes exactly **one variable**: the ~1,200 filler positions
the route traverses become real WikiText-103 tokens instead of synthetic PAD.

## Setup

Identical to Phase 07 (T=1320, D=256, L=6, four arms at a fixed 100-edge budget, reach guard proving
`C_window100`=3 hops / `B_expander100`,`B_random100`=2 hops / no direct 256-edge, 40k ceiling,
stop-on-solve, flat-stop OFF) **except** the haystack: every sequence is a real WikiText-103 window
(GPT-2 BPE, ids 0–50256), with the recall needle and 48 reserved distractor key→value pairs
(markers at ids ≥ 50257) overwriting the text. The model must still key-match the query to the needle
value among 48 distractors — now through ~1,200 tokens of natural language instead of PAD. Vocab 50,386
(44.7M params); the head is applied only at the query position (the full T×50k logits would OOM).
Sanity gate (D_in=8 through real text): all arms ≥ 0.99 — task and training are sound, so D=256 effects
are reach/regime, not training failure. ~20.6 GPU-hours, 20 cells.

## Headline table (recall @ D=256 through a WikiText haystack; chance = 0.0156; n=4 seeds)

| Arm | hops→256 | seeds solving | recall (median) | steps-to-solve (median) | vs Phase 07 (synthetic) |
|-----|:--------:|:-------------:|:---------------:|:-----------------------:|-------------------------|
| `A_full` | 1 | **4/4** | 0.981 | 1,250 | ≈ (was 4/4, 1,750) |
| `B_expander100` (spread) | 2 | **4/4** | 0.960 | 2,750 | ≈, mild ~1.6× slow (was 4/4, 1,750) |
| `B_random100` (spread) | 2 | **4/4** | 0.979 | 3,250 | ≈ (was 4/4, 2,250) |
| `C_window100` (bunched) | 3 | **2/4** | 0.814 | 36,500 | **broke** (was 4/4, 12,250) |

Per-seed steps-to-solve: `A_full` [1000, 500, 1500, 11000]; `B_expander100` [3000, 2500, 1500, 12000];
`B_random100` [2000, 4000, 5500, 2500]; `C_window100` [38000, 35000, **None**, **None**] — the two
non-solvers reached only 0.70 (s1339, still oscillating upward at step 40k) and 0.28 (s1340, grinding).
Seed 1340 is hard for every arm (full and expander both needed ~11–12k), but full/expander/random all
*solve* it while the window does not.

## What the result shows

1. **Viability transfers to real language.** `B_expander100` (0.960) and `B_random100` (0.979) reach
   ~98–100% of `A_full`'s 0.981 recall, all four seeds, with the dependency routed through real
   WikiText. A 100-edge spread graph (~6.6× fewer edges on average, ~13× at the query end) matches full
   attention even when the relay nodes hold natural language — Phase 07 was not a synthetic artifact.

2. **Construction null, a sixth time.** `B_random100 ≈ B_expander100 ≈ A_full`. The *spread* property,
   not the algebraic identity of the construction, is what keeps the route at 2 hops and matches full
   attention — now confirmed against a real-text haystack.

3. **Real text is where bunched-vs-spread separates on success.** Phase 07's contiguous window solved
   all 4 seeds (the spread arms were only *faster*). Through real text the same-budget window solves
   only **2/4**, and even those crawl to ~35–38k steps — against the spread arms' ~2.5–3.5k. The
   bunched 3-hop route, harmless in synthetic noise, becomes unreliable through natural language at a
   fixed budget.

4. **The mechanism: real text amplifies the depth cost.** The slowdown is concentrated entirely on the
   *extra hop*. From synthetic (P07) to real text (P08): the 2-hop spread route slows ~1.6× (1,750 →
   2,750) and stays 4/4; the 3-hop window route slows ~3× (12k → 36.5k) *and* drops to 2/4. Real-
   language relay nodes carry their own structure that the routing must see past, and each additional
   composition hop pays that tax again — so the third hop is where a generous 40k budget runs out.

## Mechanism

Combining Phases 06–08, two knobs and an amplifier:

| factor | controls | evidence |
|--------|----------|----------|
| composition depth (hops) | optimization *cost* | 1–2 hops cheap; 3 hops ~7× (P07) → ~3× more under real text |
| path multiplicity | *reliability* | single-path 3-hop = P06 lottery; redundant 3-hop = reliable (P07) |
| **real-language relay context** | **multiplies the per-hop cost** | 2-hop barely moves (1.6×, 4/4); 3-hop blows past the budget (2/4) |

The two window non-solvers were still *ascending* at the ceiling (0.70 noisily climbing; 0.28 grinding),
not dead-flat — so this is a severe **slowdown into a budget wall**, not a Phase-06 init lottery. Real
text doesn't make the third hop unlearnable; it makes it cost more than 40k steps for half the seeds.
The practical lesson sharpens: **spread your edges to keep every route at 2 hops and a 100-edge graph
matches full attention through real language at ~13× saving; bunch them and the third hop, traversing
natural-language relay nodes, becomes unreliable within any reasonable budget.**

## Pre-committed hypotheses vs measured

(Phase 08 was designed here from Phase 07's open question; the hypotheses are the design's stated bets.)

| Hypothesis | Measured | Verdict |
|------------|----------|:-------:|
| Sanity (D_in=8 through real text): all arms ~1.00 | 0.99–1.00 | ✅ |
| **Viability transfers**: spread arms ≥ 95% of A_full, ≥3/4 seeds | 98–100%, 4/4 | ✅ |
| Construction null persists (random ≈ expander) | within ~2 pts, both 4/4 | ✅ (6th) |
| Open: does the 3-hop window slow or break under real text? | **breaks** — 2/4, solvers at ceiling | resolved: breaks |
| (P07 carryover) bunched window reliable-but-slow | no longer reliable through real text | refined |

## Does the 100-edge graph give equal results — on real text?

> **Yes — a fixed ~100-edge-per-token graph, ~7–13× cheaper than full attention, matches full-attention
> recall on a dependency planted in a real WikiText-103 haystack, provided the edges are spread to keep
> every route at 2 hops; whether the spread is algebraic or random does not matter (sixth construction
> null). The control makes the lever unmistakable: the *same* 100 edges bunched into a contiguous window
> — a 3-hop route — solved every seed in synthetic noise but only half of them through real language,
> and those at the very edge of a 40k budget. Real-language relay nodes act as a depth amplifier: they
> are nearly free to traverse once (2-hop spread, ~1.6× slower, still reliable) but expensive to
> traverse twice (3-hop window, ~3× slower and unreliable). The narrowed, surviving claim of "a matrix
> built from a graph" therefore holds against real text: the graph generates effective full-attention
> connectivity at a large edge saving, and the binding constraint is geometric reach — keep routes
> short — not the construction, and not the realism of the context.**

## Caveats

- **The needle is controlled; the haystack is real but task-irrelevant.** Real WikiText fills the route
  but does not *carry* the answer (the dependency is the planted key→value pair). This isolates "does
  real-language relay context disrupt routing" cleanly, but it is not yet natural long-range *language
  modeling*, where the dependencies themselves are linguistic. That is phase 09 — and it is now earned:
  the spread graph survives a real-text haystack, so the fully-natural test is worth its cost.
- **The window "break" is a budget wall, not a hard wall.** Both non-solvers were still climbing at 40k;
  with far more steps they might solve. The honest statement is "real text inflates the 3-hop cost past
  a generous budget for half the seeds," not "the 3-hop route is impossible."
- **n=4 seeds**; the 4/4-vs-2/4 split and the ~10× steps-to-solve gap are large and consistent, but
  rates are coarse. Seed 1340 is hard across all arms (a shared optimization-difficulty draw), yet only
  the window fails it.
- Recall + steps-to-solve only; the edge saving is in count, not wall-clock (masked-dense SDPA is
  slower than flash full attention; realizing the saving needs a kernel that skips masked entries).

## File manifest

- `phase08_records.json` — config (incl. dataset, vocab, ~saving×), reach guard (per-arm/per-seed),
  sanity gate, per-(arm, seed) recall / steps-to-solve / value-CE / stop reason / curve / peak mem /
  wall, aggregated summary.
- `phase08_recall.png` — recall @ D=256 through the WikiText haystack by arm (per-seed + median),
  seeds-solving annotations, chance line.
- `phase08_README.md` — this file. (Verbose run log gitignored.)
- `../../experiments/phase08_realtext_haystack.py` — the experiment (`P8_GUARD_ONLY=1` reach guard;
  `P8_SMOKE=1` shakeout; resumable).

## Open questions → phase 09 (if pursued)

1. **Fully-natural long-range LM.** Now that the spread graph survives a real-text *haystack*, test it
   on real long-range *dependencies*: a PG-19 / Gutenberg slice with verified long arcs at the same
   100-edge spread budget, where the answer is a real linguistic continuation rather than a planted
   value. The honest hard test — and only worth the cost because Phase 08 passed.
2. **Give the window more budget.** Run `C_window100`'s two non-solvers to 100k+ to confirm the 3-hop
   route is a slowdown (eventually solvable) rather than a true wall under real text.
3. **Why does real text tax the third hop specifically?** Inspect attention at the relay nodes — does
   real-language content at the intermediate hop compete with the relay signal (an interference the
   2-hop route avoids by having one fewer relay)?
