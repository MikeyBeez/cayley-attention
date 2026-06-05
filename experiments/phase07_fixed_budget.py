"""
Phase 07 — Does a fixed ~100-edge graph match full attention once every route is inside the
            reliable 2-hop regime? (≈12× edge saving vs full attention at T=1320)

Phase 06 fixed the binding constraint: routing is RELIABLE at 2 hops, a ~42% init LOTTERY at 3, a
WALL at 4 — a reliability ladder, not a cost curve. So a sparse graph can stand in for full attention
only when the routes it needs are ≤2 hops. The earlier graphs failed at D=256 because the faithful
Cayley graph put the needle a median of 4 hops away — past the wall.

This phase tests the payoff of staying inside the budget. Give every token ~100 backward edges (vs the
full ~1320 — a ~12× saving), SPREAD so the whole sequence sits within 2 hops, and ask: does the sparse
construction now match full attention on the same D=256 needle it used to miss? The sharp control: at
the SAME 100-edge budget, a CONTIGUOUS window reaches ~100/hop → needs 3 hops for 256 (lottery), while
a SPREAD set needs ≤2 (reliable). Identical cost — only the offset structure differs.

  arm             100-edge offset set                 hops→256   predicted regime
  A_full          full causal attention (ceiling)     1 (direct) —
  C_window100     contiguous {1..100}                 3          lottery (P06 3-hop)
  B_expander100   deterministic SPREAD offsets        ≤2         reliable
  B_random100     random spread offsets (per seed)    ≤2         reliable

C_window100 and B_expander100 carry the IDENTICAL edge budget and differ only in whether the offsets
are bunched or spread. That is the whole experiment.

NOTE on "expander": a literal degree-100 SL(2,Z_n) Cayley graph is ill-defined (the group construction
is fixed low degree, and its scatter offsets can't be tuned to avoid a direct 256-edge or guarantee a
2-hop bound). Phases 01–06 already nulled Cayley-vs-random four times, so the faithful realization of
THIS phase's question (bunched vs spread at identical budget) is three 100-offset sets differing only
in WHICH offsets. B_expander100 is a deterministic spread set; B_random100 is the matched random
control. The reach guard PROVES the ≤2 / =3 / no-direct-edge properties rather than asserting them.

Budget / stopping (closes P06 open-question #1): ceiling 40k steps, stop-on-solve (recall ≥ 0.90
sustained), and NO flat-stop — every dead seed runs to the full 40k before being called a non-solver,
so "dead-flat" is observed through 40k, not inferred from a 10k flatness gate.

Model / task / generator identical to Phase 04–06 except N_LAYER=6 and T=1320.

Run:  ~/Code/HRS/.venv/bin/python experiments/phase07_fixed_budget.py
Knobs: P7_CEILING / P7_EVAL_EVERY / P7_EVAL_SEQS / P7_SEEDS ; P7_GUARD_ONLY=1 runs the reach guard and
       exits; P7_SMOKE=1 tiny end-to-end; P7_SKIP_SANITY=1 skips the D_in=8 gate (debug only).
Resumable: completed (phase, arm, seed) runs in phase07_records.json are skipped.
"""
import os, sys, json, time, math, collections
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.environ.get("P7_RESULT_DIR") or os.path.join(REPO, "results", "phase07")
JSON_PATH  = os.path.join(RESULT_DIR, "phase07_records.json")
PLOT_PATH  = os.path.join(RESULT_DIR, "phase07_recall.png")
os.makedirs(RESULT_DIR, exist_ok=True)

# ---- task vocab (identical to Phase 04–06) ----
K, V = 64, 64
PAD, QUERY = 0, 1
KEY0 = 2
VAL0 = 2 + K
VOCAB = 2 + K + V          # 130
CHANCE_CE = math.log(V)    # 4.1589

# ---- geometry ----
T = 1320                   # = |SL(2,Z_11)|, the Phase-02/07 long-context length
D = 256                    # the distance the degree-4/sparse graphs failed on
D_IN = 8                   # sanity: needle inside reach of every arm (all sets include offset 8)
DEGREE = 100               # edge budget per token (vs full ~1320 → ~12× saving)

# ---- model (identical to Phase 04–06 except depth) ----
N_LAYER, N_HEAD, N_EMBD, DROPOUT = 6, 8, 512, 0.0   # L=6, comfortably above the 2-hop need

def _envint(name, default):
    v = os.environ.get(name); return int(v) if v else default
def _envflt(name, default):
    v = os.environ.get(name); return float(v) if v else default

SMOKE        = os.environ.get("P7_SMOKE") == "1"
BATCH        = 64
LR           = 1e-3
WARMUP       = 200
WEIGHT_DECAY = 0.1
BETA1, BETA2 = 0.9, 0.95
GRAD_CLIP    = 1.0
EVAL_EVERY   = _envint("P7_EVAL_EVERY", 500)
EVAL_SEQS    = _envint("P7_EVAL_SEQS", 2048)
EVAL_SEED    = 9999
MIN_STEPS    = _envint("P7_MIN_STEPS", 3000)
CEILING      = _envint("P7_CEILING", 200 if SMOKE else 40000)   # true 40k, flat-stop OFF
SANITY_CEIL  = _envint("P7_SANITY_CEIL", 200 if SMOKE else 8000)
SOLVE_RECALL = _envflt("P7_SOLVE_RECALL", 0.90)
SEEDS        = [int(x) for x in os.environ.get("P7_SEEDS", "1337,1338,1339,1340").split(",")]
GUARD_ONLY   = os.environ.get("P7_GUARD_ONLY") == "1"
SKIP_SANITY  = os.environ.get("P7_SKIP_SANITY") == "1"
RANDOM_OFFSET_SEED = 20260605

ARMS = ["A_full", "C_window100", "B_expander100", "B_random100"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

# --------------------------------------------------------------------------------------
# Offset sets — the manipulation. All three structured arms have exactly DEGREE offsets.
# --------------------------------------------------------------------------------------
def window_offsets():
    return list(range(1, DEGREE + 1))                      # {1..100} contiguous → 3 hops to 256

def expander_offsets():
    """Deterministic SPREAD set: small offsets {1..8} (local + D_in reach) plus offsets evenly
    spread over (8, D] so that (a) D itself is excluded (no direct edge), (b) many pairs sum to D
    (2-hop reach), (c) the set reaches far. Verified by the reach guard."""
    small = list(range(1, 9))                              # 8 small (includes D_in=8)
    n_spread = DEGREE - len(small)                         # 92 spread
    # evenly spread over (8, D-3]; excludes D, keeps complements in-range
    spread = sorted(set(int(round(8 + (D - 11) * k / (n_spread))) for k in range(1, n_spread + 1)))
    offs = sorted(set(small + spread))
    # top up to DEGREE with nearby unused offsets if dedup shrank it
    o = D - 4
    while len(offs) < DEGREE:
        if o not in offs and o != D:
            offs.append(o)
        o -= 1
    offs = sorted(set(offs))[:DEGREE]
    return offs

def random_offsets(seed):
    """Random spread control: {1..8} anchor (D_in reach, matched to expander) + 92 random distinct
    offsets from (8, D]. Spread by randomness → 2-hop reach to D (guard-verified per seed)."""
    rng = np.random.default_rng(RANDOM_OFFSET_SEED + seed)
    small = list(range(1, 9))
    pool = [x for x in range(9, D + 1) if x != D]          # exclude D itself (no direct edge)
    extra = rng.choice(pool, size=DEGREE - len(small), replace=False).tolist()
    return sorted(set(small + [int(x) for x in extra]))

def offsets_for(arm, seed):
    if arm == "A_full":         return None                # full causal
    if arm == "C_window100":    return window_offsets()
    if arm == "B_expander100":  return expander_offsets()
    if arm == "B_random100":    return random_offsets(seed)
    raise ValueError(arm)

def build_mask(arm, seed):
    """Additive (-inf) mask. A_full = causal; others = self + {i-o : o in offsets}, causal."""
    allow = np.zeros((T, T), dtype=bool)
    r = np.arange(T)
    if arm == "A_full":
        allow = np.tril(np.ones((T, T), dtype=bool))
    else:
        allow[r, r] = True
        for o in offsets_for(arm, seed):
            idx = np.arange(o, T)
            allow[idx, idx - o] = True
    edges = int(allow.sum())
    add = np.zeros((T, T), dtype=np.float32); add[~allow] = float("-inf")
    return torch.from_numpy(add)[None, None].to(DEVICE, DTYPE), edges

# --- reach guard (graph distance query→needle over the offset set) --------------------
INF = 1 << 30
def shortest_hops_offsets(q, target, offs):
    """Min #edges from q down to target using backward offset edges (i -> i-o)."""
    dist = np.full(q + 1, INF, dtype=np.int64); dist[q] = 0
    dq = collections.deque([q])
    while dq:
        i = dq.popleft(); di = dist[i]
        for o in offs:
            j = i - o
            if j >= 0 and di + 1 < dist[j]:
                dist[j] = di + 1; dq.append(j)
            if j < target:
                continue
    return int(dist[target])

def reach_guard(arm, seed, dist=D, n_samples=400):
    if arm == "A_full":
        return {"arm": arm, "direct": True, "hop_med": 1, "hop_min": 1, "hop_max": 1,
                "reach_frac": 1.0, "has_direct_edge": True, "n_offsets": None}
    offs = offsets_for(arm, seed)
    rng = np.random.default_rng(7 + seed)
    qs = rng.integers(dist + 1, T - 1, size=n_samples)
    hops = np.array([shortest_hops_offsets(int(q), int(q) - dist, offs) for q in qs])
    reach = hops[hops < INF]
    return {"arm": arm, "n_offsets": len(offs), "has_direct_edge": (dist in offs),
            "reach_frac": round(float((hops < INF).mean()), 4),
            "hop_min": int(reach.min()) if reach.size else None,
            "hop_med": int(np.median(reach)) if reach.size else None,
            "hop_max": int(reach.max()) if reach.size else None,
            "frac_le2": round(float((hops <= 2).mean()), 4),
            "min_offset": int(min(offs)), "max_offset": int(max(offs))}

# --------------------------------------------------------------------------------------
# Synthetic generator (identical to Phase 04–06)
# --------------------------------------------------------------------------------------
N_PAIRS = 48
def make_batch(B, Dd, rng):
    toks = np.zeros((B, T), dtype=np.int64)
    tpos = np.zeros(B, dtype=np.int64); ttok = np.zeros(B, dtype=np.int64)
    for b in range(B):
        qk = int(rng.integers(Dd + 1, T - 1))
        if qk % 2 == 0:
            qk = qk - 1 if qk - 1 >= Dd + 1 else qk + 1
        nk, nv = qk - Dd - 1, qk - Dd
        keyset = rng.permutation(K)[:N_PAIRS]
        kq = int(keyset[0]); distractor_keys = keyset[1:]
        vq = int(rng.integers(0, V))
        even = np.arange(0, qk - 1, 2); even = even[even != nk]
        npos = min(len(distractor_keys), len(even))
        slots = rng.choice(even, size=npos, replace=False)
        dvals = rng.integers(0, V, size=npos)
        toks[b, slots] = KEY0 + distractor_keys[:npos]
        toks[b, slots + 1] = VAL0 + dvals
        toks[b, nk] = KEY0 + kq; toks[b, nv] = VAL0 + vq
        toks[b, qk - 1] = QUERY; toks[b, qk] = KEY0 + kq; toks[b, qk + 1] = VAL0 + vq
        tpos[b] = qk; ttok[b] = VAL0 + vq
    return toks, tpos, ttok

def heldout(Dd):
    return make_batch(EVAL_SEQS, Dd, np.random.default_rng(EVAL_SEED + Dd))

# --------------------------------------------------------------------------------------
# Model (identical to Phase 04–06: RoPE, SDPA additive mask)
# --------------------------------------------------------------------------------------
def _build_rope(Tn, hd, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    freqs = torch.outer(torch.arange(Tn).float(), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()
_HD = N_EMBD // N_HEAD
_ROPE_COS, _ROPE_SIN = _build_rope(T, _HD)
def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1); return torch.cat((-x2, x1), dim=-1)

class Attention(nn.Module):
    def __init__(self, add_mask):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.nh = N_HEAD
        self.register_buffer("rope_cos", _ROPE_COS[None, None], persistent=False)
        self.register_buffer("rope_sin", _ROPE_SIN[None, None], persistent=False)
        self.register_buffer("add_mask", add_mask, persistent=False)
    def forward(self, x):
        B, Tn, C = x.shape
        q, k, v = self.c_attn(x).split(N_EMBD, dim=2); hd = C // self.nh
        q = q.view(B, Tn, self.nh, hd).transpose(1, 2)
        k = k.view(B, Tn, self.nh, hd).transpose(1, 2)
        v = v.view(B, Tn, self.nh, hd).transpose(1, 2)
        cos = self.rope_cos[..., :Tn, :].to(q.dtype); sin = self.rope_sin[..., :Tn, :].to(q.dtype)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.add_mask)
        return self.c_proj(y.transpose(1, 2).contiguous().view(B, Tn, C))

class Block(nn.Module):
    def __init__(self, add_mask):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD); self.attn = Attention(add_mask)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.fc = nn.Linear(N_EMBD, 4 * N_EMBD, bias=False)
        self.proj = nn.Linear(4 * N_EMBD, N_EMBD, bias=False)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.proj(F.gelu(self.fc(self.ln2(x))))
        return x

class GPT(nn.Module):
    def __init__(self, add_mask):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, N_EMBD)
        self.blocks = nn.ModuleList([Block(add_mask) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB, bias=False)
        self.tok_emb.weight = self.head.weight
        self.apply(self._init)
        for pn, p in self.named_parameters():
            if pn.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))
    def _init(self, m):
        if isinstance(m, nn.Linear):   nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx):
        x = self.tok_emb(idx)
        for blk in self.blocks: x = blk(x)
        return self.head(self.ln_f(x))
    def n_params(self): return sum(p.numel() for p in self.parameters())

def configure_opt(model):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                              {"params": nodecay, "weight_decay": 0.0}], lr=LR, betas=(BETA1, BETA2))
def get_lr(it): return LR * (it + 1) / WARMUP if it < WARMUP else LR

@torch.no_grad()
def eval_recall(model, ev):
    model.eval()
    toks, tpos, ttok = ev
    tt = torch.from_numpy(toks).to(DEVICE); tp = torch.from_numpy(tpos).to(DEVICE); tk = torch.from_numpy(ttok).to(DEVICE)
    correct = 0; ce_sum = 0.0; nseq = toks.shape[0]
    for s in range(0, nseq, 256):
        xb = tt[s:s+256]; pb = tp[s:s+256]; kb = tk[s:s+256]
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            logits = model(xb)
        idx = torch.arange(xb.size(0), device=DEVICE)
        lg = logits[idx, pb]
        correct += (lg.argmax(-1) == kb).sum().item()
        ce_sum += F.cross_entropy(lg.float(), kb, reduction="sum").item()
    model.train()
    return correct / nseq, ce_sum / nseq

def train_one(arm, seed, Dd, ceiling):
    mask, edges = build_mask(arm, seed)
    torch.manual_seed(seed)
    model = GPT(mask).to(DEVICE)
    opt = configure_opt(model)
    rng = np.random.default_rng(seed * 100003 + Dd)
    ev = heldout(Dd)
    curve = []; best = 0.0; t0 = time.time(); step = 0
    steps_to_solve = None; solve_cand = None; stop_reason = "ceiling"
    while step <= ceiling:
        if step % EVAL_EVERY == 0:
            rec, ce = eval_recall(model, ev)
            curve.append((step, round(rec, 4), round(ce, 4)))
            best = max(best, rec)
            print(f"  [{arm} s{seed} D{Dd}] step {step:6d}  recall {rec:.4f}  ce {ce:.3f}  best {best:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            if rec >= SOLVE_RECALL:
                if solve_cand is None: solve_cand = step
                else: steps_to_solve = solve_cand; stop_reason = "solved"; break
            else:
                solve_cand = None
        for g in opt.param_groups: g["lr"] = get_lr(step)
        toks, tpos, ttok = make_batch(BATCH, Dd, rng)
        xb = torch.from_numpy(toks).to(DEVICE); pb = torch.from_numpy(tpos).to(DEVICE); kb = torch.from_numpy(ttok).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            logits = model(xb)
        idx = torch.arange(BATCH, device=DEVICE)
        loss = F.cross_entropy(logits[idx, pb].float(), kb)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
        step += 1
    final_rec, final_ce = eval_recall(model, ev)
    best = max(best, final_rec)
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
    npar = model.n_params()
    del model, opt
    if DEVICE == "cuda": torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    return {"arm": arm, "seed": seed, "D": Dd, "solved": steps_to_solve is not None,
            "steps_to_solve": steps_to_solve, "recall": best, "final_recall": final_rec,
            "value_ce": final_ce, "stop_reason": stop_reason, "stop_step": step,
            "edges_total": edges, "edges_per_tok_approx": round(edges / T, 1),
            "curve": curve, "peak_gb": round(peak, 2), "params": int(npar),
            "wall_s": round(time.time() - t0, 1)}

# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def load_results():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f: return json.load(f)
    return {"config": {}, "guard": {}, "sanity": {}, "runs": {}, "summary": {}}
def save_results(res):
    with open(JSON_PATH, "w") as f: json.dump(res, f, indent=2)

def aggregate(res):
    summ = {}
    for arm in ARMS:
        rk = [res["runs"][f"D{D}::{arm}::s{s}"] for s in SEEDS if f"D{D}::{arm}::s{s}" in res["runs"]]
        if not rk: continue
        solved = [r for r in rk if r["solved"]]
        sts = sorted(r["steps_to_solve"] for r in solved)
        summ[arm] = {"n_seeds": len(rk), "n_solved": len(solved),
                     "frac_solving": round(len(solved) / len(rk), 4),
                     "recall_mean": round(float(np.mean([r["recall"] for r in rk])), 4),
                     "recall_median": round(float(np.median([r["recall"] for r in rk])), 4),
                     "steps_to_solve_median": int(np.median(sts)) if sts else None,
                     "recalls": [round(r["recall"], 4) for r in rk]}
    res["summary"] = summ

def main():
    # ---- reach guard ----
    guard = {arm: reach_guard(arm, SEEDS[0]) for arm in ARMS}
    # random arm: report per-seed reach (offsets differ by seed)
    guard["B_random100_per_seed"] = {f"s{s}": reach_guard("B_random100", s) for s in SEEDS}
    print("===== REACH GUARD (query→needle hop distance @ D=256) =====")
    for arm in ARMS:
        g = guard[arm]
        print(f"  {arm:14s} offsets={g['n_offsets']} direct_edge={g['has_direct_edge']} "
              f"reach={g['reach_frac']} hops=[{g.get('hop_min')},{g.get('hop_med')},{g.get('hop_max')}] "
              f"frac≤2={g.get('frac_le2')}")
    cw, be = guard["C_window100"], guard["B_expander100"]
    ok = (be["hop_max"] <= 2 and not be["has_direct_edge"] and be["reach_frac"] == 1.0
          and cw["hop_med"] == 3)
    rand_ok = all(guard["B_random100_per_seed"][f"s{s}"]["frac_le2"] == 1.0 for s in SEEDS)
    print(f"  GUARD: expander ≤2 & no-direct & reach=1.0 & window=3 -> {'OK' if ok else 'FAIL'}; "
          f"random all-seeds ≤2 -> {'OK' if rand_ok else 'WARN'}")
    if not ok:
        print("[abort] reach guard failed — offset sets do not realize the intended regimes."); sys.exit(1)
    if GUARD_ONLY:
        print("[guard-only] done."); return

    res = load_results()
    res["config"] = {"task": "MQAR recall — fixed 100-edge budget, spread vs bunched",
        "T": T, "D": D, "D_in": D_IN, "degree": DEGREE, "vocab": VOCAB, "V": V,
        "chance_recall": round(1 / V, 4), "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD,
        "batch": BATCH, "lr": LR, "ceiling": CEILING, "solve_recall": SOLVE_RECALL,
        "flat_stop": False, "seeds": SEEDS, "arms": ARMS, "device": DEVICE, "dtype": str(DTYPE),
        "full_edges": T * (T + 1) // 2, "saving_x": round((T * (T + 1) // 2) / (T * DEGREE), 1)}
    res["guard"] = guard
    save_results(res)

    # ---- sanity gate (D_in=8): every arm, 1 seed, must hit ~1.0 ----
    if not SKIP_SANITY:
        print("\n===== SANITY GATE (D_in=8; every arm must hit ~1.00) =====")
        for arm in ARMS:
            sk = f"Din{D_IN}::{arm}::s{SEEDS[0]}"
            if sk in res["sanity"]:
                print(f"  [skip] {sk} recall={res['sanity'][sk]['recall']:.3f}"); continue
            out = train_one(arm, SEEDS[0], D_IN, SANITY_CEIL)
            res["sanity"][sk] = out; save_results(res)
            ok_s = out["recall"] >= 0.95
            print(f"  {arm:14s} recall={out['recall']:.3f}  {'OK' if ok_s else 'FAIL — task/training broken for this arm'}")
            if not ok_s and arm in ("A_full", "C_window100"):
                print(f"[abort] sanity failed for {arm} at D_in={D_IN} — task or training is broken, "
                      f"not a reach effect."); sys.exit(1)

    # ---- main sweep: arms × seeds at D=256, L=6 ----
    print(f"\n===== MAIN SWEEP (D={D}, L={N_LAYER}, ceiling={CEILING}, flat-stop OFF) =====")
    plan = [(arm, s) for arm in ARMS for s in SEEDS]
    for arm, seed in plan:
        rk = f"D{D}::{arm}::s{seed}"
        if rk in res["runs"]:
            r = res["runs"][rk]; print(f"[skip] {rk} solved={r['solved']} recall={r['recall']:.4f}"); continue
        print(f"[run] {rk}", flush=True)
        out = train_one(arm, seed, D, CEILING)
        res["runs"][rk] = out; aggregate(res); save_results(res)
        print(f"[done] {rk} solved={out['solved']} sts={out['steps_to_solve']} recall={out['recall']:.4f} "
              f"stop={out['stop_reason']}@{out['stop_step']} ({out['wall_s']}s)", flush=True)

    aggregate(res); save_results(res)
    try: make_plot(res)
    except Exception as e: print(f"[plot] skipped: {e}")
    print("\n===== SUMMARY (recall @ D=256, degree=100) =====")
    for arm in ARMS:
        s = res["summary"].get(arm)
        if s: print(f"  {arm:14s} {s['n_solved']}/{s['n_seeds']} solving  recall_med={s['recall_median']:.3f}  "
                     f"recalls={s['recalls']}  sts_med={s['steps_to_solve_median']}")
    print("Wrote", JSON_PATH)

def make_plot(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = res["summary"]; arms = [a for a in ARMS if a in s]
    colors = {"A_full": "#2e7d32", "C_window100": "#d1495b", "B_expander100": "#1f4e79", "B_random100": "#7aa6c2"}
    fig, ax = plt.subplots(figsize=(9, 5.2))
    chance = res["config"]["chance_recall"]
    for i, a in enumerate(arms):
        for r in s[a]["recalls"]:
            ax.plot(i, r, "o", color=colors.get(a, "#555"), alpha=0.6, ms=8, zorder=3)
        ax.plot(i, s[a]["recall_median"], "_", color="black", ms=26, mew=2.5, zorder=4)
        ax.text(i, 1.04, f"{s[a]['n_solved']}/{s[a]['n_seeds']}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(chance, ls=":", color="#888", lw=1.2, label=f"chance ({chance:.3f})")
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, rotation=12)
    ax.set_ylim(-0.04, 1.12); ax.set_ylabel("recall @ D=256");
    sv = res["config"]["saving_x"]
    ax.set_title(f"Fixed {DEGREE}-edge budget (~{sv}× saving): spread vs bunched at D=256\n"
                 f"(— = median; n/n = seeds solving)")
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout(); fig.savefig(PLOT_PATH, dpi=130); plt.close(fig)
    print("[plot] wrote", PLOT_PATH)

if __name__ == "__main__":
    main()
