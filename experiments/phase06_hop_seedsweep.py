"""
Phase 06 — Composition depth: success rate x budget, decoupled from the one-hop control.

Phase 05 measured recall(k) at n=2 with a step budget anchored to 3x the trivial k1 plateau
(12k wire). It found k2 reliable, k3 BIMODAL (one seed 0.91 solving right AT the 12k limit, one
at chance), k4 dead. But the single k3 success landing *at* 12k is the signature of a late solver
clipped by too tight a budget, not a knife-edge — so "frays at 3, dead at 4" has two readings the
n=2 / 12k data cannot separate:

  - BUDGET reading: deeper composition is learnable but costs steeply more optimization per hop;
    the "cliff" is a cost curve, not a capacity wall.
  - LOTTERY / WALL reading: k3 genuinely depends on init (more steps won't fix it) and k4 is a
    true depth wall within 8 layers.

This phase separates them by DECOUPLING the budget from the control and estimating, per hop count,
(a) what fraction of seeds solve and (b) how many steps it takes when they do.

  arm   stride   hops   seeds            role
  k2    384      2      2   (1337-1338)  calibration / scale anchor (we know it solves)
  k3    256      3      12  (1337-1348)  MAIN arm — turn the coin-flip into a rate
  k4    192      4      4   (1337-1340)  probe — does "dead at 4" survive a generous budget?

Budget change vs Phase 05:
  - ceiling 40k steps (>3x the old 12k wire),
  - FLATNESS early-stop: if best value-CE has not improved by > FLAT_EPS over a trailing FLAT_WIN
    window, stop and mark the seed converged (no seed is called null while still descending),
  - STOP-ON-SOLVE: once recall >= 0.90 is sustained, stop (we have steps_to_solve; no reason to burn
    to 40k). This is what keeps the sweep near its ~20 GPU-hr ceiling.

  Subtlety this gate cannot escape: a *pre-onset* solver sits at the chance-entropy floor (CE = ln V
  = 4.159) with zero CE movement, indistinguishable from a dead seed by CE alone (the Phase 05 k3
  solver was flat until ~5.5k, then rose). So the flatness kill is gated to NEVER fire before
  FLAT_MIN_CHECK (=10k > the old 9k null point), and requires a full FLAT_WIN of flatness. A genuine
  onset before its seed's last-non-flat point is therefore caught; an onset *later* than a flat-kill
  is not — success_rate is thus a slight LOWER bound if very-late (>~kill-step) onsets exist. The
  steps-to-solve panel (vs the 12k line) shows where real onsets actually land.

Solve definition: first eval step where held-out recall >= 0.90 AND the next eval is also >= 0.90;
that first step is steps_to_solve. A seed that hits the ceiling or the flatness stop below 0.90 is a
non-solver, and we record its trailing-window CE slope (the FALSE-NULL gate: a real failure must be
flat, not descending).

Model / task / data / edge builder / path guard are IDENTICAL to Phase 05 — only the orchestration
(seed loop, budget, stopping rule, steps-to-solve logger) is new.

Run:  ~/Code/HRS/.venv/bin/python experiments/phase06_hop_seedsweep.py
Knobs: P6_MAX_STEPS / P6_EVAL_EVERY / P6_EVAL_SEQS / P6_FLAT_WIN / P6_FLAT_EPS / P6_FLAT_MIN_CHECK /
       P6_SEEDS_K2 / P6_SEEDS_K3 / P6_SEEDS_K4 ; P6_GUARD_ONLY=1 runs the path guard and exits;
       P6_SMOKE=1 sets a tiny budget for an end-to-end shakeout.
Resumable: completed (arm, seed) runs in phase06_records.json are skipped.
"""
import os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.environ.get("P6_RESULT_DIR") or os.path.join(REPO, "results", "phase06")
JSON_PATH  = os.path.join(RESULT_DIR, "phase06_records.json")
PLOT_PATH  = os.path.join(RESULT_DIR, "phase06_success_vs_hops.png")
os.makedirs(RESULT_DIR, exist_ok=True)

# ---- task vocab (identical to Phase 04/05) ----
K, V = 64, 64
PAD, QUERY = 0, 1
KEY0 = 2
VAL0 = 2 + K
VOCAB = 2 + K + V          # 130
CHANCE_CE = math.log(V)    # 4.1589 — uniform-over-V entropy = the chance floor for value-CE

# ---- geometry (identical to Phase 05) ----
T = 1024
WINDOW = 128
D = 768
STRIDE = {"k2": 384, "k3": 256, "k4": 192}   # s = D / k, all > WINDOW
K_REQ  = {"k2": 2,   "k3": 3,   "k4": 4}

# ---- model (identical to Phase 04/05) ----
N_LAYER, N_HEAD, N_EMBD, DROPOUT = 8, 8, 512, 0.0

# ---- training (Phase 05 protocol; budget/stop are the Phase 06 change) ----
def _envint(name, default):
    v = os.environ.get(name)
    return int(v) if v else default
def _envflt(name, default):
    v = os.environ.get(name)
    return float(v) if v else default

SMOKE        = os.environ.get("P6_SMOKE") == "1"
BATCH        = 64
LR           = 1e-3
WARMUP       = 200
WEIGHT_DECAY = 0.1
BETA1, BETA2 = 0.9, 0.95
GRAD_CLIP    = 1.0
EVAL_EVERY   = _envint("P6_EVAL_EVERY", 500)
EVAL_SEQS    = _envint("P6_EVAL_SEQS", 2048)
EVAL_SEED    = 9999
MIN_STEPS    = _envint("P6_MIN_STEPS", 3000)
MAX_STEPS    = _envint("P6_MAX_STEPS", 200 if SMOKE else 40000)   # ceiling
SOLVE_RECALL = _envflt("P6_SOLVE_RECALL", 0.90)
FLAT_WIN     = _envint("P6_FLAT_WIN", 5000)                       # trailing CE-flatness window
FLAT_EPS     = _envflt("P6_FLAT_EPS", 0.02)                      # min best-CE improvement to count as "still descending"
FLAT_MIN_CHECK = _envint("P6_FLAT_MIN_CHECK", 100 if SMOKE else 10000)  # never flat-kill before this step
GUARD_ONLY   = os.environ.get("P6_GUARD_ONLY") == "1"

def _seeds(name, default):
    v = os.environ.get(name)
    return [int(x) for x in v.split(",")] if v else default
SEEDS = {
    "k2": _seeds("P6_SEEDS_K2", [1337, 1338]),
    "k3": _seeds("P6_SEEDS_K3", list(range(1337, 1349))),   # 12 seeds
    "k4": _seeds("P6_SEEDS_K4", [1337, 1338, 1339, 1340]),  # 4 seeds
}
ARMS = ["k2", "k3", "k4"]
ARM_DESC = {a: f"window w={WINDOW} + 1 dilated edge stride {STRIDE[a]} ({K_REQ[a]} hops)" for a in ARMS}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

# --------------------------------------------------------------------------------------
# Synthetic generator (identical to Phase 04/05)
# --------------------------------------------------------------------------------------
N_PAIRS = 48

def make_batch(B, Dd, rng):
    toks = np.zeros((B, T), dtype=np.int64)
    tpos = np.zeros(B, dtype=np.int64)
    ttok = np.zeros(B, dtype=np.int64)
    for b in range(B):
        qk = int(rng.integers(Dd + 1, T - 1))
        if qk % 2 == 0:
            qk = qk - 1 if qk - 1 >= Dd + 1 else qk + 1
        nk, nv = qk - Dd - 1, qk - Dd
        keyset = rng.permutation(K)[:N_PAIRS]
        kq = int(keyset[0]); distractor_keys = keyset[1:]
        vq = int(rng.integers(0, V))
        even = np.arange(0, qk - 1, 2)
        even = even[even != nk]
        npos = min(len(distractor_keys), len(even))
        slots = rng.choice(even, size=npos, replace=False)
        dvals = rng.integers(0, V, size=npos)
        toks[b, slots] = KEY0 + distractor_keys[:npos]
        toks[b, slots + 1] = VAL0 + dvals
        toks[b, nk] = KEY0 + kq
        toks[b, nv] = VAL0 + vq
        toks[b, qk - 1] = QUERY
        toks[b, qk]     = KEY0 + kq
        toks[b, qk + 1] = VAL0 + vq
        tpos[b] = qk
        ttok[b] = VAL0 + vq
    return toks, tpos, ttok

def heldout(Dd):
    rng = np.random.default_rng(EVAL_SEED + Dd)
    return make_batch(EVAL_SEQS, Dd, rng)

# --------------------------------------------------------------------------------------
# Edge sets (uniform dilated stride) + masks  (identical to Phase 05)
# --------------------------------------------------------------------------------------
def longrange_fn_for(arm):
    s = STRIDE[arm]
    return lambda i: [i - s] if i - s >= 0 else []

def build_mask(arm):
    lr_fn = longrange_fn_for(arm)
    allow = np.zeros((T, T), dtype=bool)
    r = np.arange(T); allow[r, r] = True
    for i in range(T):
        allow[i, max(0, i - WINDOW):i + 1] = True
    for i in range(T):
        for j in lr_fn(i):
            if 0 <= j <= i:
                allow[i, j] = True
    edges = int(allow.sum())
    add = np.zeros((T, T), dtype=np.float32); add[~allow] = float("-inf")
    return torch.from_numpy(add)[None, None].to(DEVICE, DTYPE), edges, lr_fn

# --- path guard (identical to Phase 05; regression check) ---------------------------
INF = 1 << 30

def shortest_hops(q, t, lr_fn):
    dist = np.full(q + 1, INF, dtype=np.int64); dist[q] = 0
    for i in range(q, t, -1):
        di = dist[i]
        if di >= INF:
            continue
        lo = max(0, i - WINDOW)
        np.minimum(dist[lo:i], di + 1, out=dist[lo:i])
        for j in lr_fn(i):
            if 0 <= j < i and di + 1 < dist[j]:
                dist[j] = di + 1
    return int(dist[t])

def path_guard(arm, n_samples=512):
    lr_fn = longrange_fn_for(arm)
    rng = np.random.default_rng(424242)
    qs = rng.integers(D + 1, T - 1, size=n_samples)
    hops = np.array([shortest_hops(int(q), int(q) - D, lr_fn) for q in qs])
    reachable = hops < INF
    reach = hops[reachable]
    return {"n_samples": int(n_samples), "reach_frac": round(float(reachable.mean()), 4),
            "shortest_med": int(np.median(reach)) if reach.size else None,
            "shortest_min": int(reach.min()) if reach.size else None,
            "shortest_max": int(reach.max()) if reach.size else None,
            "clean_path_frac": round(float((hops == K_REQ[arm]).mean()), 4),
            "k_required": K_REQ[arm]}

# --------------------------------------------------------------------------------------
# Model (identical to Phase 04/05: RoPE, SDPA additive mask)
# --------------------------------------------------------------------------------------
def _build_rope(Tn, hd, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    freqs = torch.outer(torch.arange(Tn).float(), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()

_HD = N_EMBD // N_HEAD
_ROPE_COS, _ROPE_SIN = _build_rope(T, _HD)

def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

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
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx):
        x = self.tok_emb(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

def configure_opt(model):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": WEIGHT_DECAY},
                              {"params": nodecay, "weight_decay": 0.0}], lr=LR, betas=(BETA1, BETA2))

def get_lr(it):
    return LR * (it + 1) / WARMUP if it < WARMUP else LR

@torch.no_grad()
def eval_recall(model, ev):
    model.eval()
    toks, tpos, ttok = ev
    tt = torch.from_numpy(toks).to(DEVICE)
    tp = torch.from_numpy(tpos).to(DEVICE)
    tk = torch.from_numpy(ttok).to(DEVICE)
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

def _ce_slope_per_1k(curve, win):
    """Linear-fit slope of value-CE over the trailing `win` steps, in nats per 1000 steps.
    Negative = still descending; ~0 = flat (the false-null gate wants ~0 for a real failure)."""
    pts = [(st, ce) for (st, _r, ce) in curve if st >= curve[-1][0] - win]
    if len(pts) < 2:
        return None
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    slope = np.polyfit(xs, ys, 1)[0]      # nats per step
    return float(slope * 1000.0)

def train_one(arm, seed):
    mask, edges, _ = build_mask(arm)
    torch.manual_seed(seed)
    model = GPT(mask).to(DEVICE)
    opt = configure_opt(model)
    rng = np.random.default_rng(seed * 100003 + D)
    ev = heldout(D)
    curve = []                       # (step, recall, ce)
    best = 0.0; best_ce = float("inf"); t0 = time.time(); step = 0
    steps_to_solve = None; solve_cand = None; stop_reason = "ceiling"
    while step <= MAX_STEPS:
        if step % EVAL_EVERY == 0:
            rec, ce = eval_recall(model, ev)
            curve.append((step, round(rec, 4), round(ce, 4)))
            best = max(best, rec); best_ce = min(best_ce, ce)
            print(f"  [{arm} s{seed}] step {step:6d}  recall {rec:.4f}  ce {ce:.3f}  "
                  f"best {best:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            # --- solve detection (>=0.90 sustained over the next eval) ---
            if rec >= SOLVE_RECALL:
                if solve_cand is None:
                    solve_cand = step
                else:
                    steps_to_solve = solve_cand; stop_reason = "solved"; break
            else:
                solve_cand = None
            # --- flatness early-stop (gated; never before FLAT_MIN_CHECK) ---
            if step >= FLAT_MIN_CHECK and step >= MIN_STEPS and steps_to_solve is None:
                older = [c for (st, _r, c) in curve if st <= step - FLAT_WIN]
                recent = [c for (st, _r, c) in curve if st > step - FLAT_WIN]
                if older and recent and (min(older) - min(recent)) < FLAT_EPS:
                    stop_reason = "flat"; break
        for g in opt.param_groups:
            g["lr"] = get_lr(step)
        toks, tpos, ttok = make_batch(BATCH, D, rng)
        xb = torch.from_numpy(toks).to(DEVICE)
        pb = torch.from_numpy(tpos).to(DEVICE)
        kb = torch.from_numpy(ttok).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            logits = model(xb)
        idx = torch.arange(BATCH, device=DEVICE)
        loss = F.cross_entropy(logits[idx, pb].float(), kb)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); opt.step()
        step += 1
    final_rec, final_ce = eval_recall(model, ev)
    best = max(best, final_rec)
    solved = steps_to_solve is not None
    slope = None if solved else _ce_slope_per_1k(curve, FLAT_WIN)
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
    npar = model.n_params()
    del model, opt
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    return {"arm": arm, "seed": seed, "k": K_REQ[arm],
            "solved": solved, "steps_to_solve": steps_to_solve,
            "recall": best, "final_recall": final_rec, "value_ce": final_ce,
            "stop_reason": stop_reason, "stop_step": step,
            "final_ce_slope_per_1k": (round(slope, 5) if slope is not None else None),
            "edges_total": edges, "curve": curve,
            "peak_gb": round(peak, 2), "params": int(npar), "wall_s": round(time.time() - t0, 1)}

# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def load_results():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            return json.load(f)
    return {"config": {}, "runs": {}, "summary": {}}

def save_results(res):
    with open(JSON_PATH, "w") as f:
        json.dump(res, f, indent=2)

def aggregate(res):
    summ = {}
    for arm in ARMS:
        rk = [res["runs"][f"{arm}::s{s}"] for s in SEEDS[arm] if f"{arm}::s{s}" in res["runs"]]
        if not rk:
            continue
        solved = [r for r in rk if r["solved"]]
        sts = sorted(r["steps_to_solve"] for r in solved)
        nonsolver_slopes = [r["final_ce_slope_per_1k"] for r in rk if not r["solved"]]
        summ[arm] = {
            "k": K_REQ[arm], "n_seeds": len(rk), "n_solved": len(solved),
            "success_rate": round(len(solved) / len(rk), 4),
            "steps_to_solve": sts,
            "steps_to_solve_median": int(np.median(sts)) if sts else None,
            "steps_to_solve_min": (sts[0] if sts else None),
            "steps_to_solve_max": (sts[-1] if sts else None),
            "n_solved_past_12k": sum(1 for s in sts if s > 12000),
            "final_recall_mean": round(float(np.mean([r["recall"] for r in rk])), 4),
            "nonsolver_ce_slopes_per_1k": [s for s in nonsolver_slopes if s is not None],
            "nonsolver_max_slope_per_1k": (round(max(nonsolver_slopes), 5)
                                           if any(s is not None for s in nonsolver_slopes) else None),
        }
    res["summary"] = summ

def main():
    guard = {arm: path_guard(arm) for arm in ARMS}
    print("===== PATH GUARD (regression check; expect reach=1.00, shortest==k) =====")
    for arm in ARMS:
        g = guard[arm]
        ok = (g["reach_frac"] == 1.0 and g["shortest_med"] == K_REQ[arm]
              and g["shortest_min"] == K_REQ[arm] and g["shortest_max"] == K_REQ[arm])
        print(f"  {arm}: reach={g['reach_frac']} shortest=[{g['shortest_min']},{g['shortest_med']},"
              f"{g['shortest_max']}] clean_k={g['clean_path_frac']} k_req={K_REQ[arm]}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            print(f"[abort] path guard failed for {arm} — construction wrong, not training.")
            sys.exit(1)
    if GUARD_ONLY:
        print("[guard-only] done."); return

    res = load_results()
    res["config"] = {
        "task": "MQAR associative recall (composition depth: success rate x budget)",
        "vocab": VOCAB, "K": K, "V": V, "T": T, "window": WINDOW, "D": D,
        "stride": STRIDE, "k_required": K_REQ, "chance_recall": round(1 / V, 4),
        "chance_ce": round(CHANCE_CE, 4),
        "model": {"n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD, "dropout": DROPOUT},
        "batch": BATCH, "lr": LR, "max_steps": MAX_STEPS, "min_steps": MIN_STEPS,
        "solve_recall": SOLVE_RECALL, "flat_win": FLAT_WIN, "flat_eps": FLAT_EPS,
        "flat_min_check": FLAT_MIN_CHECK, "eval_every": EVAL_EVERY, "eval_seqs": EVAL_SEQS,
        "arms": ARM_DESC, "path_guard": guard, "seeds": SEEDS,
        "device": DEVICE, "dtype": str(DTYPE),
    }
    save_results(res)

    plan = [(a, s) for a in ARMS for s in SEEDS[a]]    # k2 (calibration) -> k3 (main) -> k4 (probe)
    for arm, seed in plan:
        rk = f"{arm}::s{seed}"
        if rk in res["runs"]:
            r = res["runs"][rk]
            print(f"[skip] {rk} solved={r['solved']} steps_to_solve={r['steps_to_solve']} "
                  f"recall={r['recall']:.4f}")
            continue
        print(f"[run] {rk}  (ceiling={MAX_STEPS}, flat_win={FLAT_WIN}@>{FLAT_MIN_CHECK})", flush=True)
        out = train_one(arm, seed)
        res["runs"][rk] = out
        aggregate(res); save_results(res)
        print(f"[done] {rk} solved={out['solved']} steps_to_solve={out['steps_to_solve']} "
              f"recall={out['recall']:.4f} stop={out['stop_reason']}@{out['stop_step']} "
              f"slope/1k={out['final_ce_slope_per_1k']} ({out['wall_s']}s)", flush=True)

    aggregate(res); save_results(res)
    try:
        make_plot(res)
    except Exception as e:
        print(f"[plot] skipped: {e}")
    print("\n===== SUMMARY (success rate @ D=768) =====")
    for arm in ARMS:
        s = res["summary"].get(arm)
        if not s:
            continue
        print(f"  {arm} (k={s['k']}): {s['n_solved']}/{s['n_seeds']} solved "
              f"(rate {s['success_rate']:.2f})  steps_to_solve med={s['steps_to_solve_median']} "
              f"max={s['steps_to_solve_max']}  past12k={s['n_solved_past_12k']}  "
              f"nonsolver_max_slope/1k={s['nonsolver_max_slope_per_1k']}")
    print("Wrote", JSON_PATH)

def make_plot(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = res["summary"]
    ks = [K_REQ[a] for a in ARMS if a in s]
    arms = [a for a in ARMS if a in s]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel A: success rate vs hop count
    rates = [s[a]["success_rate"] for a in arms]
    bars = axL.bar([str(k) for k in ks], rates, color="#1f4e79", width=0.55, zorder=3)
    for a, b in zip(arms, bars):
        n, ns = s[a]["n_seeds"], s[a]["n_solved"]
        axL.text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
                 f"{ns}/{n}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axL.axhline(0.5, ls=":", color="#888", lw=1.2, label="Phase 05 n=2 coin-flip (0.5)")
    axL.set_ylim(0, 1.08); axL.set_xlabel("required hops k"); axL.set_ylabel("success rate (recall ≥ 0.90)")
    axL.set_title("Composition success rate vs hop count")
    axL.legend(loc="upper right", fontsize=8)

    # Panel B: steps-to-solve distribution per k, with the old 12k cutoff line
    axR.axhline(12000, ls="--", color="#d1495b", lw=1.5, label="Phase 05 cutoff (12k)")
    axR.axhline(MAX_STEPS, ls=":", color="#999", lw=1.2, label=f"Phase 06 ceiling ({MAX_STEPS//1000}k)")
    for a, k in zip(arms, ks):
        sts = s[a]["steps_to_solve"]
        if sts:
            jitter = (np.arange(len(sts)) - (len(sts)-1)/2) * 0.04
            axR.plot([k + j for j in jitter], sts, "o", color="#7aa6c2", ms=8, alpha=0.85, zorder=3)
        # non-solvers as red x at the ceiling
        nns = s[a]["n_seeds"] - s[a]["n_solved"]
        if nns:
            jitter = (np.arange(nns) - (nns-1)/2) * 0.04
            axR.plot([k + j for j in jitter], [MAX_STEPS]*nns, "x", color="#d1495b", ms=9, mew=2, zorder=3,
                     label="_nolegend_")
    axR.set_xticks(ks); axR.set_xlabel("required hops k"); axR.set_ylabel("steps to solve")
    axR.set_title("Steps-to-solve per hop count (× = non-solver at ceiling)")
    axR.legend(loc="upper left", fontsize=8)
    fig.tight_layout(); fig.savefig(PLOT_PATH, dpi=130); plt.close(fig)
    print("[plot] wrote", PLOT_PATH)

if __name__ == "__main__":
    main()
