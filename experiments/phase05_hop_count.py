"""
Phase 05 — Hop-count ablation: how many composition hops will the model actually perform
            when a clean multi-hop route to the needle is handed to it?

Phase 04 showed a single edge placed *on* the needle (offset = D) flips recall chance->1.00,
while random / Cayley edge sets sit at chance. The interpretation we want — "reachable isn't
reached" (the model won't *traverse* a multi-hop route even when one exists) — is confounded in
Phase 04: with 8 scattered edges and a randomized needle, there probably was no clean short
multi-hop path to the needle either. So "won't traverse a path" and "no clean path existed" are
tangled.

This phase removes that confound BY CONSTRUCTION. With a single UNIFORM dilated stride s, every
position attends exactly s tokens back, so the needle at D = k*s is reachable in exactly k hops
from every query and — because s > window — in no fewer (max single-hop reach is s, and D is an
exact multiple of s). We hold D=768, vary the stride, and read recall(k):

  arm           k   stride s=768/k   s>window?   role
  k1_direct     1   768              yes         positive control (reproduces Phase 04 B_dilated)
  k2            2   384              yes         2-hop composition
  k3            3   256              yes         3-hop composition
  k4            4   192              yes         4-hop composition
  window_only   -   (no long edge)   -           floor (D=768 is supra-window -> chance)
  scattered_k3  ~3  random offsets   -           same long-edge COUNT as k3, no clean chain
                                                 (replicates Phase 04 rand: it's the clean PATH,
                                                  not edge count, that matters)

All strides exceed the window (192 > 128), so each hop genuinely needs the dilated edge, not the
window. k tops out at 4 while the model has 8 layers, so a cliff at k=2/3 is a genuine routing-depth
limit, NOT a depth-ceiling artifact.

Path guard (methodological core, runs BEFORE training): for a sample of queries, verify
(a) a clean k-hop path to the needle exists (fraction == 1.00 for k1..k4 by construction) and
(b) no shorter path (measured shortest-path length == exactly k). If either fails for a k-arm the
construction is wrong -> stop, do not train.

Pre-committed primary prediction: a sharp recall collapse by k=2 or k=3 (recall(k=2) < 0.30,
k3/k4 ~ chance), i.e. effective routing depth ~1 hop, far below the 8-layer ceiling.

Model / task / training are identical to Phase 04 (25.2M decoder, RoPE, MQAR recall); only the
attention mask (edge set) changes.

Run:  ~/Code/HRS/.venv/bin/python experiments/phase05_hop_count.py
Knobs (smoke test): P5_MAX_STEPS / P5_MIN_STEPS / P5_FAIL_CUTOFF / P5_EVAL_EVERY / P5_EVAL_SEQS /
                    P5_SEEDS env vars; P5_GUARD_ONLY=1 runs the path guard and exits.
Resumable: completed (arm, seed) runs in phase05_records.json are skipped.
"""
import os, sys, json, time, math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.environ.get("P5_RESULT_DIR") or os.path.join(REPO, "results", "phase05")
JSON_PATH  = os.path.join(RESULT_DIR, "phase05_records.json")
PLOT_PATH  = os.path.join(RESULT_DIR, "phase05_recall_vs_hops.png")
os.makedirs(RESULT_DIR, exist_ok=True)

# ---- task vocab (identical to Phase 04) ----
K, V = 64, 64
PAD, QUERY = 0, 1
KEY0 = 2
VAL0 = 2 + K
VOCAB = 2 + K + V          # 130

# ---- geometry ----
T = 1024
WINDOW = 128
D = 768                    # fixed supra-window needle distance for every arm
STRIDE = {"k1_direct": 768, "k2": 384, "k3": 256, "k4": 192}   # s = D / k, all > WINDOW
K_REQ  = {"k1_direct": 1,   "k2": 2,   "k3": 3,   "k4": 4}
SCATTER_SRC_MIN = STRIDE["k3"]   # scattered_k3 equips exactly the nodes k3 does (i >= 256) -> matched edge count

# ---- model (identical to Phase 04) ----
N_LAYER, N_HEAD, N_EMBD, DROPOUT = 8, 8, 512, 0.0

# ---- training (Phase 04 protocol; env-overridable for smoke tests) ----
def _envint(name, default):
    v = os.environ.get(name)
    return int(v) if v else default
BATCH        = 64
LR           = 1e-3
WARMUP       = 200
WEIGHT_DECAY = 0.1
BETA1, BETA2 = 0.9, 0.95
GRAD_CLIP    = 1.0
EVAL_EVERY   = _envint("P5_EVAL_EVERY", 500)
EVAL_SEQS    = _envint("P5_EVAL_SEQS", 2048)
EVAL_SEED    = 9999
PLATEAU_DELTA= 0.005
PLATEAU_WIN  = 2000
MIN_STEPS    = _envint("P5_MIN_STEPS", 3000)
MAX_STEPS    = _envint("P5_MAX_STEPS", 12000)
FAIL_CUTOFF  = _envint("P5_FAIL_CUTOFF", 9000)   # = 3x the control's expected ~3000-step plateau
FAIL_RECALL  = 0.05
SEEDS        = [int(x) for x in os.environ.get("P5_SEEDS", "1337,1338").split(",")]
GUARD_ONLY   = os.environ.get("P5_GUARD_ONLY") == "1"
SCATTER_SEED = 20260603

ARMS = ["k1_direct", "k2", "k3", "k4", "window_only", "scattered_k3"]
ARM_DESC = {
    "k1_direct":    f"window w={WINDOW} + 1 dilated edge stride {STRIDE['k1_direct']} (1 hop; control)",
    "k2":           f"window w={WINDOW} + 1 dilated edge stride {STRIDE['k2']} (2 hops)",
    "k3":           f"window w={WINDOW} + 1 dilated edge stride {STRIDE['k3']} (3 hops)",
    "k4":           f"window w={WINDOW} + 1 dilated edge stride {STRIDE['k4']} (4 hops)",
    "window_only":  f"causal window w={WINDOW} only (floor)",
    "scattered_k3": f"window w={WINDOW} + 1 random long edge/node (i>={SCATTER_SRC_MIN}); k3 edge count, no clean chain",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

# --------------------------------------------------------------------------------------
# Synthetic generator (identical to Phase 04)
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
# Edge sets (uniform dilated stride; window-only; scattered) + masks
# --------------------------------------------------------------------------------------
def longrange_fn_for(arm, seed):
    if arm == "window_only":
        return None
    if arm in STRIDE:
        s = STRIDE[arm]
        return lambda i: [i - s] if i - s >= 0 else []
    if arm == "scattered_k3":
        rng = np.random.default_rng(SCATTER_SEED + seed)
        chosen = [None] * T
        for i in range(SCATTER_SRC_MIN, T):       # equip exactly the nodes k3 equips -> matched count
            chosen[i] = int(rng.integers(0, i))   # one random backward target, no uniform stride
        return lambda i: [chosen[i]] if (i < T and chosen[i] is not None) else []
    raise ValueError(arm)

def build_mask(arm, seed):
    """Additive (-inf) mask = causal window (w) [+ long edges]. Returns (mask, edges, lr_fn)."""
    lr_fn = longrange_fn_for(arm, seed)
    allow = np.zeros((T, T), dtype=bool)
    r = np.arange(T); allow[r, r] = True
    for i in range(T):
        allow[i, max(0, i - WINDOW):i + 1] = True
    if lr_fn is not None:
        for i in range(T):
            for j in lr_fn(i):
                if 0 <= j <= i:
                    allow[i, j] = True
    edges = int(allow.sum())
    add = np.zeros((T, T), dtype=np.float32); add[~allow] = float("-inf")
    return torch.from_numpy(add)[None, None].to(DEVICE, DTYPE), edges, lr_fn

# --- path guard ---------------------------------------------------------------------
INF = 1 << 30

def shortest_hops(q, t, lr_fn):
    """Min #edges from query q down to target t (t<q) over window (i->i-1..i-WINDOW) + long edges.
    Edges strictly decrease index, so a single descending sweep gives exact shortest paths."""
    dist = np.full(q + 1, INF, dtype=np.int64); dist[q] = 0
    for i in range(q, t, -1):
        di = dist[i]
        if di >= INF:
            continue
        lo = max(0, i - WINDOW)
        np.minimum(dist[lo:i], di + 1, out=dist[lo:i])      # window edges (contiguous band)
        if lr_fn is not None:
            for j in lr_fn(i):
                if 0 <= j < i and di + 1 < dist[j]:
                    dist[j] = di + 1
    return int(dist[t])

def path_guard(arm, seed, n_samples):
    lr_fn = longrange_fn_for(arm, seed)
    rng = np.random.default_rng(424242 + seed)
    qs = rng.integers(D + 1, T - 1, size=n_samples)
    hops = np.array([shortest_hops(int(q), int(q) - D, lr_fn) for q in qs])
    reachable = hops < INF
    reach_frac = float(reachable.mean())
    reach = hops[reachable]
    out = {"n_samples": int(n_samples), "reach_frac": round(reach_frac, 4),
           "shortest_med": int(np.median(reach)) if reach.size else None,
           "shortest_min": int(reach.min()) if reach.size else None,
           "shortest_max": int(reach.max()) if reach.size else None}
    if arm in K_REQ:
        out["clean_path_frac"] = round(float((hops == K_REQ[arm]).mean()), 4)
        out["k_required"] = K_REQ[arm]
    return out

# --------------------------------------------------------------------------------------
# Model (identical to Phase 04: RoPE, SDPA additive mask)
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

def train_one(arm, seed, fail_cutoff):
    mask, edges, _ = build_mask(arm, seed)
    torch.manual_seed(seed)
    model = GPT(mask).to(DEVICE)
    opt = configure_opt(model)
    rng = np.random.default_rng(seed * 100003 + D)
    ev = heldout(D)
    curve = []
    best = 0.0; t0 = time.time(); step = 0
    while step <= MAX_STEPS:
        if step % EVAL_EVERY == 0:
            rec, ce = eval_recall(model, ev)
            curve.append((step, round(rec, 4)))
            best = max(best, rec)
            print(f"  [{arm} s{seed}] step {step:5d}  recall {rec:.4f}  ce {ce:.3f}  "
                  f"best {best:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            if step >= fail_cutoff and best < FAIL_RECALL:
                break
            if step >= MIN_STEPS and best >= FAIL_RECALL:
                past = [r for (st, r) in curve if st <= step - PLATEAU_WIN]
                if past and (best - max(past)) < PLATEAU_DELTA:
                    break
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
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
    npar = model.n_params()
    del model, opt
    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.empty_cache()
    return {"recall": best, "final_recall": final_rec, "value_ce": final_ce,
            "steps_to_plateau": step, "fail_cutoff": fail_cutoff, "edges_total": edges,
            "curve": curve, "peak_gb": round(peak, 2), "params": int(npar),
            "wall_s": round(time.time() - t0, 1)}

# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------
def load_results():
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH) as f:
            return json.load(f)
    return {"config": {}, "runs": {}}

def save_results(res):
    with open(JSON_PATH, "w") as f:
        json.dump(res, f, indent=2)

def aggregate(res):
    summ = {}
    for arm in ARMS:
        recs = [res["runs"][f"{arm}::s{s}"]["recall"] for s in SEEDS if f"{arm}::s{s}" in res["runs"]]
        ces  = [res["runs"][f"{arm}::s{s}"]["value_ce"] for s in SEEDS if f"{arm}::s{s}" in res["runs"]]
        if recs:
            summ[arm] = {"n": len(recs), "recall_mean": float(np.mean(recs)),
                         "recall_min": float(np.min(recs)), "recall_max": float(np.max(recs)),
                         "value_ce_mean": float(np.mean(ces))}
    res["summary"] = summ
    return summ

def main():
    print(f"[env] device={DEVICE} dtype={DTYPE}  vocab={VOCAB}  T={T} w={WINDOW} D={D}  seeds={SEEDS}")

    # ---- path guard (before any training) ----
    n_guard = _envint("P5_GUARD_SAMPLES", 512)
    print(f"[guard] shortest-path to needle (offset D={D}) per arm, {n_guard} sampled queries:")
    guard = {}
    for arm in ARMS:
        g = path_guard(arm, SEEDS[0], n_guard)
        guard[arm] = g
        cp = f" clean(k={g.get('k_required')})={g['clean_path_frac']:.3f}" if "clean_path_frac" in g else ""
        print(f"  {arm:13s} reach={g['reach_frac']:.3f}  shortest med/min/max="
              f"{g['shortest_med']}/{g['shortest_min']}/{g['shortest_max']}{cp}")
    # hard gate: k-arms must have a guaranteed clean k-hop path and no shorter one
    bad = []
    for arm in K_REQ:
        g = guard[arm]
        if g["clean_path_frac"] < 0.999 or g["shortest_med"] != K_REQ[arm] or g["shortest_max"] != K_REQ[arm]:
            bad.append((arm, g))
    if bad:
        print("[guard] FAILED — construction is wrong, not training:")
        for arm, g in bad:
            print(f"   {arm}: {g}")
        sys.exit(1)
    print(f"[guard] OK: k1..k4 each have a clean k-hop path (frac 1.00) and shortest == k "
          f"(same uniform stride rule for every query). window_only needs {guard['window_only']['shortest_med']} "
          f"hops (supra-window); scattered_k3 has only irregular, query-specific paths (no single uniform chain).")

    if GUARD_ONLY:
        print("[guard-only] exiting before training."); return

    res = load_results()
    res["config"] = {
        "task": "MQAR associative recall (hop-count ablation)", "vocab": VOCAB, "K": K, "V": V,
        "T": T, "window": WINDOW, "D": D, "stride": STRIDE, "k_required": K_REQ,
        "scattered_src_min": SCATTER_SRC_MIN,
        "model": {"n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD, "dropout": DROPOUT},
        "batch": BATCH, "lr": LR, "max_steps": MAX_STEPS, "min_steps": MIN_STEPS,
        "fail_cutoff_base": FAIL_CUTOFF, "plateau_delta": PLATEAU_DELTA, "plateau_win": PLATEAU_WIN,
        "eval_seqs": EVAL_SEQS, "chance": round(1 / V, 4), "arms": ARM_DESC,
        "path_guard": guard, "seeds": SEEDS, "device": DEVICE, "dtype": str(DTYPE),
    }
    save_results(res)

    # k1_direct (control) first, to set the fail-cutoff = 3x its plateau for the other arms
    plan = [("k1_direct", s) for s in SEEDS] + \
           [(a, s) for s in SEEDS for a in ARMS if a != "k1_direct"]
    fail_cutoff = FAIL_CUTOFF
    for arm, seed in plan:
        rk = f"{arm}::s{seed}"
        if rk in res["runs"]:
            print(f"[skip] {rk} recall={res['runs'][rk]['recall']:.4f}")
            continue
        fc = FAIL_CUTOFF if arm == "k1_direct" else fail_cutoff
        print(f"[run] {rk}  (fail_cutoff={fc})", flush=True)
        out = train_one(arm, seed, fc)
        res["runs"][rk] = out
        aggregate(res); save_results(res)
        print(f"[done] {rk} recall={out['recall']:.4f} steps={out['steps_to_plateau']} "
              f"ce={out['value_ce']:.3f} ({out['wall_s']}s)", flush=True)
        # after both control seeds are in, lock fail_cutoff = max(base, 3x control plateau)
        if all(f"k1_direct::s{s}" in res["runs"] for s in SEEDS):
            ctrl = max(res["runs"][f"k1_direct::s{s}"]["steps_to_plateau"] for s in SEEDS)
            fail_cutoff = max(FAIL_CUTOFF, 3 * ctrl)

    aggregate(res)
    res["config"]["fail_cutoff_used"] = fail_cutoff
    save_results(res)
    try:
        make_plot(res)
    except Exception as e:
        print(f"[plot] skipped: {e}")
    print("\n===== SUMMARY (recall @ D=768) =====")
    for arm in ARMS:
        s = res["summary"].get(arm)
        kr = f"k={K_REQ[arm]}" if arm in K_REQ else "  - "
        print(f"  {arm:13s} {kr}  recall={s['recall_mean']:.3f}  ce={s['value_ce_mean']:.3f}" if s
              else f"  {arm:13s} --")
    print("Wrote", JSON_PATH)

def make_plot(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = res["summary"]
    karms = ["k1_direct", "k2", "k3", "k4"]
    ks = [K_REQ[a] for a in karms]
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    # per-seed points + mean line for the hop-count arms
    for a, kk in zip(karms, ks):
        for seed in SEEDS:
            r = res["runs"].get(f"{a}::s{seed}")
            if r:
                ax.plot(kk, r["recall"], "o", color="#7aa6c2", alpha=0.55, ms=6, zorder=3)
    means = [s.get(a, {}).get("recall_mean", np.nan) for a in karms]
    ax.plot(ks, means, "-o", color="#1f4e79", lw=2, ms=8, label="recall vs required hops k", zorder=4)
    for kk, m in zip(ks, means):
        if not np.isnan(m):
            ax.text(kk, m + 0.03, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    # floors
    wo = s.get("window_only", {}).get("recall_mean")
    sc = s.get("scattered_k3", {}).get("recall_mean")
    if wo is not None:
        ax.axhline(wo, ls=":", color="#9467bd", lw=1.4, label=f"window-only floor ({wo:.3f})")
    if sc is not None:
        ax.axhline(sc, ls="-.", color="#d1495b", lw=1.4, label=f"scattered_k3 floor ({sc:.3f})")
    ax.axhline(1 / V, ls="--", color="gray", lw=1, label=f"chance (1/V={1/V:.3f})")
    ax.set_xticks(ks); ax.set_xlabel("required composition hops  k  (stride s = D/k, D=768, w=128)")
    ax.set_ylabel("recall accuracy"); ax.set_ylim(-0.03, 1.08)
    ax.set_title("Phase 05 — recall vs guaranteed-clean routing depth (8-layer model)")
    ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOT_PATH, dpi=130)
    print(f"[plot] wrote {PLOT_PATH}")

if __name__ == "__main__":
    main()
