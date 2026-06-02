"""
Phase 03b — Loading the scaffold: shrink the window so long-range edges must work
            (WikiText-103, T=4096, window sweep)

Phase 03's first pass failed its own validity gate: at T=4096 with a 256-wide window,
arm C (window-only, ZERO long-range edges) tied arm B (window + 8 long-range) — so the
window did all the work and the long-range edges carried no measurable load. "Sparse
matches full" was therefore TRUE but uninformative about the expander: a local window
already suffices on WikiText-103 at this length, exactly as on tiny Shakespeare.

Fix: bottleneck the window. Hold T=4096 and sweep the window DOWN (w in {128, 64}) so
local attention can no longer cover the dependencies. Now the decisive questions become
load-bearing:
  - Does B (narrow window + 8 long-range edges) still match A (full attention)?  -> sufficiency
  - Does B clearly beat C (narrow window, no long-range)?                        -> the gate opens
If at a narrow window C falls well below B while B stays near A, the long-range edges are
demonstrably doing the work the window cannot — the expander scaffold is sufficient *with
the scaffold actually loaded*, not trivially.

Training budget doubled vs the 10k-step first pass (MAX_ITERS=20000, ~1.4 epochs of
WikiText-103) to reduce the undertraining confound; per-eval trajectories are logged so
the approach-to-plateau is visible.

Arms (T=4096, per window w):
  A_full     full causal attention (window-independent baseline / ceiling)
  B_random   window w + 8 random long-range/token   (the proposed sufficient scaffold)
  C_window   window w, no long-range                (load check: what long-range buys)

Run plan sweeps w in {128, 64}; w=256 result (gate closed) carries over from phase 03.

Run:  ~/Code/HRS/.venv/bin/python experiments/phase03b_window_sweep_load.py
Resumable: completed (T, arm, w, seed) runs in phase03b.json are skipped.
"""
import os, sys, json, time, math, itertools, collections
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

# --------------------------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------------------------
REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(REPO, "data", "wikitext103")
RESULT_DIR = os.path.join(REPO, "results", "phase03b")
JSON_PATH  = os.path.join(RESULT_DIR, "phase03b.json")
PLOT_PATH  = os.path.join(RESULT_DIR, "window_load_curve.png")
os.makedirs(RESULT_DIR, exist_ok=True)

# Model — GPT-2-small scale, identical across every arm.
N_LAYER, N_HEAD, N_EMBD = 12, 12, 768
DROPOUT   = 0.1
VOCAB     = 50257
LR_K      = 8            # long-range edges/token for B (random)

# Training — identical across arms.
MAX_ITERS    = 20000     # ~1.4 epochs of WikiText-103 (doubled vs phase 03's 10k first pass)
EVAL_INTERVAL= 2000
EVAL_ITERS   = 80
WARMUP_ITERS = 400
LR           = 3e-4      # small (8192-tok) batch -> conservative LR
MIN_LR       = 3e-5
WEIGHT_DECAY = 0.1
BETA1, BETA2 = 0.9, 0.95
GRAD_CLIP    = 1.0
BATCH = {1024: 8, 1320: 6, 4096: 2, 4896: 2}   # ~8192 tokens/step at every T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

T_FIXED = 4096
ARM_DESC = {
    "A_full":   "full causal attention (window-independent baseline / ceiling)",
    "B_random": "window w + 8 random long-range/token (proposed scaffold)",
    "C_window": "window w, no long-range (load check)",
}

# Ordered run plan: (T, arm, window, seed). Window-independent A uses window=0.
# Headline is the NARROW window (w=64): does B match A while C fails?
RUN_PLAN = [
    # P1: baseline (window-independent), 2 seeds
    (4096, "A_full", 0, 1337), (4096, "A_full", 0, 1338),
    # P2: NARROW window w=64 — the decisive loaded test, 2 seeds
    (4096, "B_random", 64, 1337), (4096, "C_window", 64, 1337),
    (4096, "B_random", 64, 1338), (4096, "C_window", 64, 1338),
    # P3: intermediate window w=128 — trace where the gate opens, 1 seed
    (4096, "B_random", 128, 1337), (4096, "C_window", 128, 1337),
]

# --------------------------------------------------------------------------------------
# Cayley graph (8-regular SL(2,Z_n)) for arm D
# --------------------------------------------------------------------------------------
def _sl2_elems(n):
    elems = [(a, b, c, d) for a, b, c, d in itertools.product(range(n), repeat=4)
             if (a * d - b * c) % n == 1]
    return elems

def cayley_adjacency8(n):
    """8-regular directed Cayley graph of SL(2,Z_n): generators a,a^-1,b,b^-1,a^2,a^-2,
    b^2,b^-2 with a=(1,1,0,1), b=(1,0,1,1). Returns (N, adj) with adj[i] an ordered list
    of i's <=8 distinct out-neighbors. (8 generators -> matches arm B's k=8 budget.)"""
    elems = _sl2_elems(n)
    idx = {e: i for i, e in enumerate(elems)}
    def mul(x, y):
        a, b, c, d = x; e, f, g, h = y
        return ((a*e + b*g) % n, (a*f + b*h) % n, (c*e + d*g) % n, (c*f + d*h) % n)
    a = (1, 1, 0, 1); b = (1, 0, 1, 1)
    gens = [a, (1, n-1, 0, 1), b, (1, 0, n-1, 1),
            (1, 2 % n, 0, 1), (1, (n-2) % n, 0, 1), (1, 0, 2 % n, 1), (1, 0, (n-2) % n, 1)]
    N = len(elems)
    adj = [[] for _ in range(N)]
    for i, e in enumerate(elems):
        for g in gens:
            j = idx[mul(e, g)]
            if j != i and j not in adj[i]:
                adj[i].append(j)
    return N, adj

_CAYLEY_SIZES = {6: 2, 24: 3, 120: 5, 336: 7, 1320: 11, 2184: 13, 4896: 17}  # |SL2(Z_n)|->n

def cayley_longrange_fn(T):
    """Return (longrange_fn, mode) for arm D at context length T.
    Exact node=position if T is an SL(2,Z_n) size; else a scaled mapping
    i -> floor(i*Nc/T) with Nc the smallest Cayley size >= T."""
    if T in _CAYLEY_SIZES:
        n = _CAYLEY_SIZES[T]; N, adj = cayley_adjacency8(n)
        assert N == T, (N, T)
        return (lambda i: adj[i]), f"exact(n={n}, N={N})"
    # scaled
    candidates = sorted(s for s in _CAYLEY_SIZES if s >= T)
    Nc = candidates[0] if candidates else max(_CAYLEY_SIZES)
    n = _CAYLEY_SIZES[Nc]; N, adj = cayley_adjacency8(n)
    def fn(i):
        node = (i * N) // T
        out = []
        for e in adj[node]:
            p = (e * T) // N
            if p != i and p not in out:
                out.append(p)
        return out
    return fn, f"scaled(n={n}, Nc={N}->T={T})"

# --------------------------------------------------------------------------------------
# Mask construction (additive -inf bias, (1,1,T,T))
# --------------------------------------------------------------------------------------
def build_allow(T, window, longrange_fn):
    allow = np.zeros((T, T), dtype=bool)
    rows = np.arange(T)
    allow[rows, rows] = True
    if window is not None:
        for i in range(T):
            allow[i, max(0, i - window):i + 1] = True
    if longrange_fn is not None:
        for i in range(T):
            for j in longrange_fn(i):
                if j <= i:
                    allow[i, j] = True
    return allow

def random_longrange_fn(T, k, seed):
    rng = np.random.default_rng(20260601 + seed)
    chosen = [rng.integers(0, i + 1, size=k).tolist() for i in range(T)]   # causal by construction
    return lambda i: chosen[i]

def make_mask_and_diag(T, arm, window, seed):
    """Returns (additive_mask or None, edges_total, diagnostics, mode_str).
    None mask => full causal (arm A) handled via is_causal."""
    if arm == "A_full":
        edges = T * (T + 1) // 2
        return None, edges, {"window_reach_frac": 1.0}, "full_causal"
    if arm == "B_random":
        lr_fn = random_longrange_fn(T, LR_K, seed); mode = f"random_w{window}"
    elif arm == "C_window":
        lr_fn = None; mode = f"window_only_w{window}"
    elif arm == "D_cayley":
        lr_fn, mode = cayley_longrange_fn(T)
    else:
        raise ValueError(arm)
    allow = build_allow(T, window, lr_fn)
    edges = int(allow.sum())
    diag = {"window_reach_frac": round(window / T, 4), "window": window, "mode": mode}
    if lr_fn is not None:
        diag.update(longrange_diag(T, lr_fn))
    add = np.zeros((T, T), dtype=np.float32)
    add[~allow] = float("-inf")
    mask = torch.from_numpy(add)[None, None].to(DEVICE, DTYPE)
    return mask, edges, diag, mode

def longrange_diag(T, lr_fn):
    """Cheap connectivity diagnostic of the symmetrized causal long-range graph
    (window/diagonal excluded). Spectral gap omitted (eigvalsh too costly at T=4096)."""
    adj = [[] for _ in range(T)]
    for i in range(T):
        for j in lr_fn(i):
            if 0 <= j <= i and j != i:
                adj[i].append(j); adj[j].append(i)
    seen = np.zeros(T, dtype=bool); n_comp = 0; giant = 0
    for s in range(T):
        if seen[s]:
            continue
        n_comp += 1; cnt = 0; q = collections.deque([s]); seen[s] = True
        while q:
            u = q.popleft(); cnt += 1
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True; q.append(v)
        giant = max(giant, cnt)
    deg = float(np.mean([len(a) for a in adj]))
    return {"lr_components": n_comp, "lr_giant": giant, "lr_mean_deg": round(deg, 3)}

# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_split(split):
    return np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")

def get_batch(data_arr, T, bs, gen):
    ix = torch.randint(len(data_arr) - T - 1, (bs,), generator=gen)
    x = torch.stack([torch.from_numpy(data_arr[i:i+T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data_arr[i+1:i+1+T].astype(np.int64)) for i in ix])
    return x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)

# --------------------------------------------------------------------------------------
# Model (SDPA attention; full=is_causal, sparse=additive mask via efficient backend)
# --------------------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, add_mask):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.resid_drop = nn.Dropout(DROPOUT)
        self.nh = N_HEAD
        self.causal = add_mask is None
        if add_mask is not None:
            self.register_buffer("add_mask", add_mask, persistent=False)
        else:
            self.add_mask = None

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(N_EMBD, dim=2)
        hd = C // self.nh
        q = q.view(B, T, self.nh, hd).transpose(1, 2)
        k = k.view(B, T, self.nh, hd).transpose(1, 2)
        v = v.view(B, T, self.nh, hd).transpose(1, 2)
        p = DROPOUT if self.training else 0.0
        if self.causal:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                              SDPBackend.MATH]):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=p)
        else:
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.add_mask, dropout_p=p)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.c_proj(y))

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(N_EMBD, 4 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD, bias=False)
        self.drop = nn.Dropout(DROPOUT)
    def forward(self, x):
        return self.drop(self.c_proj(F.gelu(self.c_fc(x))))

class Block(nn.Module):
    def __init__(self, add_mask):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD); self.attn = Attention(add_mask)
        self.ln2 = nn.LayerNorm(N_EMBD); self.mlp = MLP()
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, T, add_mask):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, N_EMBD)
        self.pos_emb = nn.Embedding(T, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block(add_mask) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, VOCAB, bias=False)
        self.tok_emb.weight = self.head.weight
        self.apply(self._init)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))
    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None])
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
    def num_params(self):
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()

def configure_optimizer(model):
    decay, nodecay = [], []
    for _, p in model.named_parameters():
        if p.requires_grad:
            (decay if p.dim() >= 2 else nodecay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": nodecay, "weight_decay": 0.0}], lr=LR, betas=(BETA1, BETA2))

def get_lr(it):
    if it < WARMUP_ITERS:
        return LR * (it + 1) / WARMUP_ITERS
    if it > MAX_ITERS:
        return MIN_LR
    r = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    return MIN_LR + 0.5 * (1 + math.cos(math.pi * r)) * (LR - MIN_LR)

@torch.no_grad()
def estimate_val_loss(model, val, T, bs):
    model.eval()
    gen = torch.Generator().manual_seed(0)   # same eval batches for every arm at this T
    losses = torch.zeros(EVAL_ITERS)
    for kk in range(EVAL_ITERS):
        x, y = get_batch(val, T, bs, gen)
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            _, loss = model(x, y)
        losses[kk] = loss.item()
    model.train()
    return losses.mean().item()

def train_one(T, arm, window, seed, splits):
    bs = BATCH[T]
    mask, edges, diag, mode = make_mask_and_diag(T, arm, window, seed)
    torch.manual_seed(seed); np.random.seed(seed)
    model = GPT(T, mask).to(DEVICE)
    opt = configure_optimizer(model)
    data_gen = torch.Generator().manual_seed(seed)
    best_val = float("inf"); best_iter = -1; t0 = time.time()
    for it in range(MAX_ITERS + 1):
        for g in opt.param_groups:
            g["lr"] = get_lr(it)
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
            vl = estimate_val_loss(model, splits["val"], T, bs)
            if vl < best_val:
                best_val, best_iter = vl, it
            print(f"  [{arm} w{window} s{seed}] it {it:5d}  val {vl:.4f}  ppl {math.exp(vl):.2f}  "
                  f"best {best_val:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            if it == MAX_ITERS:
                break
        x, y = get_batch(splits["train"], T, bs, data_gen)
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
    torch.cuda.reset_peak_memory_stats() if DEVICE == "cuda" else None
    np_ = model.num_params()
    del model, opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"best_val_loss": best_val, "best_ppl": math.exp(best_val), "best_iter": best_iter,
            "edges_total": edges, "bs": bs, "mode": mode, "diagnostics": diag,
            "peak_gb": round(peak, 2), "params": int(np_), "wall_s": round(time.time() - t0, 1)}

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
    for (T, arm, window, seed) in RUN_PLAN:
        rk = f"T{T}::{arm}::w{window}::{seed}"
        if rk not in res["runs"]:
            continue
        key = f"{arm}::w{window}"
        summ.setdefault(key, []).append(res["runs"][rk]["best_val_loss"])
    out = {}
    for key, vals in summ.items():
        v = np.array(vals)
        out[key] = {"n": len(v), "mean_loss": float(v.mean()),
                    "std_loss": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    "mean_ppl": float(math.exp(v.mean()))}
    # A is window-independent baseline; per window, gap B-A (sufficiency) and C-B (load)
    a = out.get("A_full::w0")
    for w in sorted(set(win for _, arm, win, _ in RUN_PLAN if arm != "A_full")):
        b, c = out.get(f"B_random::w{w}"), out.get(f"C_window::w{w}")
        if a and b:
            b["gap_to_full"] = b["mean_loss"] - a["mean_loss"]
        if b and c:
            c["gap_to_B"] = c["mean_loss"] - b["mean_loss"]
            c["gap_to_full"] = c["mean_loss"] - a["mean_loss"] if a else None
    res["summary"] = out
    return out

def main():
    print(f"[env] device={DEVICE} dtype={DTYPE}")
    splits = {"train": load_split("train"), "val": load_split("val")}
    print(f"[data] train={len(splits['train']):,} val={len(splits['val']):,} tokens")
    res = load_results()
    res["config"] = {
        "model": {"n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD, "vocab": VOCAB,
                  "dropout": DROPOUT},
        "T": T_FIXED, "lr_k": LR_K, "max_iters": MAX_ITERS, "lr": LR, "min_lr": MIN_LR,
        "warmup": WARMUP_ITERS, "weight_decay": WEIGHT_DECAY, "betas": [BETA1, BETA2],
        "batch_per_T": BATCH, "eval_iters": EVAL_ITERS, "dataset": "wikitext-103-raw-v1 (GPT-2 BPE)",
        "arms": ARM_DESC, "device": DEVICE, "dtype": str(DTYPE),
        "note": "quality-sufficiency only; masked-dense/SDPA is slower not faster than flash full attn",
    }
    save_results(res)

    for (T, arm, window, seed) in RUN_PLAN:
        rk = f"T{T}::{arm}::w{window}::{seed}"
        if rk in res["runs"]:
            print(f"[skip] {rk} best_val={res['runs'][rk]['best_val_loss']:.4f}")
            continue
        print(f"[run] {rk}  (bs={BATCH[T]})", flush=True)
        out = train_one(T, arm, window, seed, splits)
        res["runs"][rk] = out
        aggregate(res)
        save_results(res)
        print(f"[done] {rk} best_val={out['best_val_loss']:.4f} ppl={out['best_ppl']:.2f} "
              f"peak={out['peak_gb']}GB ({out['wall_s']}s)", flush=True)

    aggregate(res); save_results(res)
    try:
        make_plot(res)
    except Exception as e:
        print(f"[plot] skipped: {e}")
    print("\n===== SUMMARY =====")
    for key in sorted(res["summary"]):
        s = res["summary"][key]
        extra = ""
        if "gap_to_full" in s: extra += f"  gap_to_full={s['gap_to_full']:+.4f}"
        if "gap_to_B" in s: extra += f"  gap_to_B={s['gap_to_B']:+.4f}"
        print(f"{key:20s} val {s['mean_loss']:.4f} ± {s['std_loss']:.4f}  ppl {s['mean_ppl']:.2f}  (n={s['n']}){extra}")
    print("Wrote", JSON_PATH)

def make_plot(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = res["summary"]
    ws = sorted(set(win for _, arm, win, _ in RUN_PLAN if arm != "A_full"))
    a = s.get("A_full::w0", {})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # left: ppl vs window for B and C, with the (window-independent) full-attn baseline line
    for arm, col, mk, lab in [("B_random", "#2e86ab", "D", "B expander (window+8 long-range)"),
                              ("C_window", "#d1495b", "s", "C window-only")]:
        xs, ys = [], []
        for w in ws:
            key = f"{arm}::w{w}"
            if key in s:
                xs.append(w); ys.append(s[key]["mean_ppl"])
        if xs:
            ax[0].plot(xs, ys, marker=mk, color=col, label=lab, lw=2, ms=9)
    if a:
        ax[0].axhline(a["mean_ppl"], ls="--", color="#444", label="A full attention (baseline)")
    ax[0].invert_xaxis()
    ax[0].set_xlabel("window width w  (narrower →)"); ax[0].set_ylabel("val perplexity")
    ax[0].set_title(f"T={T_FIXED}: perplexity as the window narrows"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # right: gaps vs window — B-A (sufficiency) and C-B (load, gate opens as window narrows)
    ga, gc = [], []
    for w in ws:
        b = s.get(f"B_random::w{w}", {}); c = s.get(f"C_window::w{w}", {})
        ga.append(b.get("gap_to_full", np.nan)); gc.append(c.get("gap_to_B", np.nan))
    ax[1].axhline(0, color="gray", lw=1)
    ax[1].plot(ws, ga, marker="D", color="#2e86ab", lw=2, ms=9, label="B−A (sufficiency; thesis: ≈0)")
    ax[1].plot(ws, gc, marker="s", color="#d1495b", lw=2, ms=9, label="C−B (load; gate opens if >0)")
    ax[1].invert_xaxis()
    ax[1].set_xlabel("window width w  (narrower →)"); ax[1].set_ylabel("val-loss gap (nats)")
    ax[1].set_title("Sufficiency (B−A) and load (C−B) vs window"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOT_PATH, dpi=130)
    print(f"[plot] wrote {PLOT_PATH}")

if __name__ == "__main__":
    main()
