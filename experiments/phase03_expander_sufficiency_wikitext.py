"""
Phase 03 — Is an expander a *sufficient* attention scaffold? (WikiText-103, long context)

The thesis: attention's full Cartesian product (O(N^2)) is oversized scaffolding for
gradient descent. A sparse EXPANDER scaffold — local window + a constant handful of
long-range edges/token, O(N) — is SUFFICIENT: trained end-to-end it reaches the same loss
as the full product, even when real long-range dependency must be learned.

Phases 01-02 confirmed sufficiency on tiny Shakespeare but that task barely loads the
long-range edges (window alone nearly suffices). This phase puts real long-range load on
the scaffold via WikiText-103 (token-level, GPT-2 BPE) at two context lengths, with the
window held fixed (w=256) so the fraction of context the window can reach SHRINKS as T
grows (256/1024 = 25%  ->  256/4096 = 6%). The long-range edges must then carry dependency
the window cannot.

Prior art (Yi et al., LCEG, arXiv 2409.12181): approximate-attention methods systematically
UNDERPERFORM full attention on long-context tasks. The field expects sparse to lose. So
B matching A at T=4096 is an expectation-violating (strong) result; B falling behind
replicates the field and locates the sufficiency boundary.

Reading A (this script): the expander+window is a FIXED boolean (T,T) mask intersected with
causality, applied as a -inf additive bias before softmax (via SDPA's memory-efficient
backend, which scales to T=4096 without materializing the B*H*T*T score matrix). The graph
is frozen; all weights train from scratch. Only the mask varies across arms.

Arms (per context length T):
  A_full     full causal attention                          (ceiling / the thing replaced)
  B_random   window w=256 + k=8 random long-range/token     (the proposed sufficient scaffold)
  C_window   window w=256, NO long-range                    (load check: what long-range buys)
  D_cayley   window w=256 + 8 Cayley long-range/token       (construction-null confirmation)

Construction is settled by phases 01-02 (Cayley == random), so B (random) is the default
expander; D confirms the null a third time. At exact Cayley sizes (T=1320=|SL2(Z_11)|,
T=4896=|SL2(Z_17)|) D uses the pure node=position mapping of phases 01-02; at the spec'd
T=1024/4096 it uses a scaled mapping i->floor(i*Nc/T).

Headline: A vs B (does sparse match full?). Load check: C vs B (do long-range edges matter
at long T?). Construction: D vs B.

This phase measures QUALITY SUFFICIENCY only. The masked-dense/SDPA implementation is
slower, not faster, than full flash attention; the O(N) compute payoff needs a real sparse
kernel (a separate phase). Keep those separate.

Run:  ~/Code/HRS/.venv/bin/python experiments/phase03_expander_sufficiency_wikitext.py
Resumable: completed (T, arm, seed) runs in phase03.json are skipped.
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
RESULT_DIR = os.path.join(REPO, "results", "phase03")
JSON_PATH  = os.path.join(RESULT_DIR, "phase03.json")
PLOT_PATH  = os.path.join(RESULT_DIR, "sufficiency_curve.png")
os.makedirs(RESULT_DIR, exist_ok=True)

# Model — GPT-2-small scale, identical across every arm at a given T.
N_LAYER, N_HEAD, N_EMBD = 12, 12, 768
DROPOUT   = 0.1
VOCAB     = 50257
WINDOW    = 256          # fixed across both context lengths
LR_K      = 8            # long-range edges/token for B (random) and D (Cayley)

# Training — identical across arms at a given T.
MAX_ITERS    = 10000     # ~0.7 epoch of WikiText-103 (119M tok / 8192 tok/step ~ 14.5k/epoch)
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

ARM_DESC = {
    "A_full":   "full causal attention (ceiling)",
    "B_random": "window w=256 + 8 random long-range/token (proposed scaffold)",
    "C_window": "window w=256, no long-range (load check)",
    "D_cayley": "window w=256 + 8 Cayley long-range/token (construction null)",
}

# Ordered run plan — headline (loaded regime) first; resumable.
RUN_PLAN = [
    # P1: headline sufficiency at the LOADED regime (long context)
    (4096, "A_full", 1337), (4096, "B_random", 1337), (4096, "C_window", 1337),
    # P2: headline at short context
    (1024, "A_full", 1337), (1024, "B_random", 1337), (1024, "C_window", 1337),
    # P3: second seed for a noise estimate
    (4096, "A_full", 1338), (4096, "B_random", 1338), (4096, "C_window", 1338),
    (1024, "A_full", 1338), (1024, "B_random", 1338), (1024, "C_window", 1338),
    # P4: construction null at EXACT Cayley size (long), node=position mapping
    (4896, "B_random", 1337), (4896, "D_cayley", 1337),
    # P5: scaled-Cayley D at the spec'd long T (the other way to "do both")
    (4096, "D_cayley", 1337),
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

def make_mask_and_diag(T, arm, seed):
    """Returns (additive_mask or None, edges_total, diagnostics, mode_str).
    None mask => full causal (arm A) handled via is_causal."""
    if arm == "A_full":
        edges = T * (T + 1) // 2
        return None, edges, {"window_reach_frac": 1.0}, "full_causal"
    window = WINDOW
    if arm == "B_random":
        lr_fn = random_longrange_fn(T, LR_K, seed); mode = "random"
    elif arm == "C_window":
        lr_fn = None; mode = "window_only"
    elif arm == "D_cayley":
        lr_fn, mode = cayley_longrange_fn(T)
    else:
        raise ValueError(arm)
    allow = build_allow(T, window, lr_fn)
    edges = int(allow.sum())
    diag = {"window_reach_frac": round(WINDOW / T, 4), "mode": mode}
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

def train_one(T, arm, seed, splits):
    bs = BATCH[T]
    mask, edges, diag, mode = make_mask_and_diag(T, arm, seed)
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
            print(f"  [T{T} {arm} s{seed}] it {it:5d}  val {vl:.4f}  ppl {math.exp(vl):.2f}  "
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
    for (T, arm, seed) in RUN_PLAN:
        rk = f"T{T}::{arm}::{seed}"
        if rk not in res["runs"]:
            continue
        key = f"T{T}::{arm}"
        summ.setdefault(key, []).append(res["runs"][rk]["best_val_loss"])
    out = {}
    for key, vals in summ.items():
        v = np.array(vals)
        out[key] = {"n": len(v), "mean_loss": float(v.mean()),
                    "std_loss": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                    "mean_ppl": float(math.exp(v.mean()))}
    # headline gaps B-A and load-check C-B per T
    for T in sorted(set(t for t, _, _ in RUN_PLAN)):
        a, b, c = f"T{T}::A_full", f"T{T}::B_random", f"T{T}::C_window"
        if a in out and b in out:
            out[b]["gap_to_full"] = out[b]["mean_loss"] - out[a]["mean_loss"]
        if b in out and c in out:
            out[c]["gap_to_B"] = out[c]["mean_loss"] - out[b]["mean_loss"]
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
        "window": WINDOW, "lr_k": LR_K, "max_iters": MAX_ITERS, "lr": LR, "min_lr": MIN_LR,
        "warmup": WARMUP_ITERS, "weight_decay": WEIGHT_DECAY, "betas": [BETA1, BETA2],
        "batch_per_T": BATCH, "eval_iters": EVAL_ITERS, "dataset": "wikitext-103-raw-v1 (GPT-2 BPE)",
        "arms": ARM_DESC, "device": DEVICE, "dtype": str(DTYPE),
        "note": "quality-sufficiency only; masked-dense/SDPA is slower not faster than flash full attn",
    }
    save_results(res)

    for (T, arm, seed) in RUN_PLAN:
        rk = f"T{T}::{arm}::{seed}"
        if rk in res["runs"]:
            print(f"[skip] {rk} best_val={res['runs'][rk]['best_val_loss']:.4f}")
            continue
        print(f"[run] {rk}  (bs={BATCH[T]})", flush=True)
        out = train_one(T, arm, seed, splits)
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
    Ts = [1024, 4096]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    # left: ppl per arm at each spec T
    arms = [("A_full", "#444", "o"), ("B_random", "#2e86ab", "D"),
            ("C_window", "#d1495b", "s"), ("D_cayley", "#3c8d40", "^")]
    for arm, col, mk in arms:
        xs, ys = [], []
        for T in Ts:
            key = f"T{T}::{arm}"
            if key in s:
                xs.append(T); ys.append(s[key]["mean_ppl"])
        if xs:
            ax[0].plot(xs, ys, marker=mk, color=col, label=arm, lw=2, ms=9)
    ax[0].set_xscale("log", base=2); ax[0].set_xticks(Ts); ax[0].set_xticklabels([str(t) for t in Ts])
    ax[0].set_xlabel("context length T"); ax[0].set_ylabel("val perplexity")
    ax[0].set_title("Perplexity by arm vs context length"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    # right: B-A gap (sufficiency) and C-B gap (load) vs T
    ga, gc = [], []
    for T in Ts:
        b = s.get(f"T{T}::B_random", {}); c = s.get(f"T{T}::C_window", {})
        ga.append(b.get("gap_to_full", np.nan)); gc.append(c.get("gap_to_B", np.nan))
    ax[1].axhline(0, color="gray", lw=1)
    ax[1].plot(Ts, ga, marker="D", color="#2e86ab", lw=2, ms=9, label="B−A gap (sufficiency; thesis: ≈0)")
    ax[1].plot(Ts, gc, marker="s", color="#d1495b", lw=2, ms=9, label="C−B gap (load; valid if >0 at large T)")
    ax[1].set_xscale("log", base=2); ax[1].set_xticks(Ts); ax[1].set_xticklabels([str(t) for t in Ts])
    ax[1].set_xlabel("context length T"); ax[1].set_ylabel("val-loss gap (nats)")
    ax[1].set_title("Sufficiency (B−A) and load (C−B)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(PLOT_PATH, dpi=130)
    print(f"[plot] wrote {PLOT_PATH}")

if __name__ == "__main__":
    main()
