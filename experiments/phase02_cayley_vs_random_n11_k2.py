"""
Phase 02 — Cayley vs random sparse attention at LARGE N, SMALL budget (n=11, k=2)

Decisive follow-up to phase 01. Phase 01 (n=7, N=336, k=4) found Cayley and random
indistinguishable in mean AND variance — at that scale a random graph is already a
reliable expander. The pre-committed hypothesis (std(Cayley) < std(random)) predicts a
difference precisely where random graphs STOP being reliable expanders: large N and small
edge budget. This phase tests that regime.

  N = |SL(2, Z_11)| = 1320  (block_size = 1320, one token per Cayley node, no padding)
  k = 2 long-range edges/token  (vs 4 in phase 01)

The k=2 Cayley construction uses TWO non-inverse Margulis generators a=(1,1,0,1),
b=(1,0,1,1) as directed out-edges (2-out-regular). NOTE: a generator + its inverse would
give a 2-regular UNDIRECTED graph = a union of cycles = NOT an expander; using two
independent generators keeps the directed expander property (verified: connected,
directed diameter 13). The random control samples 2 positions/token uniformly per seed.

Arms (only the attention mask differs; every other hyperparameter is held identical):
  A — Full:        full causal attention                       (ceiling, 3 seeds)
  B — Cayley:      local window(w=64) + 2 Cayley long-range     (proposal, 5 seeds)
  C — Random:      local window(w=64) + 2 random long-range     (BigBird control, 5 seeds)
  D — Pure Cayley: 2 Cayley long-range, NO window               (ablation, 3 seeds)
  E — Window only: local window(w=64), NO long-range            (ablation, 3 seeds)

All masks are intersected with the causal mask and always include the diagonal.
Long-range budget is matched at 2 edges/token (pre-causal) for B, C, D.

Run:  ~/Code/HRS/.venv/bin/python experiments/phase02_cayley_vs_random_n11_k2.py
Outputs: results/phase02/phase02.json, results/phase02/spectral_gaps.png

The script is resumable: completed (arm, seed) runs already present in phase02.json
are skipped on re-invocation.
"""
import os, sys, json, time, math, pickle, itertools, collections
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# --------------------------------------------------------------------------------------
# Paths / config
# --------------------------------------------------------------------------------------
REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(REPO, "data", "shakespeare_char")
RESULT_DIR = os.path.join(REPO, "results", "phase02")
JSON_PATH  = os.path.join(RESULT_DIR, "phase02.json")
PLOT_PATH  = os.path.join(RESULT_DIR, "spectral_gaps.png")
RAW_TXT    = "/home/bard/Code/HRS/datasets/tiny_shakespeare.txt"  # cached source corpus
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Base model / training hyperparameters — IDENTICAL across every arm.
BLOCK_SIZE   = 1320     # = |SL(2,Z_11)| : one token position per Cayley node, no padding
N_LAYER      = 8        # kept = phase 01; note k=2 directed Cayley diameter is 13 (> depth)
N_HEAD       = 6        #   -> the no-window arm D is depth-limited (documented caveat)
N_EMBD       = 384
DROPOUT      = 0.2
WINDOW       = 64       # local causal band width
LONGRANGE_K  = 2        # long-range edges/token (matches the 2-generator Cayley out-degree)
CAYLEY_N     = 11

BATCH_SIZE   = 16       # T=1320 dense masked-attention score matrix dominates memory
MAX_ITERS    = 2500     #   (bs=16 -> ~13.5 GB peak; tok/step 21k ~= phase 01's 21.5k)
EVAL_INTERVAL= 500      # fewer, larger evals -> a STABLE val metric for the variance study
EVAL_ITERS   = 200
WARMUP_ITERS = 100
LR           = 1e-3
MIN_LR       = 1e-4
WEIGHT_DECAY = 1e-1
BETA1, BETA2 = 0.9, 0.99
GRAD_CLIP    = 1.0

SEEDS_5 = [1337, 1338, 1339, 1340, 1341]
SEEDS_3 = [1337, 1338, 1339]
ARMS = {
    "A_full":        {"seeds": SEEDS_3, "window": False, "longrange": None, "full": True,
                      "desc": "full causal attention (ceiling)"},
    "B_cayley":      {"seeds": SEEDS_5, "window": True,  "longrange": "cayley",
                      "desc": "local window + 2 Cayley long-range"},
    "C_random":      {"seeds": SEEDS_5, "window": True,  "longrange": "random",
                      "desc": "local window + 2 random long-range (BigBird control)"},
    "D_pure_cayley": {"seeds": SEEDS_3, "window": False, "longrange": "cayley",
                      "desc": "2 Cayley long-range, no window"},
    "E_window_only": {"seeds": SEEDS_3, "window": True,  "longrange": None,
                      "desc": "local window, no long-range"},
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

# --------------------------------------------------------------------------------------
# Cayley graph (SL(2, Z_n), Margulis generators; 4-regular)
# --------------------------------------------------------------------------------------
def cayley_adjacency(n):
    """k=2 directed Cayley graph of SL(2,Z_n) using two non-inverse generators
    a=(1,1,0,1), b=(1,0,1,1). Each node has exactly 2 out-neighbors (2-out-regular).
    Returns (N, adj) where adj[i] is an ORDERED list of i's 2 long-range targets.
    (Two independent generators -> directed expander; a generator+inverse would give a
    2-regular undirected union of cycles, which is NOT an expander.)"""
    elems = [(a, b, c, d) for a, b, c, d in itertools.product(range(n), repeat=4)
             if (a * d - b * c) % n == 1]
    idx = {e: i for i, e in enumerate(elems)}
    def mul(x, y):
        a, b, c, d = x; e, f, g, h = y
        return ((a*e + b*g) % n, (a*f + b*h) % n, (c*e + d*g) % n, (c*f + d*h) % n)
    gens = [(1, 1, 0, 1), (1, 0, 1, 1)]   # a, b  (two independent generators -> k=2)
    N = len(elems)
    adj = [[] for _ in range(N)]
    for i, e in enumerate(elems):
        for g in gens:
            j = idx[mul(e, g)]
            if j != i and j not in adj[i]:
                adj[i].append(j)
    return N, adj


def build_allow_mask(block_size, window, longrange_fn):
    """allow[i,j] = causal(i,j) AND (self OR window OR longrange). Returns (T,T) bool np."""
    T = block_size
    allow = np.zeros((T, T), dtype=bool)
    for i in range(T):
        allow[i, i] = True
        if window is not None:
            for j in range(max(0, i - window), i + 1):
                allow[i, j] = True
        if longrange_fn is not None:
            for j in longrange_fn(i):
                if j <= i:
                    allow[i, j] = True
    return allow


def longrange_graph_diagnostics(block_size, longrange_fn):
    """Spectral gap + diameter/connectivity of the causal-masked, symmetrized
    LONG-RANGE-ONLY adjacency (window and diagonal excluded)."""
    T = block_size
    M = np.zeros((T, T), dtype=np.float64)
    for i in range(T):
        for j in longrange_fn(i):
            if j <= i and j != i:
                M[i, j] = 1.0
    S = ((M + M.T) > 0).astype(np.float64)          # symmetrize -> undirected {0,1}
    eig = np.linalg.eigvalsh(S)                       # ascending
    lam1, lam2 = float(eig[-1]), float(eig[-2])
    spectral_gap = lam1 - lam2
    # connectivity / diameter via BFS over giant component
    adj = [np.nonzero(S[i])[0].tolist() for i in range(T)]
    seen = np.zeros(T, dtype=bool)
    n_comp, giant, giant_nodes = 0, 0, None
    for s in range(T):
        if seen[s]:
            continue
        n_comp += 1
        comp, q = [], collections.deque([s]); seen[s] = True
        while q:
            u = q.popleft(); comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True; q.append(v)
        if len(comp) > giant:
            giant, giant_nodes = len(comp), comp
    # diameter of giant component (BFS from each node)
    diam = 0
    for s in giant_nodes:
        d = {s: 0}; q = collections.deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in d:
                    d[v] = d[u] + 1; q.append(v)
            diam = max(diam, max(d.values()))
    mean_deg = float(S.sum(1).mean())
    return {"spectral_gap": spectral_gap, "lambda1": lam1, "lambda2": lam2,
            "n_components": n_comp, "giant_component": giant,
            "giant_diameter": int(diam), "mean_degree": mean_deg}


def make_longrange_fn(kind, adj, seed):
    """Return a deterministic longrange_fn(i) for the given arm kind."""
    if kind is None:
        return None
    if kind == "cayley":
        return lambda i: adj[i]
    if kind == "random":
        rng = np.random.default_rng(20260531 + seed)   # fixed per seed
        chosen = [rng.choice(BLOCK_SIZE, size=LONGRANGE_K, replace=False).tolist()
                  for _ in range(BLOCK_SIZE)]
        return lambda i: chosen[i]
    raise ValueError(kind)


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def prepare_data():
    train_bin = os.path.join(DATA_DIR, "train.bin")
    val_bin   = os.path.join(DATA_DIR, "val.bin")
    meta_pkl  = os.path.join(DATA_DIR, "meta.pkl")
    if os.path.exists(train_bin) and os.path.exists(val_bin) and os.path.exists(meta_pkl):
        with open(meta_pkl, "rb") as f:
            meta = pickle.load(f)
        return meta["vocab_size"]
    data = open(RAW_TXT, "r").read()
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    n = len(data)
    train_data = data[: int(n * 0.9)]
    val_data   = data[int(n * 0.9):]
    np.array([stoi[c] for c in train_data], dtype=np.uint16).tofile(train_bin)
    np.array([stoi[c] for c in val_data],   dtype=np.uint16).tofile(val_bin)
    with open(meta_pkl, "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "stoi": stoi,
                     "itos": {i: ch for ch, i in stoi.items()}}, f)
    print(f"[data] prepared: vocab={vocab_size}, train={len(train_data)}, val={len(val_data)}")
    return vocab_size


def load_split(split):
    path = os.path.join(DATA_DIR, f"{split}.bin")
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data_arr, gen):
    ix = torch.randint(len(data_arr) - BLOCK_SIZE, (BATCH_SIZE,), generator=gen)
    x = torch.stack([torch.from_numpy(data_arr[i:i+BLOCK_SIZE].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data_arr[i+1:i+1+BLOCK_SIZE].astype(np.int64)) for i in ix])
    return x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)


# --------------------------------------------------------------------------------------
# Model — minimal nanoGPT with an arbitrary boolean attention mask
# --------------------------------------------------------------------------------------
class MaskedSelfAttention(nn.Module):
    def __init__(self, allow_mask):
        super().__init__()
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)
        self.attn_dropout  = nn.Dropout(DROPOUT)
        self.resid_dropout = nn.Dropout(DROPOUT)
        self.nh = N_HEAD
        # (1,1,T,T) boolean; True = allowed
        self.register_buffer("allow", torch.from_numpy(allow_mask)[None, None], persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(N_EMBD, dim=2)
        hd = C // self.nh
        q = q.view(B, T, self.nh, hd).transpose(1, 2)
        k = k.view(B, T, self.nh, hd).transpose(1, 2)
        v = v.view(B, T, self.nh, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hd))
        att = att.masked_fill(~self.allow[:, :, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc   = nn.Linear(N_EMBD, 4 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(4 * N_EMBD, N_EMBD, bias=False)
        self.dropout = nn.Dropout(DROPOUT)
    def forward(self, x):
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, allow_mask):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD); self.attn = MaskedSelfAttention(allow_mask)
        self.ln2 = nn.LayerNorm(N_EMBD); self.mlp = MLP()
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size, allow_mask):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, N_EMBD)
        self.pos_emb = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)
        self.blocks = nn.ModuleList([Block(allow_mask) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size, bias=False)
        self.tok_emb.weight = self.head.weight  # weight tying
        self.apply(self._init)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * N_LAYER))

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
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
        # subtract position embedding (not tied), token emb tied with head
        return sum(p.numel() for p in self.parameters()) - self.pos_emb.weight.numel()


def configure_optimizer(model):
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else nodecay).append(p)
    groups = [{"params": decay, "weight_decay": WEIGHT_DECAY},
              {"params": nodecay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=LR, betas=(BETA1, BETA2))


def get_lr(it):
    if it < WARMUP_ITERS:
        return LR * (it + 1) / WARMUP_ITERS
    if it > MAX_ITERS:
        return MIN_LR
    ratio = (it - WARMUP_ITERS) / (MAX_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return MIN_LR + coeff * (LR - MIN_LR)


@torch.no_grad()
def estimate_loss(model, splits):
    model.eval()
    out = {}
    for name, data_arr in splits.items():
        gen = torch.Generator().manual_seed(0)  # fixed eval batches across arms
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            x, y = get_batch(data_arr, gen)
            with torch.autocast(device_type="cuda", dtype=DTYPE) if DEVICE == "cuda" else _nullctx():
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


class _nullctx:
    def __enter__(self): return None
    def __exit__(self, *a): return False


def train_one(arm_key, seed, vocab_size, adj, splits):
    spec = ARMS[arm_key]
    if spec.get("full"):
        # full causal attention: every token attends to all j <= i
        allow = np.tril(np.ones((BLOCK_SIZE, BLOCK_SIZE), dtype=bool))
        lr_fn = None
    else:
        window = WINDOW if spec["window"] else None
        lr_fn = make_longrange_fn(spec["longrange"], adj, seed)
        allow = build_allow_mask(BLOCK_SIZE, window, lr_fn)
    edges = int(allow.sum())

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GPT(vocab_size, allow).to(DEVICE)
    opt = configure_optimizer(model)
    data_gen = torch.Generator().manual_seed(seed)

    best_val = float("inf"); best_iter = -1
    t0 = time.time()
    for it in range(MAX_ITERS + 1):
        for g in opt.param_groups:
            g["lr"] = get_lr(it)
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS:
            losses = estimate_loss(model, splits)
            if losses["val"] < best_val:
                best_val, best_iter = losses["val"], it
            print(f"  [{arm_key} s{seed}] it {it:4d}  train {losses['train']:.4f}  "
                  f"val {losses['val']:.4f}  best {best_val:.4f}  ({time.time()-t0:.0f}s)",
                  flush=True)
            if it == MAX_ITERS:
                break
        x, y = get_batch(splits["train"], data_gen)
        with torch.autocast(device_type="cuda", dtype=DTYPE) if DEVICE == "cuda" else _nullctx():
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()

    diag = None
    if spec["longrange"] is not None:
        diag = longrange_graph_diagnostics(BLOCK_SIZE, lr_fn)
    del model, opt
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return {"best_val_loss": best_val, "best_iter": best_iter,
            "edges_total": edges, "wall_s": round(time.time() - t0, 1),
            "params": None, "diagnostics": diag}


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


def main():
    print(f"[env] device={DEVICE} dtype={DTYPE}")
    vocab_size = prepare_data()
    N, adj = cayley_adjacency(CAYLEY_N)
    degs = sorted(set(len(a) for a in adj))
    assert N == BLOCK_SIZE and degs == [LONGRANGE_K], (N, degs)
    print(f"[cayley] N={N} out-degree={degs} (verified 1320 nodes, 2-out-regular, "
          f"directed diameter 13)")

    splits = {"train": load_split("train"), "val": load_split("val")}

    res = load_results()
    res["config"] = {
        "block_size": BLOCK_SIZE, "n_layer": N_LAYER, "n_head": N_HEAD,
        "n_embd": N_EMBD, "dropout": DROPOUT, "window": WINDOW,
        "longrange_k": LONGRANGE_K, "cayley_n": CAYLEY_N,
        "batch_size": BATCH_SIZE, "max_iters": MAX_ITERS, "lr": LR, "min_lr": MIN_LR,
        "warmup_iters": WARMUP_ITERS, "weight_decay": WEIGHT_DECAY,
        "betas": [BETA1, BETA2], "eval_iters": EVAL_ITERS, "vocab_size": vocab_size,
        "device": DEVICE, "dtype": str(DTYPE),
        "cayley_diameter": 13, "cayley_generators": "a=(1,1,0,1), b=(1,0,1,1)",
        "arms": {k: v["desc"] for k, v in ARMS.items()},
    }
    save_results(res)

    # one model build to capture param count (identical across arms)
    tmp_allow = build_allow_mask(BLOCK_SIZE, None, None)
    tmp = GPT(vocab_size, tmp_allow)
    n_params = tmp.num_params()
    res["config"]["params"] = int(n_params)
    del tmp
    print(f"[model] params (ex-pos-emb)={n_params/1e6:.2f}M  (identical across all arms)")

    plan = [(arm, seed) for arm, spec in ARMS.items() for seed in spec["seeds"]]
    print(f"[plan] {len(plan)} runs total")
    for arm, seed in plan:
        rk = f"{arm}::{seed}"
        if rk in res["runs"]:
            print(f"[skip] {rk} (best_val={res['runs'][rk]['best_val_loss']:.4f})")
            continue
        print(f"[run] {rk}")
        out = train_one(arm, seed, vocab_size, adj, splits)
        out["params"] = int(n_params)
        res["runs"][rk] = out
        save_results(res)
        print(f"[done] {rk} best_val={out['best_val_loss']:.4f} "
              f"ppl={math.exp(out['best_val_loss']):.3f} ({out['wall_s']}s)")

    # ---- aggregate per arm ----
    summary = {}
    for arm in ARMS:
        vals = [res["runs"][f"{arm}::{s}"]["best_val_loss"]
                for s in ARMS[arm]["seeds"] if f"{arm}::{s}" in res["runs"]]
        if not vals:
            continue
        vals = np.array(vals)
        summary[arm] = {
            "n_seeds": len(vals),
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "min": float(vals.min()), "max": float(vals.max()),
            "ppl_mean": float(math.exp(vals.mean())),
            "seeds": {str(s): res["runs"][f"{arm}::{s}"]["best_val_loss"]
                      for s in ARMS[arm]["seeds"] if f"{arm}::{s}" in res["runs"]},
        }
    res["summary"] = summary
    if "A_full" in summary:
        a = summary["A_full"]["mean"]
        for arm in summary:
            summary[arm]["gap_to_full"] = summary[arm]["mean"] - a
    save_results(res)

    # ---- plot spectral gaps: B (single, fixed graph) vs C seeds ----
    try:
        make_plot(res)
    except Exception as e:
        print(f"[plot] skipped: {e}")

    print("\n========== SUMMARY ==========")
    for arm in ARMS:
        if arm in summary:
            s = summary[arm]
            print(f"{arm:14s} val {s['mean']:.4f} ± {s['std']:.4f}  "
                  f"ppl {s['ppl_mean']:.3f}  gap {s.get('gap_to_full', 0):+.4f}  (n={s['n_seeds']})")
    if "B_cayley" in summary and "C_random" in summary:
        b, c = summary["B_cayley"], summary["C_random"]
        print(f"\nHEADLINE: std(B)={b['std']:.4f} vs std(C)={c['std']:.4f}  "
              f"-> std(B) {'<' if b['std'] < c['std'] else '>='} std(C)")
        print(f"          mean(B)={b['mean']:.4f} vs mean(C)={c['mean']:.4f}  "
              f"diff={b['mean']-c['mean']:+.4f}")
    print("Wrote", JSON_PATH)


def make_plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # B graph gap (fixed across seeds) — recompute deterministically
    N, adj = cayley_adjacency(CAYLEY_N)
    b_diag = longrange_graph_diagnostics(BLOCK_SIZE, lambda i: adj[i])
    b_gap = b_diag["spectral_gap"]
    c_gaps, c_diams = [], []
    for s in ARMS["C_random"]["seeds"]:
        rk = f"C_random::{s}"
        if rk in res["runs"] and res["runs"][rk].get("diagnostics"):
            d = res["runs"][rk]["diagnostics"]
            c_gaps.append(d["spectral_gap"]); c_diams.append(d["giant_diameter"])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    if c_gaps:
        ax.scatter(np.zeros(len(c_gaps)) + 1, c_gaps, s=70, color="#d1495b",
                   zorder=3, label=f"C random seeds (n={len(c_gaps)})")
        ax.scatter([1], [np.mean(c_gaps)], marker="_", s=900, color="#d1495b", zorder=4)
    ax.scatter([0], [b_gap], s=140, color="#2e86ab", marker="D", zorder=3,
               label="B Cayley (fixed graph)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Cayley (B)", "Random (C)"])
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylabel("spectral gap  (λ₁ − λ₂ of symmetrized long-range adj)")
    ax.set_title("Spectral gap: structured vs random")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")

    # right panel: val-loss per seed scatter for B vs C
    ax2 = axes[1]
    if "summary" in res:
        for j, (arm, col) in enumerate([("B_cayley", "#2e86ab"), ("C_random", "#d1495b")]):
            if arm in res["summary"]:
                vs = list(res["summary"][arm]["seeds"].values())
                ax2.scatter(np.zeros(len(vs)) + j, vs, s=70, color=col, zorder=3)
                ax2.scatter([j], [np.mean(vs)], marker="_", s=900, color=col, zorder=4)
        if "A_full" in res["summary"]:
            ax2.axhline(res["summary"]["A_full"]["mean"], ls="--", color="gray",
                        label="A full (mean)")
        ax2.set_xticks([0, 1]); ax2.set_xticklabels(["Cayley (B)", "Random (C)"])
        ax2.set_xlim(-0.6, 1.6)
        ax2.set_ylabel("best val loss (nats)")
        ax2.set_title("Best val loss per seed")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=130)
    print(f"[plot] wrote {PLOT_PATH}  (B gap={b_gap:.3f}, C gaps={['%.3f'%g for g in c_gaps]})")


if __name__ == "__main__":
    main()
