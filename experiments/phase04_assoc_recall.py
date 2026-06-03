"""
Phase 04 — Synthetic associative recall: does a long-range edge earn its place when the
           dependency is *provably* past the window?

Phase 03b's null was uninterpretable: WikiText-103 dependencies are mostly local (a
128-token window matched full attention), so the long-range edges "had nothing to do" —
indistinguishable from "edges are useless." This phase removes that confound by
construction: a (key, value) needle is placed at a CHOSEN distance D before the query, so
the answer lives at a known supra-window distance and a long-range edge has a real job.

Task — multi-key associative recall (MQAR-style), procedurally generated (no stored corpus):
  [ distractor key/value bigrams ... NEEDLE(key_q,value_q) ... more distractors ]
  QUERY  key_q  value_q
  predict value_q at the key_q position (loss/recall measured there only).
The needle's VALUE sits exactly D tokens before the query-key position. The query position
is RANDOMIZED per sequence (only D is fixed), so the answer cannot be read off absolute
position — only content (matching key_q) identifies the needle. key_q is unique among keys.

Distance regimes (window w=128 fixed):
  D_in  = 64   -> needle INSIDE the window. Sanity: every arm, incl. window-only, must recall.
  D_out = 768  -> needle WELL beyond the window. The discriminating regime.

Arms (only the attention mask differs):
  A_full     full causal attention                                  (ceiling)
  C_window   causal window w=128                                    (floor: must fail at D_out)
  B_rand     window + 8 uniform-random long edges/node              (the phase-03 expander)
  B_dilated  window + 8 dilated edges at offsets {128..1024}        (a *well-placed* long edge)
  B_cayley   window + 8-regular scaled-Cayley edges                 (the named construction)

The discriminating axis is EDGE PLACEMENT. Direct-reach analysis (see placement guard):
dilated has an edge at offset exactly 768 (reaches the needle from every query node);
Cayley's scaled offsets are scattered and hit 768 for ~0% of query nodes; random ~1%. So
B_dilated can route the dependency in one hop; B_rand/B_cayley can only via multi-hop
composition over layers (n_layer=8). This separates "long-range edges are useless" from
"phase-03's edges were aimed wrong."

Primary metric: RECALL (argmax at the value position == value_q), chance = 1/V ~ 1.5%. The
signal is large and unambiguous — a reachable arm ~100%, an unreachable one ~chance.

Convergence guard (avoid 03b's undertraining trap): train each arm to a recall PLATEAU
(< 0.5% improvement over a 2k-step window), hard ceiling 30k steps; log the recall curve so
"did not converge" is distinguishable from "cannot reach".

Run:  ~/Code/HRS/.venv/bin/python experiments/phase04_assoc_recall.py
Resumable: completed (arm, D, seed) runs in phase04_records.json are skipped.
"""
import os, sys, json, time, math, itertools
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(REPO, "results", "phase04")
JSON_PATH  = os.path.join(RESULT_DIR, "phase04_records.json")
PLOT_PATH  = os.path.join(RESULT_DIR, "phase04_recall.png")
os.makedirs(RESULT_DIR, exist_ok=True)

# ---- task vocab ----
K, V = 64, 64
PAD, QUERY = 0, 1
KEY0 = 2                    # keys:   2 .. 2+K-1
VAL0 = 2 + K               # values: 2+K .. 2+K+V-1
VOCAB = 2 + K + V          # 130

# ---- geometry ----
T = 1024
WINDOW = 128
D_IN, D_OUT = 64, 768
DILATED_OFFS = [128 * k for k in range(1, 9)]   # {128,256,...,1024}; contains D_out=768
LR_K = 8

# ---- model (moderate; deep enough for induction + multi-hop composition) ----
N_LAYER, N_HEAD, N_EMBD, DROPOUT = 8, 8, 512, 0.0

# ---- training (recall-plateau stopping) ----
BATCH        = 64
LR           = 1e-3
WARMUP       = 200
WEIGHT_DECAY = 0.1
BETA1, BETA2 = 0.9, 0.95
GRAD_CLIP    = 1.0
EVAL_EVERY   = 500
EVAL_SEQS    = 2048
EVAL_SEED    = 9999
PLATEAU_DELTA= 0.005       # < 0.5% improvement ...
PLATEAU_WIN  = 2000        # ... over a 2k-step window
MIN_STEPS    = 3000
MAX_STEPS    = 12000
FAIL_CUTOFF  = 9000        # if still < FAIL_RECALL by here, it never left chance -> stop
FAIL_RECALL  = 0.05

ARMS = ["A_full", "C_window", "B_rand", "B_dilated", "B_cayley"]
ARM_DESC = {
    "A_full":   "full causal attention (ceiling)",
    "C_window":  f"causal window w={WINDOW} (floor)",
    "B_rand":   f"window + {LR_K} uniform-random long edges/node",
    "B_dilated": f"window + {LR_K} dilated edges at offsets {DILATED_OFFS}",
    "B_cayley":  f"window + {LR_K}-regular scaled-Cayley edges",
}
SEEDS = [1337, 1338]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

# --------------------------------------------------------------------------------------
# Synthetic generator
# --------------------------------------------------------------------------------------
N_PAIRS = 48   # distinct (key,value) bigrams scattered through the context (standard MQAR density)

def make_batch(B, D, rng):
    """Returns toks (B,T) int64, tpos (B,) int64, ttok (B,) int64.
    N_PAIRS distinct-key bigrams are scattered at random even positions in [0, qk-1); one of
    them is the needle, placed so its VALUE sits exactly D tokens before the query-key
    position qk (randomized per sequence). key_q is unique. Predict value_q at logits[qk]."""
    toks = np.zeros((B, T), dtype=np.int64)
    tpos = np.zeros(B, dtype=np.int64)
    ttok = np.zeros(B, dtype=np.int64)
    for b in range(B):
        qk = int(rng.integers(D + 1, T - 1))            # query-key position, odd
        if qk % 2 == 0:
            qk = qk - 1 if qk - 1 >= D + 1 else qk + 1
        nk, nv = qk - D - 1, qk - D                     # needle key (even), value (odd)
        # distinct keys: kq plus N_PAIRS-1 distractor keys (all distinct)
        keyset = rng.permutation(K)[:N_PAIRS]
        kq = int(keyset[0]); distractor_keys = keyset[1:]
        vq = int(rng.integers(0, V))
        # scatter distractor key-slots at random even positions in [0, qk-1), excluding needle
        even = np.arange(0, qk - 1, 2)
        even = even[even != nk]
        npos = min(len(distractor_keys), len(even))
        slots = rng.choice(even, size=npos, replace=False)
        dvals = rng.integers(0, V, size=npos)
        toks[b, slots] = KEY0 + distractor_keys[:npos]
        toks[b, slots + 1] = VAL0 + dvals
        toks[b, nk] = KEY0 + kq                          # needle
        toks[b, nv] = VAL0 + vq
        toks[b, qk - 1] = QUERY
        toks[b, qk]     = KEY0 + kq                      # query key
        toks[b, qk + 1] = VAL0 + vq                      # teacher-forced target token
        tpos[b] = qk
        ttok[b] = VAL0 + vq
    return toks, tpos, ttok

def heldout(D):
    rng = np.random.default_rng(EVAL_SEED + D)
    return make_batch(EVAL_SEQS, D, rng)

# --------------------------------------------------------------------------------------
# Masks (additive -inf bias) + placement guard
# --------------------------------------------------------------------------------------
def _sl2_elems(n):
    return [(a, b, c, d) for a, b, c, d in itertools.product(range(n), repeat=4)
            if (a * d - b * c) % n == 1]

def cayley_adjacency8(n):
    elems = _sl2_elems(n); idx = {e: i for i, e in enumerate(elems)}
    def mul(x, y):
        a, b, c, d = x; e, f, g, h = y
        return ((a*e+b*g) % n, (a*f+b*h) % n, (c*e+d*g) % n, (c*f+d*h) % n)
    a = (1, 1, 0, 1); b = (1, 0, 1, 1)
    gens = [a, (1, n-1, 0, 1), b, (1, 0, n-1, 1),
            (1, 2 % n, 0, 1), (1, (n-2) % n, 0, 1), (1, 0, 2 % n, 1), (1, 0, (n-2) % n, 1)]
    N = len(elems); adj = [[] for _ in range(N)]
    for i, e in enumerate(elems):
        for g in gens:
            j = idx[mul(e, g)]
            if j != i and j not in adj[i]:
                adj[i].append(j)
    return N, adj

def cayley_scaled_fn(T):
    # smallest |SL2(Z_n)| >= T ; n=17 -> 4896 covers T=1024
    Nc, n = 4896, 17
    _, adj = cayley_adjacency8(n)
    def fn(i):
        node = (i * Nc) // T; out = []
        for e in adj[node]:
            p = (e * T) // Nc
            if p != i and p not in out:
                out.append(p)
        return out
    return fn

def longrange_fn_for(arm, seed):
    if arm in ("A_full", "C_window"):
        return None
    if arm == "B_rand":
        rng = np.random.default_rng(20260602 + seed)
        chosen = [rng.integers(0, i + 1, size=LR_K).tolist() for i in range(T)]
        return lambda i: chosen[i]
    if arm == "B_dilated":
        return lambda i: [i - off for off in DILATED_OFFS if i - off >= 0]
    if arm == "B_cayley":
        return cayley_scaled_fn(T)
    raise ValueError(arm)

def build_mask(arm, seed):
    """Returns (additive_mask or None, edges, longrange_fn). None => full causal."""
    if arm == "A_full":
        return None, T * (T + 1) // 2, None
    window = WINDOW
    lr_fn = longrange_fn_for(arm, seed)
    allow = np.zeros((T, T), dtype=bool)
    r = np.arange(T); allow[r, r] = True
    for i in range(T):
        allow[i, max(0, i - window):i + 1] = True
    if lr_fn is not None:
        for i in range(T):
            for j in lr_fn(i):
                if j <= i:
                    allow[i, j] = True
    edges = int(allow.sum())
    add = np.zeros((T, T), dtype=np.float32); add[~allow] = float("-inf")
    return torch.from_numpy(add)[None, None].to(DEVICE, DTYPE), edges, lr_fn

def direct_reach_fraction(arm, seed, D):
    """Fraction of valid query nodes q in [D+1,T-2] that can read the needle value (at q-D)
    via a one-hop edge: window if D<=w, else a long-range edge at offset exactly D."""
    if arm == "A_full":
        return 1.0
    if D <= WINDOW:
        return 1.0
    lr_fn = longrange_fn_for(arm, seed)
    if lr_fn is None:
        return 0.0
    hit = tot = 0
    for q in range(D + 1, T - 1):
        tot += 1
        if any(q - j == D for j in lr_fn(q) if j <= q):
            hit += 1
    return hit / tot

# --------------------------------------------------------------------------------------
# Model (SDPA: full=is_causal, sparse=additive mask via efficient backend)
# --------------------------------------------------------------------------------------
# Rotary position embeddings (RoPE) — relative position; required for the model to learn
# content-based associative recall. (Absolute learned pos-emb provably fails here: full
# attention stays at chance — see phase04_README diagnostic.)
def _build_rope(Tn, hd, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    freqs = torch.outer(torch.arange(Tn).float(), inv)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos(), emb.sin()   # (Tn, hd)

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
        self.nh = N_HEAD; self.causal = add_mask is None
        self.register_buffer("rope_cos", _ROPE_COS[None, None], persistent=False)
        self.register_buffer("rope_sin", _ROPE_SIN[None, None], persistent=False)
        if add_mask is not None:
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
        if self.causal:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
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
        self.tok_emb = nn.Embedding(VOCAB, N_EMBD)   # position handled by RoPE in attention
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
        lg = logits[idx, pb]                    # (b, vocab) at the query position
        correct += (lg.argmax(-1) == kb).sum().item()
        ce_sum += F.cross_entropy(lg.float(), kb, reduction="sum").item()
    model.train()
    return correct / nseq, ce_sum / nseq

def train_one(arm, D, seed):
    mask, edges, _ = build_mask(arm, seed)
    reach = direct_reach_fraction(arm, seed, D)
    torch.manual_seed(seed)
    model = GPT(mask).to(DEVICE)
    opt = configure_opt(model)
    rng = np.random.default_rng(seed * 100003 + D)
    ev = heldout(D)
    curve = []   # (step, recall)
    best = 0.0; best_step = 0; t0 = time.time()
    step = 0
    while step <= MAX_STEPS:
        if step % EVAL_EVERY == 0:
            rec, ce = eval_recall(model, ev)
            curve.append((step, round(rec, 4)))
            best = max(best, rec); best_step = step if rec >= best else best_step
            print(f"  [{arm} D{D} s{seed}] step {step:5d}  recall {rec:.4f}  ce {ce:.3f}  "
                  f"best {best:.4f}  ({time.time()-t0:.0f}s)", flush=True)
            # stop conditions:
            #  (a) confirmed floor — never left chance by FAIL_CUTOFF
            if step >= FAIL_CUTOFF and best < FAIL_RECALL:
                break
            #  (b) plateau — for an arm that DID learn (best above chance), best in the
            #      last PLATEAU_WIN improved < PLATEAU_DELTA
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
            "steps_to_plateau": step, "direct_reach_frac": round(reach, 4),
            "edges_total": edges, "curve": curve, "peak_gb": round(peak, 2),
            "params": int(npar), "wall_s": round(time.time() - t0, 1)}

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
        for D in (D_IN, D_OUT):
            recs = [res["runs"][f"T{T}::{arm}::w{WINDOW}::D{D}::{s}"]["recall"]
                    for s in SEEDS if f"T{T}::{arm}::w{WINDOW}::D{D}::{s}" in res["runs"]]
            if recs:
                summ[f"{arm}::D{D}"] = {"n": len(recs), "recall_mean": float(np.mean(recs)),
                                        "recall_min": float(np.min(recs)), "recall_max": float(np.max(recs))}
    res["summary"] = summ
    return summ

def main():
    print(f"[env] device={DEVICE} dtype={DTYPE}  vocab={VOCAB}  T={T} w={WINDOW}")
    # placement guard
    print("[guard] direct one-hop reach of the needle (offset == D) per arm:")
    guard = {}
    for arm in ARMS:
        for D in (D_IN, D_OUT):
            fr = direct_reach_fraction(arm, 1337, D)
            guard[f"{arm}::D{D}"] = fr
            print(f"  {arm:10s} D={D:4d}: {fr*100:5.1f}% of query nodes")
    assert guard[f"B_dilated::D{D_OUT}"] > 0.99, "B_dilated must directly reach D_out — guard failed"
    print(f"[guard] OK: B_dilated reaches D_out={D_OUT} directly ({guard[f'B_dilated::D{D_OUT}']*100:.0f}%); "
          f"B_rand/B_cayley do not (must compose over {N_LAYER} layers).")

    res = load_results()
    res["config"] = {
        "task": "MQAR associative recall", "vocab": VOCAB, "K": K, "V": V, "T": T,
        "window": WINDOW, "D_in": D_IN, "D_out": D_OUT, "dilated_offsets": DILATED_OFFS,
        "lr_k": LR_K, "model": {"n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD, "dropout": DROPOUT},
        "batch": BATCH, "lr": LR, "max_steps": MAX_STEPS, "plateau_delta": PLATEAU_DELTA,
        "plateau_win": PLATEAU_WIN, "eval_seqs": EVAL_SEQS, "chance": round(1 / V, 4),
        "arms": ARM_DESC, "direct_reach_guard": guard, "device": DEVICE, "dtype": str(DTYPE),
    }
    save_results(res)

    # order: sanity (D_in) then headline (D_out), seed 1337 first
    plan = [(arm, D, s) for s in SEEDS for D in (D_IN, D_OUT) for arm in ARMS]
    for arm, D, seed in plan:
        rk = f"T{T}::{arm}::w{WINDOW}::D{D}::{seed}"
        if rk in res["runs"]:
            print(f"[skip] {rk} recall={res['runs'][rk]['recall']:.4f}")
            continue
        print(f"[run] {rk}", flush=True)
        out = train_one(arm, D, seed)
        res["runs"][rk] = out
        aggregate(res); save_results(res)
        print(f"[done] {rk} recall={out['recall']:.4f} reach={out['direct_reach_frac']} "
              f"steps={out['steps_to_plateau']} ({out['wall_s']}s)", flush=True)

    aggregate(res); save_results(res)
    try:
        make_plot(res)
    except Exception as e:
        print(f"[plot] skipped: {e}")
    print("\n===== SUMMARY (recall) =====")
    for arm in ARMS:
        cells = []
        for D in (D_IN, D_OUT):
            s = res["summary"].get(f"{arm}::D{D}")
            cells.append(f"D{D}={s['recall_mean']:.3f}" if s else f"D{D}=--")
        print(f"  {arm:10s} {'  '.join(cells)}")
    print("Wrote", JSON_PATH)

def make_plot(res):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = res["summary"]
    x = np.arange(len(ARMS)); wdt = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for k, (D, col) in enumerate([(D_IN, "#7aa6c2"), (D_OUT, "#d1495b")]):
        ys = [s.get(f"{a}::D{D}", {}).get("recall_mean", np.nan) for a in ARMS]
        ax.bar(x + (k - 0.5) * wdt, ys, wdt, label=f"D={D} ({'inside' if D<=WINDOW else 'beyond'} window)",
               color=col)
        for xi, yi in zip(x + (k - 0.5) * wdt, ys):
            if not np.isnan(yi):
                ax.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(1 / V, ls="--", color="gray", lw=1, label=f"chance (1/V={1/V:.3f})")
    ax.set_xticks(x); ax.set_xticklabels(ARMS, fontsize=9)
    ax.set_ylabel("recall accuracy"); ax.set_ylim(0, 1.08)
    ax.set_title(f"Associative recall by arm (T={T}, w={WINDOW}): needle inside vs beyond window")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(PLOT_PATH, dpi=130)
    print(f"[plot] wrote {PLOT_PATH}")

if __name__ == "__main__":
    main()
