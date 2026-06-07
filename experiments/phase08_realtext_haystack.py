"""
Phase 08 — Does the fixed 100-edge spread graph still match full attention when the haystack is
            REAL LANGUAGE? (controlled recall needle in a WikiText-103 haystack)

Phase 07 showed a 100-edge SPREAD graph (2-hop reach to D=256) matches full attention on a synthetic
MQAR needle at ~13× edge saving, construction-agnostic, while a same-budget BUNCHED window (3-hop) also
solves but ~7× slower. The honest next question (README phase-08): does that survive when the context
the route traverses is natural language rather than synthetic noise?

Why not plain LM on real text: phases 03/03b already paid for that — on WikiText-103 a w=128 window
matches full attention, so long-range edges carry no load and any null is uninterpretable. So this
phase keeps the dependency CONTROLLED (a recall needle provably D=256 past the window) and changes ONE
variable: the filler positions the route passes through become real WikiText tokens instead of PAD.

  Same as Phase 07: D=256, T=1320, L=6, four arms at a fixed 100-edge budget, identical reach guard
  (C_window100 = 3 hops, B_expander100 / B_random100 = 2 hops, no direct 256-edge), 40k ceiling,
  stop-on-solve, flat-stop OFF.
  Changed from Phase 07: the ~1200 non-needle, non-distractor positions are filled with a real
  WikiText-103 window (GPT-2 BPE ids 0..50256). The 48 reserved distractor key→value pairs remain
  (so the model must still key-match, not just copy the lone reserved value). Vocab gains the real BPE
  block; markers (QUERY / keys / values) live at reserved ids ≥ 50257. The head is applied only at the
  query position (full T×50k logits would OOM).

Question: does B_expander100 / B_random100 still match A_full through a real-language haystack, and
does C_window100's 3-hop route slow down or break when intermediate relay nodes hold real text? If the
2-hop spread graph still matches full attention, Phase 07's viability transfers toward real language;
if real-text relay nodes disrupt routing, that is the first crack and bounds the claim.

Run:  ~/Code/HRS/.venv/bin/python experiments/phase08_realtext_haystack.py
Knobs: P8_CEILING / P8_EVAL_EVERY / P8_EVAL_SEQS / P8_SEEDS ; P8_GUARD_ONLY=1 reach guard only;
       P8_SMOKE=1 tiny end-to-end; P8_SKIP_SANITY=1 skips the D_in=8 gate (debug).
Resumable: completed (phase, arm, seed) runs in phase08_records.json are skipped.
"""
import os, sys, json, time, math, collections
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

REPO       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.environ.get("P8_RESULT_DIR") or os.path.join(REPO, "results", "phase08")
DATA_DIR   = os.path.join(REPO, "data", "wikitext103")
JSON_PATH  = os.path.join(RESULT_DIR, "phase08_records.json")
PLOT_PATH  = os.path.join(RESULT_DIR, "phase08_recall.png")
os.makedirs(RESULT_DIR, exist_ok=True)

# ---- vocab: real BPE text block + reserved markers ----
TEXT_VOCAB = 50257                 # GPT-2 BPE ids 0..50256 = real WikiText
K, V = 64, 64
QUERY = TEXT_VOCAB                  # 50257
KEY0  = TEXT_VOCAB + 1             # 50258 .. 50321
VAL0  = TEXT_VOCAB + 1 + K         # 50322 .. 50385
VOCAB = TEXT_VOCAB + 1 + K + V     # 50386

# ---- geometry / model (identical to Phase 07) ----
T = 1320
D = 256
D_IN = 8
DEGREE = 100
N_LAYER, N_HEAD, N_EMBD, DROPOUT = 6, 8, 512, 0.0

def _envint(name, default):
    v = os.environ.get(name); return int(v) if v else default
def _envflt(name, default):
    v = os.environ.get(name); return float(v) if v else default

SMOKE        = os.environ.get("P8_SMOKE") == "1"
BATCH        = _envint("P8_BATCH", 64)
LR           = 1e-3
WARMUP       = 200
WEIGHT_DECAY = 0.1
BETA1, BETA2 = 0.9, 0.95
GRAD_CLIP    = 1.0
EVAL_EVERY   = _envint("P8_EVAL_EVERY", 500)
EVAL_SEQS    = _envint("P8_EVAL_SEQS", 2048)
EVAL_SEED    = 9999
MIN_STEPS    = _envint("P8_MIN_STEPS", 3000)
CEILING      = _envint("P8_CEILING", 200 if SMOKE else 40000)
SANITY_CEIL  = _envint("P8_SANITY_CEIL", 200 if SMOKE else 8000)
SOLVE_RECALL = _envflt("P8_SOLVE_RECALL", 0.90)
SEEDS        = [int(x) for x in os.environ.get("P8_SEEDS", "1337,1338,1339,1340").split(",")]
GUARD_ONLY   = os.environ.get("P8_GUARD_ONLY") == "1"
SKIP_SANITY  = os.environ.get("P8_SKIP_SANITY") == "1"
RANDOM_OFFSET_SEED = 20260606

ARMS = ["A_full", "C_window100", "B_expander100", "B_random100"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE  = torch.bfloat16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32

# --------------------------------------------------------------------------------------
# Offset sets + masks + reach guard (IDENTICAL to Phase 07)
# --------------------------------------------------------------------------------------
def window_offsets():  return list(range(1, DEGREE + 1))
def expander_offsets():
    small = list(range(1, 9)); n_spread = DEGREE - len(small)
    spread = sorted(set(int(round(8 + (D - 11) * k / n_spread)) for k in range(1, n_spread + 1)))
    offs = sorted(set(small + spread)); o = D - 4
    while len(offs) < DEGREE:
        if o not in offs and o != D: offs.append(o)
        o -= 1
    return sorted(set(offs))[:DEGREE]
def random_offsets(seed):
    rng = np.random.default_rng(RANDOM_OFFSET_SEED + seed)
    small = list(range(1, 9)); pool = [x for x in range(9, D + 1) if x != D]
    extra = rng.choice(pool, size=DEGREE - len(small), replace=False).tolist()
    return sorted(set(small + [int(x) for x in extra]))
def offsets_for(arm, seed):
    if arm == "A_full":         return None
    if arm == "C_window100":    return window_offsets()
    if arm == "B_expander100":  return expander_offsets()
    if arm == "B_random100":    return random_offsets(seed)
    raise ValueError(arm)

def build_mask(arm, seed):
    allow = np.zeros((T, T), dtype=bool); r = np.arange(T)
    if arm == "A_full":
        allow = np.tril(np.ones((T, T), dtype=bool))
    else:
        allow[r, r] = True
        for o in offsets_for(arm, seed):
            idx = np.arange(o, T); allow[idx, idx - o] = True
    edges = int(allow.sum())
    add = np.zeros((T, T), dtype=np.float32); add[~allow] = float("-inf")
    return torch.from_numpy(add)[None, None].to(DEVICE, DTYPE), edges

INF = 1 << 30
def shortest_hops_offsets(q, target, offs):
    dist = np.full(q + 1, INF, dtype=np.int64); dist[q] = 0
    dq = collections.deque([q])
    while dq:
        i = dq.popleft(); di = dist[i]
        for o in offs:
            j = i - o
            if j >= 0 and di + 1 < dist[j]: dist[j] = di + 1; dq.append(j)
    return int(dist[target])
def reach_guard(arm, seed, dist=D, n_samples=400):
    if arm == "A_full":
        return {"arm": arm, "n_offsets": None, "has_direct_edge": True, "reach_frac": 1.0,
                "hop_min": 1, "hop_med": 1, "hop_max": 1, "frac_le2": None}
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
# Real-text haystack generator: WikiText window + planted needle + 48 distractor pairs
# --------------------------------------------------------------------------------------
N_PAIRS = 48
def _load_text(split):
    return np.memmap(os.path.join(DATA_DIR, f"{split}.bin"), dtype=np.uint16, mode="r")

def make_batch(B, Dd, rng, text):
    """Each sequence: a real WikiText-103 window (BPE ids), with a planted recall needle at distance
    Dd and 48 reserved distractor key→value pairs overwriting text at scattered even positions."""
    ntext = len(text)
    toks = np.zeros((B, T), dtype=np.int64)
    tpos = np.zeros(B, dtype=np.int64); ttok = np.zeros(B, dtype=np.int64)
    for b in range(B):
        st = int(rng.integers(0, ntext - T - 1))
        toks[b] = np.asarray(text[st:st + T], dtype=np.int64)        # real-text haystack
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
        toks[b, slots] = KEY0 + distractor_keys[:npos]               # overwrite text with distractor pairs
        toks[b, slots + 1] = VAL0 + dvals
        toks[b, nk] = KEY0 + kq; toks[b, nv] = VAL0 + vq             # the needle pair
        toks[b, qk - 1] = QUERY; toks[b, qk] = KEY0 + kq; toks[b, qk + 1] = VAL0 + vq
        tpos[b] = qk; ttok[b] = VAL0 + vq
    return toks, tpos, ttok

def heldout(Dd, text):
    return make_batch(EVAL_SEQS, Dd, np.random.default_rng(EVAL_SEED + Dd), text)

# --------------------------------------------------------------------------------------
# Model (Phase 04–07 model; forward returns hidden — head applied only at the query position)
# --------------------------------------------------------------------------------------
def _build_rope(Tn, hd, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, hd, 2).float() / hd))
    freqs = torch.outer(torch.arange(Tn).float(), inv)
    emb = torch.cat((freqs, freqs), dim=-1); return emb.cos(), emb.sin()
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
        x = x + self.proj(F.gelu(self.fc(self.ln2(x)))); return x

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
        return self.ln_f(x)                     # hidden states; head applied by caller at one position
    def logits_at(self, h, idx, pos):
        return F.linear(h[idx, pos], self.head.weight)   # (B, VOCAB) — avoids the full T×VOCAB tensor
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
            h = model(xb)
        idx = torch.arange(xb.size(0), device=DEVICE)
        lg = model.logits_at(h, idx, pb).float()
        correct += (lg.argmax(-1) == kb).sum().item()
        ce_sum += F.cross_entropy(lg, kb, reduction="sum").item()
    model.train()
    return correct / nseq, ce_sum / nseq

def train_one(arm, seed, Dd, ceiling, text):
    mask, edges = build_mask(arm, seed)
    torch.manual_seed(seed)
    model = GPT(mask).to(DEVICE); opt = configure_opt(model)
    rng = np.random.default_rng(seed * 100003 + Dd)
    ev = heldout(Dd, text)
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
            else: solve_cand = None
        for g in opt.param_groups: g["lr"] = get_lr(step)
        toks, tpos, ttok = make_batch(BATCH, Dd, rng, text)
        xb = torch.from_numpy(toks).to(DEVICE); pb = torch.from_numpy(tpos).to(DEVICE); kb = torch.from_numpy(ttok).to(DEVICE)
        with torch.autocast(device_type="cuda", dtype=DTYPE):
            h = model(xb)
        idx = torch.arange(BATCH, device=DEVICE)
        loss = F.cross_entropy(model.logits_at(h, idx, pb).float(), kb)
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
            "edges_total": edges, "curve": curve, "peak_gb": round(peak, 2),
            "params": int(npar), "wall_s": round(time.time() - t0, 1)}

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
        solved = [r for r in rk if r["solved"]]; sts = sorted(r["steps_to_solve"] for r in solved)
        summ[arm] = {"n_seeds": len(rk), "n_solved": len(solved),
                     "frac_solving": round(len(solved) / len(rk), 4),
                     "recall_mean": round(float(np.mean([r["recall"] for r in rk])), 4),
                     "recall_median": round(float(np.median([r["recall"] for r in rk])), 4),
                     "steps_to_solve_median": int(np.median(sts)) if sts else None,
                     "recalls": [round(r["recall"], 4) for r in rk]}
    res["summary"] = summ

def main():
    guard = {arm: reach_guard(arm, SEEDS[0]) for arm in ARMS}
    guard["B_random100_per_seed"] = {f"s{s}": reach_guard("B_random100", s) for s in SEEDS}
    print("===== REACH GUARD (query→needle hops @ D=256; identical to Phase 07) =====")
    for arm in ARMS:
        g = guard[arm]
        print(f"  {arm:14s} offsets={g['n_offsets']} direct={g['has_direct_edge']} reach={g['reach_frac']} "
              f"hops=[{g.get('hop_min')},{g.get('hop_med')},{g.get('hop_max')}] frac≤2={g.get('frac_le2')}")
    cw, be = guard["C_window100"], guard["B_expander100"]
    ok = (be["hop_max"] <= 2 and not be["has_direct_edge"] and be["reach_frac"] == 1.0 and cw["hop_med"] == 3)
    print(f"  GUARD: {'OK' if ok else 'FAIL'}")
    if not ok: print("[abort] reach guard failed."); sys.exit(1)
    if GUARD_ONLY: print("[guard-only] done."); return

    text_tr = _load_text("train"); text_ev = _load_text("val")
    print(f"[data] train={len(text_tr):,} tok  val={len(text_ev):,} tok  (GPT-2 BPE; markers at ≥{TEXT_VOCAB})")

    res = load_results()
    res["config"] = {"task": "MQAR recall through a real WikiText-103 haystack (controlled D)",
        "T": T, "D": D, "D_in": D_IN, "degree": DEGREE, "vocab": VOCAB, "text_vocab": TEXT_VOCAB,
        "K": K, "V": V, "n_distractor_pairs": N_PAIRS, "chance_recall": round(1 / V, 4),
        "n_layer": N_LAYER, "n_head": N_HEAD, "n_embd": N_EMBD, "batch": BATCH, "lr": LR,
        "ceiling": CEILING, "solve_recall": SOLVE_RECALL, "flat_stop": False, "seeds": SEEDS,
        "arms": ARMS, "dataset": "wikitext-103-raw-v1 (GPT-2 BPE)", "device": DEVICE, "dtype": str(DTYPE),
        "full_edges": T * (T + 1) // 2, "saving_x": round((T * (T + 1) // 2) / (T * DEGREE), 1)}
    res["guard"] = guard; save_results(res)

    if not SKIP_SANITY:
        print("\n===== SANITY GATE (D_in=8 through real text; every arm ~1.00) =====")
        for arm in ARMS:
            sk = f"Din{D_IN}::{arm}::s{SEEDS[0]}"
            if sk in res["sanity"]:
                print(f"  [skip] {sk} recall={res['sanity'][sk]['recall']:.3f}"); continue
            out = train_one(arm, SEEDS[0], D_IN, SANITY_CEIL, text_tr)
            res["sanity"][sk] = out; save_results(res)
            ok_s = out["recall"] >= 0.95
            print(f"  {arm:14s} recall={out['recall']:.3f}  {'OK' if ok_s else 'FAIL'}")
            if not ok_s and arm in ("A_full", "C_window100"):
                print(f"[abort] sanity failed for {arm} — task/training broken through real text, not reach."); sys.exit(1)

    print(f"\n===== MAIN SWEEP (D={D}, L={N_LAYER}, real-text haystack, ceiling={CEILING}) =====")
    for arm, seed in [(a, s) for a in ARMS for s in SEEDS]:
        rk = f"D{D}::{arm}::s{seed}"
        if rk in res["runs"]:
            r = res["runs"][rk]; print(f"[skip] {rk} solved={r['solved']} recall={r['recall']:.4f}"); continue
        print(f"[run] {rk}", flush=True)
        out = train_one(arm, seed, D, CEILING, text_tr)
        res["runs"][rk] = out; aggregate(res); save_results(res)
        print(f"[done] {rk} solved={out['solved']} sts={out['steps_to_solve']} recall={out['recall']:.4f} "
              f"stop={out['stop_reason']}@{out['stop_step']} ({out['wall_s']}s)", flush=True)

    aggregate(res); save_results(res)
    try: make_plot(res)
    except Exception as e: print(f"[plot] skipped: {e}")
    print("\n===== SUMMARY (recall @ D=256 through WikiText haystack) =====")
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
    fig, ax = plt.subplots(figsize=(9, 5.2)); chance = res["config"]["chance_recall"]
    for i, a in enumerate(arms):
        for r in s[a]["recalls"]:
            ax.plot(i, r, "o", color=colors.get(a, "#555"), alpha=0.6, ms=8, zorder=3)
        ax.plot(i, s[a]["recall_median"], "_", color="black", ms=26, mew=2.5, zorder=4)
        ax.text(i, 1.04, f"{s[a]['n_solved']}/{s[a]['n_seeds']}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(chance, ls=":", color="#888", lw=1.2, label=f"chance ({chance:.3f})")
    ax.set_xticks(range(len(arms))); ax.set_xticklabels(arms, rotation=12)
    ax.set_ylim(-0.04, 1.12); ax.set_ylabel("recall @ D=256 (real-text haystack)")
    sv = res["config"]["saving_x"]
    ax.set_title(f"Phase 08 — fixed {DEGREE}-edge budget through a WikiText haystack (~{sv}× saving)\n(— = median; n/n = seeds solving)")
    ax.legend(loc="center right", fontsize=9)
    fig.tight_layout(); fig.savefig(PLOT_PATH, dpi=130); plt.close(fig); print("[plot] wrote", PLOT_PATH)

if __name__ == "__main__":
    main()
