#!/usr/bin/env python3
"""
Edge Server Placement - Custom Extension I (convex, M/M/1) and II (non-convex, concave capacity cost).
Single self-contained file for Aatraya's part. Server cities are DECISION VARIABLES: they come out of the optimisation.

Custom I  (convex)      min  sum vol*tau*w + alpha * sum_e f_e/(C_e - f_e)   s.t. f_e <= util_cap*C_e, CapEx <= B
  c1-socp : CVXPY SOCP relaxation -> top-K cities -> integer modules -> exact M/M/1 flow refinement
            (--mip solves y binary / z integer in one shot; needs SCIP, GUROBI or MOSEK)
  c1-pwl  : exact integer MILP, M/M/1 term replaced by tangent cuts per module level (HiGHS)
Custom II (non-convex)  CapEx = c_server*sum y + sum_e alpha_e*z_e^gamma <= B,  0 < gamma < 1
  c2-sos2 : piecewise-linear concave cost with SOS2 binaries -> MILP (HiGHS), global up to the segment error
  c2-ipopt: non-convex NLP solved with IPOPT (CasADi), multi-start, then top-K rounding and polish

Install : pip install numpy scipy networkx cvxpy casadi
Run     : python track4_5_custom_extensions.py data/janos-us.txt
          python track4_5_custom_extensions.py data/germany50.json --methods c1-socp c2-ipopt --out results_germany.json
Inputs  : SNDlib native .txt (links with module costs) or node-link .json (module size/cost then assumed, see flags).
"""
import argparse, json, math, re, sys, time
from itertools import islice
import numpy as np
import networkx as nx

# ======================================================================= data
def hav(a, b):
    (lo1, la1), (lo2, la2) = a, b
    p1, p2 = math.radians(la1), math.radians(la2)
    x = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2
    return 6371.0 * 2 * math.asin(math.sqrt(x))

def parse_sndlib(path):
    sec, nodes, links, demands, n_paths = None, {}, [], [], 0
    for line in open(path, encoding="utf-8", errors="replace"):
        s = line.strip()
        m = re.match(r"^(NODES|LINKS|DEMANDS|ADMISSIBLE_PATHS)\s*\(", s)
        if m: sec = m.group(1); continue
        if s.startswith(")"): sec = None; continue
        if not s or s[0] in "#?": continue
        if sec == "NODES":
            m = re.match(r"^(\S+)\s*\(\s*(-?[\d.]+)\s+(-?[\d.]+)\s*\)", s); nodes[m.group(1)] = (float(m.group(2)), float(m.group(3)))
        elif sec == "LINKS":
            m = re.match(r"^(\S+)\s*\(\s*(\S+)\s+(\S+)\s*\)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*\(([^)]*)\)", s)
            mods = [float(x) for x in m.group(8).split()]
            links.append(dict(src=m.group(2), dst=m.group(3), pre_cap=float(m.group(4)), mod_cap=mods[0], mod_cost=mods[1], n_modules=len(mods) // 2))
        elif sec == "DEMANDS":
            m = re.match(r"^(\S+)\s*\(\s*(\S+)\s+(\S+)\s*\)\s+\d+\s+([\d.]+)", s); demands.append((m.group(2), m.group(3), float(m.group(4))))
        elif sec == "ADMISSIBLE_PATHS": n_paths += 1
    return dict(nodes=nodes, links=links, demands=demands, n_paths=n_paths, source="sndlib-native")

def load_json(path, module_gbps, module_cost_per_km):
    d = json.load(open(path)); names = [x["name"] for x in d["nodes"]]; pos = [tuple(x["pos"]) for x in d["nodes"]]
    assert [x["id"] for x in d["nodes"]] == list(range(len(names))) and len(set(names)) == len(names)
    links = []
    for e in d["edges"]:
        u, v = names[e["source"]], names[e["target"]]
        for a, b in ((u, v), (v, u)):                                   # undirected link -> two directed arcs
            links.append(dict(src=a, dst=b, pre_cap=0.0, mod_cap=module_gbps, mod_cost=round(module_cost_per_km * e["dist"], 2), n_modules=1))
    raw = {(int(s), int(t)): float(v) for s, m in d["graph"]["demands"].items() for t, v in m.items()}
    mirrored = not all((t, s) in raw for s, t in raw)
    if mirrored: raw = {**{(t, s): v for (s, t), v in raw.items()}, **raw}      # one direction stored -> add reverse
    st = d["graph"]["stats"]; assert (len(names), len(d["edges"]), len(raw)) == (st["nodes"], st["links"], st["demands"]), "counts differ from file stats"
    return dict(nodes=dict(zip(names, pos)), links=links, demands=[(names[s], names[t], v) for (s, t), v in raw.items()],
                n_paths=0, source="json (module size/cost ASSUMED)", mirrored=mirrored)

def audit(data):
    cost = {(l["src"], l["dst"]): l["mod_cost"] for l in data["links"]}
    caps = sorted({l["mod_cap"] for l in data["links"]})
    assert len(caps) == 1 and {l["n_modules"] for l in data["links"]} == {1}, "code assumes a single module type"
    print(f"data: {data['source']}: {len(data['nodes'])} cities, {len(data['links'])} directed links, {len(data['demands'])} demands, "
          f"module {caps[0]:g} @ {min(cost.values()):.0f}-{max(cost.values()):.0f}, pre-installed {sorted({l['pre_cap'] for l in data['links']})}, "
          f"admissible paths in file: {data['n_paths']}" + (", demands mirrored" if data.get("mirrored") else ""))
    assert all(a in data["nodes"] and b in data["nodes"] for a, b, _ in data["demands"])

class Instance:
    """Arcs from the file, demand summed by source (the model only needs where traffic enters), Yen k-shortest paths."""
    def __init__(self, data, K, B, c_server, demand_scale, us_per_km=5.0, k_paths=3):
        self.names = list(data["nodes"]); self.pos = [data["nodes"][c] for c in self.names]; self.n = n = len(self.names)
        idx = {c: i for i, c in enumerate(self.names)}
        self.K, self.B, self.c_server, self.C_mod = K, B, c_server, data["links"][0]["mod_cap"]
        G = nx.DiGraph(); self.arcs, c_mod, C_base = [], [], []
        for l in data["links"]:
            a, b = idx[l["src"]], idx[l["dst"]]
            G.add_edge(a, b, delay=hav(self.pos[a], self.pos[b]) * us_per_km / 1000.0, eid=len(self.arcs))     # ms
            self.arcs.append((a, b)); c_mod.append(l["mod_cost"]); C_base.append(l["pre_cap"])
        self.c_mod, self.C_base, self.m = np.array(c_mod), np.array(C_base), len(self.arcs)
        self.vol = np.zeros(n)
        for s, t, v in data["demands"]: self.vol[idx[s]] += v * demand_scale                                   # Gbps entering at s
        self.P = []                                                      # (source, target, delay_ms, arc ids)
        for s in range(n):
            for t in range(n):
                if s == t: self.P.append((s, t, 0.0, [])); continue
                for p in islice(nx.shortest_simple_paths(G, s, t, weight="delay"), k_paths):                   # Yen
                    self.P.append((s, t, sum(G[a][b]["delay"] for a, b in zip(p, p[1:])), [G[a][b]["eid"] for a, b in zip(p, p[1:])]))
        self.src = np.array([p[0] for p in self.P]); self.tgt = np.array([p[1] for p in self.P]); self.tau = np.array([p[2] for p in self.P])
        self.A = np.zeros((self.m, len(self.P)))                         # link load f = A @ w  (Gbps)
        for j, (s, _, _, eids) in enumerate(self.P):
            for e in eids: self.A[e, j] = self.vol[s]
        self.by_src = [np.where(self.src == s)[0] for s in range(n)]
    def latency(self, w): return float(self.vol[self.src] * self.tau @ w)             # sum demand*tau*w  (ms*Gbps)
    def city_names(self, y): return [self.names[i] for i in np.where(np.asarray(y) > 0.5)[0]]

def clean(w): return np.where(np.asarray(w) < 1e-6, 0.0, w)

def exact_queue(inst, w, z, alpha):
    f = inst.A @ w; C = inst.C_base + inst.C_mod * z; used = f > 1e-6
    return float(alpha * np.sum(f[used] / (C[used] - f[used])))

def feasibility_estimate(inst):
    """Greedy p-median servers + nearest-server shortest-path routing: a rough CapEx so unit/budget mistakes show up early."""
    best = {}
    for j, (s, t, tau, _) in enumerate(inst.P):
        if (s, t) not in best or tau < inst.P[best[(s, t)]][2]: best[(s, t)] = j
    d = lambda s, t: inst.P[best[(s, t)]][2]; srv = []
    for _ in range(inst.K):
        srv.append(min((t for t in range(inst.n) if t not in srv), key=lambda t: sum(inst.vol[s] * min(d(s, x) for x in srv + [t]) for s in range(inst.n))))
    load = sum(inst.A[:, best[(s, min(srv, key=lambda x: d(s, x)))]] for s in range(inst.n))
    mods = np.ceil(np.maximum(load - inst.C_base, 0) / inst.C_mod - 1e-9)
    est = inst.c_server * inst.K + inst.c_mod @ mods
    print(f"feasibility estimate: about {est:,.0f} CapEx ({int(mods.sum())} modules) vs budget {inst.B:,.0f} -> "
          + ("OK" if est <= inst.B else "BUDGET TOO SMALL for these units (raise --B or lower --demand-scale)"))

def verify_theory():
    f = np.linspace(0, 0.95, 400); C = 1.0
    g2 = 2 * C / (C - f) ** 3; z = np.linspace(0.01, 10, 400)
    for gm in (0.5, 0.8, 0.95):
        h2 = gm * (gm - 1) * z ** (gm - 2); assert np.all(h2 < 0)
    assert np.all(g2 > 0)
    print("theory check: g''(f) = 2C/(C-f)^3 > 0 on [0, 0.95C) (convex); h''(z) = gamma(gamma-1)z^(gamma-2) < 0 for 0<gamma<1 (concave)")

# ======================================================================= Custom I
def custom1_socp(inst, alpha=10.0, util_cap=0.95, capex_weight=0.001, solver=None, mip=False, verbose=False):
    """SOCP form: alpha * f^2/(C-f) (CVXPY quad_over_lin); the exact alpha * f/(C-f) is optimised in the last step."""
    import cvxpy as cp
    def build(y_fixed=None, integer=False):
        w = cp.Variable(len(inst.P), nonneg=True); z = cp.Variable(inst.m, integer=integer, nonneg=True)
        C = inst.C_base + inst.C_mod * z; f = inst.A @ w
        if y_fixed is None:
            y = cp.Variable(inst.n, boolean=integer); cons = [cp.sum(y) == inst.K, w <= y[inst.tgt]]
            if not integer: cons += [y >= 0, y <= 1]
            server_cost = inst.c_server * cp.sum(y)
        else:
            y = y_fixed; cons = [w <= y[inst.tgt]]; server_cost = inst.c_server * inst.K
        cons += [f <= util_cap * C, inst.c_mod @ z + server_cost <= inst.B]
        cons += [cp.sum(w[inst.by_src[s]]) == 1 for s in range(inst.n)]
        queue = cp.sum(cp.hstack([cp.quad_over_lin(f[e], C[e] - f[e]) for e in range(inst.m)]))
        return cp.Problem(cp.Minimize(inst.vol[inst.src] * inst.tau @ w + alpha * queue + capex_weight * (inst.c_mod @ z)), cons), y, w, z
    ok = ("optimal", "optimal_inaccurate")
    if mip:
        p, y, w, z = build(integer=True); p.solve(solver=solver, verbose=verbose)
        if p.status not in ok: raise RuntimeError(f"Custom I MISOCP {p.status}")
        y_bar, z_int, lb, how = np.round(y.value), np.ceil(z.value - 1e-9), None, "MISOCP"
    else:
        pA, yA, wA, zA = build(); pA.solve(solver=solver or cp.CLARABEL, verbose=verbose)
        if pA.status not in ok: raise RuntimeError(f"Custom I relaxation {pA.status}: budget too small for these units")
        y_bar = np.zeros(inst.n); y_bar[np.argsort(-yA.value)[: inst.K]] = 1             # top-K cities of the relaxation
        pB, _, wB, zB = build(y_fixed=y_bar); pB.solve(solver=solver or cp.CLARABEL, verbose=verbose)
        need = np.ceil(((inst.A @ wB.value) / util_cap - inst.C_base) / inst.C_mod - 1e-6).clip(min=0)   # modules the loads require
        for z_int in (np.maximum(need, np.floor(zB.value + 1e-6)), need):                                 # keep extras if CapEx allows
            if inst.c_server * inst.K + inst.c_mod @ z_int <= inst.B + 1e-6: break
        lb, how = pA.value, "SOCP relaxation + top-K + exact refine"
    w_ref = refine_exact(inst, y_bar, z_int, alpha, util_cap)
    if w_ref is not None: how += " (refined)"
    w_fin = clean(w_ref if w_ref is not None else wB.value)
    return report1(inst, y_bar, w_fin, z_int, alpha, how, lb)

def refine_exact(inst, y_bar, z_int, alpha, util_cap):
    """Integer capacities fixed: the exact M/M/1 term f/(C-f) = C*inv_pos(C-f) - 1 is convex in the routing, so re-optimise w."""
    import cvxpy as cp
    w = cp.Variable(len(inst.P), nonneg=True); C = inst.C_base + inst.C_mod * z_int; f = inst.A @ w
    cons = [w <= y_bar[inst.tgt], f <= util_cap * C] + [cp.sum(w[inst.by_src[s]]) == 1 for s in range(inst.n)]
    terms = [C[e] * cp.inv_pos(C[e] - f[e]) - 1 for e in range(inst.m) if C[e] > 0]
    p = cp.Problem(cp.Minimize(inst.vol[inst.src] * inst.tau @ w + alpha * cp.sum(cp.hstack(terms))), cons)
    try: p.solve(solver=cp.CLARABEL)
    except Exception: return None
    return w.value if p.status in ("optimal", "optimal_inaccurate") else None

def report1(inst, y, w, z, alpha, how, lower_bound=None):
    capex = inst.c_server * y.sum() + inst.c_mod @ z; f = inst.A @ w
    util = [f[e] / (inst.C_base[e] + inst.C_mod * z[e]) for e in range(inst.m) if f[e] > 1e-6]
    out = dict(method=how, servers=inst.city_names(y), latency=inst.latency(w), exact_mm1_objective=inst.latency(w) + exact_queue(inst, w, z, alpha),
               modules=int(z.sum()), capex=float(capex), within_budget=bool(capex <= inst.B + 1e-6), max_util=float(max(util, default=0.0)))
    if lower_bound is not None: out["relaxation_value_cone_form"] = float(lower_bound)
    return out

class MILP:
    """Tiny model builder on scipy.optimize.milp (HiGHS). No PuLP, so no PuLP version problems."""
    def __init__(self): self.lb, self.ub, self.integ, self.c, self.rows, self.rlo, self.rhi = [], [], [], [], [], [], []
    def var(self, lb=0.0, ub=np.inf, integer=False, cost=0.0):
        self.lb.append(lb); self.ub.append(ub); self.integ.append(int(integer)); self.c.append(cost); return len(self.c) - 1
    def row(self, coefs, lo=-np.inf, hi=np.inf): self.rows.append(coefs); self.rlo.append(lo); self.rhi.append(hi)
    def solve(self, time_limit):
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import csr_matrix
        ind = [k for r in self.rows for k in r]; dat = [v for r in self.rows for v in r.values()]; ptr = np.cumsum([0] + [len(r) for r in self.rows])
        A = csr_matrix((dat, ind, ptr), shape=(len(self.rows), len(self.c)))
        res = milp(np.array(self.c), constraints=LinearConstraint(A, self.rlo, self.rhi), integrality=np.array(self.integ),
                   bounds=Bounds(self.lb, self.ub), options=dict(time_limit=time_limit, disp=False))
        if res.status == 2: raise RuntimeError("MILP infeasible: budget too small for these units (see the feasibility estimate)")
        if res.x is None: raise RuntimeError(f"MILP found no solution: {res.message}")
        return res.x, ("Optimal" if res.status == 0 else "Time limit reached: best solution found, optimality NOT proven")

def custom1_pwl(inst, alpha=10.0, util_cap=0.95, capex_weight=0.001, time_limit=300, msg=False):
    """Exact integer MILP: per link one binary per module level n; for each level the convex term f/(C_n-f) is bounded below by tangent cuts."""
    nmax = max(1, math.ceil(inst.vol.sum() / util_cap / inst.C_mod)); rho = [0.0, 0.3, 0.55, 0.75, 0.88, util_cap]
    M = MILP(); P = range(len(inst.P))
    y = [M.var(0, 1, True) for _ in range(inst.n)]
    w = [M.var(0, 1, cost=inst.vol[inst.src[j]] * inst.tau[j]) for j in P]; q = [M.var(0, np.inf, cost=alpha) for _ in range(inst.m)]
    b = [[M.var(0, 1, True, cost=capex_weight * inst.c_mod[e] * k) for k in range(nmax + 1)] for e in range(inst.m)]
    M.row({y[i]: 1.0 for i in range(inst.n)}, inst.K, inst.K)
    for s in range(inst.n): M.row({w[j]: 1.0 for j in inst.by_src[s]}, 1, 1)
    for j in P: M.row({w[j]: 1.0, y[inst.tgt[j]]: -1.0}, -np.inf, 0)
    for e in range(inst.m):
        load = {w[j]: inst.A[e, j] for j in np.where(inst.A[e] > 0)[0]}; Cn = [inst.C_base[e] + inst.C_mod * k for k in range(nmax + 1)]
        M.row({b[e][k]: 1.0 for k in range(nmax + 1)}, 1, 1)
        M.row({**load, **{b[e][k]: -util_cap * Cn[k] for k in range(nmax + 1)}}, -np.inf, 0)         # f <= util * C
        f_hi = util_cap * Cn[-1]; tangents = []
        for k in range(nmax + 1):
            if Cn[k] <= 0: continue
            for r in rho:
                f0, slope = r * Cn[k], 1.0 / (Cn[k] * (1 - r) ** 2); g0 = r / (1 - r)
                tangents.append((k, g0 - slope * f0, slope, g0 + slope * (f_hi - f0)))
        Mbig = max([t[3] for t in tangents] + [1.0])
        for k, c0, slope, _ in tangents:                                                                # q >= tangent when level k is chosen
            M.row({q[e]: 1.0, **{v: -slope * c for v, c in load.items()}, b[e][k]: -Mbig}, c0 - Mbig, np.inf)
    budget = {y[i]: inst.c_server for i in range(inst.n)}
    for e in range(inst.m):
        for k in range(nmax + 1): budget[b[e][k]] = inst.c_mod[e] * k
    M.row(budget, -np.inf, inst.B)
    x, st = M.solve(time_limit)
    yv = np.round(x[y]); wv = clean(x[w]); zv = np.array([sum(k * x[b[e][k]] for k in range(nmax + 1)) for e in range(inst.m)]).round()
    out = report1(inst, yv, wv, zv, alpha, "tangent-cut MILP (HiGHS)"); out["status"] = st
    return out

# ======================================================================= Custom II
def concave_cost(inst, z, gamma):
    alpha = inst.c_mod / inst.C_mod ** gamma                       # one C_mod block costs c_mod_e
    return float((alpha * np.maximum(z, 0) ** gamma).sum())

def report2(inst, y, w, gamma, how, extra=None):
    w = clean(w); z = np.maximum(inst.A @ w - inst.C_base, 0)       # cheapest bandwidth that carries the loads (cost is increasing)
    capex = inst.c_server * np.sum(y > 0.5) + concave_cost(inst, z, gamma)
    out = dict(method=how, servers=inst.city_names(y), latency=inst.latency(w), capex=float(capex), within_budget=bool(capex <= inst.B + 1e-6),
               bandwidth_needed_gbps=float(z.sum()))
    out.update(extra or {}); return out

def custom2_sos2(inst, gamma=0.8, n_break=8, time_limit=300, msg=False):
    """Concave cost -> chords with SOS2 (adjacent lambdas + one binary per segment). Chords lie below the curve, so true CapEx is re-computed."""
    alpha = inst.c_mod / inst.C_mod ** gamma; zmax = float(inst.vol.sum())
    bp = np.concatenate([[0.0], np.geomspace(inst.C_mod / 8, zmax, n_break - 1)])
    M = MILP(); P = range(len(inst.P))
    y = [M.var(0, 1, True) for _ in range(inst.n)]; w = [M.var(0, 1, cost=inst.vol[inst.src[j]] * inst.tau[j]) for j in P]
    z = [M.var(0, zmax) for _ in range(inst.m)]
    M.row({y[i]: 1.0 for i in range(inst.n)}, inst.K, inst.K)
    for s in range(inst.n): M.row({w[j]: 1.0 for j in inst.by_src[s]}, 1, 1)
    for j in P: M.row({w[j]: 1.0, y[inst.tgt[j]]: -1.0}, -np.inf, 0)
    budget = {y[i]: inst.c_server for i in range(inst.n)}
    for e in range(inst.m):
        M.row({**{w[j]: inst.A[e, j] for j in np.where(inst.A[e] > 0)[0]}, z[e]: -1.0}, -np.inf, inst.C_base[e])     # load <= C_base + z
        lam = [M.var(0, 1) for _ in range(n_break)]; seg = [M.var(0, 1, True) for _ in range(n_break - 1)]
        M.row({l: 1.0 for l in lam}, 1, 1); M.row({s_: 1.0 for s_ in seg}, 1, 1)
        M.row({z[e]: 1.0, **{lam[k]: -bp[k] for k in range(n_break)}}, 0, 0)
        for k in range(n_break):                                                                                   # SOS2: two adjacent lambdas at most
            r = {lam[k]: 1.0}
            if k > 0: r[seg[k - 1]] = -1.0
            if k < n_break - 1: r[seg[k]] = -1.0
            M.row(r, -np.inf, 0)
        for k in range(n_break): budget[lam[k]] = alpha[e] * bp[k] ** gamma
    M.row(budget, -np.inf, inst.B)
    x, st = M.solve(time_limit)
    yv = np.round(x[y]); wv = np.array(x[w])
    return report2(inst, yv, wv, gamma, "SOS2 piecewise-linear MILP (HiGHS)", dict(status=st))

def custom2_ipopt(inst, gamma=0.8, n_starts=8, seed=0, eps_frac=1e-3, max_iter=600):
    """Continuous non-convex NLP with IPOPT (CasADi). z^gamma is smoothed as (z+eps)^gamma - eps^gamma (infinite slope at 0).
    Each start: random y0 -> relaxed NLP -> keep K largest y -> NLP over (w, z) with y fixed. Best feasible start wins."""
    import casadi as ca, scipy.sparse as sp
    alpha = inst.c_mod / inst.C_mod ** gamma; eps = eps_frac * inst.C_mod; nP = len(inst.P); n = inst.n
    A = ca.DM(sp.csc_matrix(inst.A)); S = np.zeros((n, nP)); T = np.zeros((nP, n))
    for j in range(nP): S[inst.src[j], j] = 1; T[j, inst.tgt[j]] = 1
    S, T = ca.DM(sp.csc_matrix(S)), ca.DM(sp.csc_matrix(T)); cvec = inst.vol[inst.src] * inst.tau
    def solve(y_fixed, y0, w0, z0):
        o = ca.Opti(); w = o.variable(nP); z = o.variable(inst.m)
        y = o.variable(n) if y_fixed is None else ca.DM(y_fixed)
        o.subject_to(o.bounded(0, w, 1)); o.subject_to(z >= 0); o.subject_to(ca.mtimes(S, w) == 1); o.subject_to(w <= ca.mtimes(T, y))
        o.subject_to(ca.mtimes(A, w) <= ca.DM(inst.C_base) + z)
        o.subject_to(inst.c_server * ca.sum1(y) + ca.sum1(ca.DM(alpha) * ((z + eps) ** gamma - eps ** gamma)) <= inst.B)
        if y_fixed is None: o.subject_to(o.bounded(0, y, 1)); o.subject_to(ca.sum1(y) == inst.K); o.set_initial(y, y0)
        o.minimize(ca.dot(ca.DM(cvec), w)); o.set_initial(w, w0); o.set_initial(z, z0)
        o.solver("ipopt", {"print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes", "ipopt.max_iter": max_iter, "ipopt.tol": 1e-7})
        try: sol = o.solve()
        except RuntimeError: return None
        return (None if y_fixed is not None else np.array(sol.value(y)).ravel()), np.array(sol.value(w)).ravel(), np.array(sol.value(z)).ravel()
    rng = np.random.default_rng(seed); w0 = np.zeros(nP)
    for s in range(n): w0[inst.by_src[s]] = 1.0 / len(inst.by_src[s])
    z0 = np.maximum(inst.A @ w0 - inst.C_base, 0); results, tried = [], 0
    for k in range(n_starts):
        y0 = np.full(n, inst.K / n) if k == 0 else np.minimum(rng.dirichlet(np.ones(n) * 0.5) * inst.K, 1.0)
        r = solve(None, y0, w0, z0); tried += 1
        if r is None: continue
        y_bar = np.zeros(n); y_bar[np.argsort(-r[0])[: inst.K]] = 1
        r2 = solve(y_bar, None, r[1], r[2])
        if r2 is None: continue
        res = report2(inst, y_bar, r2[1], gamma, "IPOPT multi-start (CasADi)")
        res["relaxed_y_max_fraction_gap"] = float(np.max(np.minimum(r[0], 1 - r[0])))
        results.append(res)
    feas = [r for r in results if r["within_budget"]]
    if not feas: raise RuntimeError(f"Custom II IPOPT: no feasible start ({len(results)}/{tried} solved); check budget/units")
    best = min(feas, key=lambda r: r["latency"])
    best.update(starts_run=tried, starts_solved=len(results), starts_feasible=len(feas), latency_per_start=[round(r["latency"], 4) for r in results])
    return best

# ======================================================================= main
METHODS = ["c1-socp", "c1-pwl", "c2-sos2", "c2-ipopt"]

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file"); ap.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    ap.add_argument("--K", type=int, default=5); ap.add_argument("--B", type=float, default=150000.0); ap.add_argument("--c-server", type=float, default=1000.0)
    ap.add_argument("--demand-scale", type=float, default=1 / 1000, help="file demand units -> Gbps (ASSUMED Mbps)")
    ap.add_argument("--module-gbps", type=float, default=1.0, help="json files only (ASSUMED)")
    ap.add_argument("--module-cost-per-km", type=float, default=0.78, help="json files only (ASSUMED)")
    ap.add_argument("--k-paths", type=int, default=3); ap.add_argument("--alpha", type=float, default=10.0)
    ap.add_argument("--util-cap", type=float, default=0.95); ap.add_argument("--capex-weight", type=float, default=0.001)
    ap.add_argument("--gamma", type=float, default=0.8); ap.add_argument("--n-break", type=int, default=8)
    ap.add_argument("--time-limit", type=int, default=300); ap.add_argument("--starts", type=int, default=8); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solver", default=None, help="CVXPY solver for c1-socp, e.g. MOSEK"); ap.add_argument("--mip", action="store_true", help="c1-socp: one-shot MISOCP")
    ap.add_argument("--out", default=None, help="write results to this JSON file")
    a = ap.parse_args()
    data = parse_sndlib(a.file) if a.file.endswith(".txt") else load_json(a.file, a.module_gbps, a.module_cost_per_km)
    audit(data); verify_theory()
    inst = Instance(data, a.K, a.B, a.c_server, a.demand_scale, k_paths=a.k_paths)
    print(f"instance: {inst.n} cities, {inst.m} arcs, {len(inst.P)} candidate paths, traffic {inst.vol.sum():.2f} Gbps"); feasibility_estimate(inst)
    run = {"c1-socp": lambda: custom1_socp(inst, a.alpha, a.util_cap, a.capex_weight, a.solver, a.mip),
           "c1-pwl": lambda: custom1_pwl(inst, a.alpha, a.util_cap, a.capex_weight, a.time_limit),
           "c2-sos2": lambda: custom2_sos2(inst, a.gamma, a.n_break, a.time_limit),
           "c2-ipopt": lambda: custom2_ipopt(inst, a.gamma, a.starts, a.seed)}
    results = {}
    for m in a.methods:
        t = time.time()
        try: results[m] = run[m](); results[m]["seconds"] = round(time.time() - t, 1)
        except Exception as e: results[m] = {"error": str(e)}; print(f"[{m}] FAILED: {e}", file=sys.stderr)
    print("\n=== results (servers are chosen by the optimiser) ===")
    for m, r in results.items():
        print(f"\n[{m}]"); [print(f"  {k:>28}: {v}") for k, v in r.items()]
    if a.out:
        json.dump(dict(args=vars(a), results=results), open(a.out, "w"), indent=2, default=float); print(f"\nsaved {a.out}")

if __name__ == "__main__":
    main()
