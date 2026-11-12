"""Fixed-placement routing sub-LP (used by Hard Track 2; reusable to verify MILP solutions)."""
import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix, vstack


def solve_fixed_placement(net, servers, cache, cfg):
    """Servers fixed (binary y). Optimise path fractions w and capacity modules z.

    integer_modules=True : modules are integers (small MILP, as in the report's Eq. 10).
    integer_modules=False: modules solved continuously (fast LP), rounded UP, and the
                           CapEx budget enforced on the rounded values (repair loop below).
    """
    servers = [int(s) for s in servers]
    S, M = len(net.src_nodes), net.m
    bad = dict(objective=np.inf, feasible=False, modules=None, capex=np.inf)

    server_capex = float(net.server_cost[servers].sum())
    budget_left = cfg.budget - server_capex
    if budget_left < 0:
        return bad

    delay, col_src, col_edges = [], [], []
    for si, s in enumerate(net.src_nodes):
        for t in servers:
            for d, eidx in cache.get(int(s), t):
                delay.append(d); col_src.append(si); col_edges.append(eidx)
    if not delay or np.bincount(col_src, minlength=S).min() == 0:
        return bad                      # some origin cannot reach any chosen server

    P = len(delay)
    delay, col_src = np.array(delay), np.array(col_src)
    vcol = net.volume[col_src]
    nv = P + M

    # tiny CapEx tie-breaker so the LP never buys modules it does not need (excluded from reported Z)
    c = np.concatenate([vcol * delay, 1e-4 * net.mod_cost / max(net.mod_cost.max(), 1e-12)])
    A_eq = coo_matrix((np.ones(P), (col_src, np.arange(P))), shape=(S, nv)).tocsr()
    b_eq = np.ones(S)

    lens = np.array([len(e) for e in col_edges])
    cap_rows = np.concatenate(col_edges) if lens.sum() else np.empty(0, dtype=np.int64)
    cap_cols = np.repeat(np.arange(P), lens)
    cap_data = np.repeat(vcol, lens)
    A_cap = coo_matrix(
        (np.concatenate([cap_data, -np.full(M, cfg.module_capacity)]),
         (np.concatenate([cap_rows, np.arange(M)]), np.concatenate([cap_cols, P + np.arange(M)]))),
        shape=(M, nv)).tocsr()
    A_bud = coo_matrix((net.mod_cost, (np.zeros(M, dtype=int), P + np.arange(M))), shape=(1, nv)).tocsr()

    A_ub = vstack([A_cap, A_bud]).tocsr()

    if getattr(cfg, "integer_modules", False):
        # Sub-MILP: y is fixed, so only the modules z are integer (much smaller than the full MILP).
        res = milp(c, constraints=[LinearConstraint(A_eq, b_eq, b_eq),
                                   LinearConstraint(A_ub, -np.inf, np.concatenate([net.cap_base, [budget_left]]))],
                   integrality=np.concatenate([np.zeros(P), np.ones(M)]), bounds=Bounds(0, np.inf),
                   options={"time_limit": cfg.milp_time_limit, "mip_rel_gap": 1e-6})
        if res.x is None:
            return bad
        z_int = np.round(res.x[P:]).clip(min=0)
        capex = server_capex + float(net.mod_cost @ z_int)
        return dict(objective=float(c[:P] @ res.x[:P]), feasible=bool(capex <= cfg.budget + 1e-6),
                    modules=z_int, capex=capex)
    # Modules are solved continuously, then rounded UP.  If rounding breaks the CapEx budget,
    # shrink the LP budget by the overshoot and re-solve (usually converges in 1-3 rounds).
    eff = budget_left
    for _ in range(8):
        res = linprog(c, A_ub=A_ub, b_ub=np.concatenate([net.cap_base, [eff]]),
                      A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method="highs")
        if not res.success:
            return bad
        z_int = np.ceil(res.x[P:] - 1e-7).clip(min=0)
        capex = server_capex + float(net.mod_cost @ z_int)
        if capex <= cfg.budget + 1e-6:
            return dict(objective=float(c[:P] @ res.x[:P]), feasible=True, modules=z_int, capex=capex)
        eff -= (capex - cfg.budget)
        if eff <= 0:
            break
    return bad