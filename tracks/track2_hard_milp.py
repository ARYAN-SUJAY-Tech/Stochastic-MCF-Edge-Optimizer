"""Hard Track 1 (MILP): Exact joint edge server placement, routing, and modular capacity expansion."""
import time
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix, vstack

from edgeopt.config import Config
from edgeopt.network import load_network
from edgeopt.paths import PathCache
from edgeopt.routing import solve_fixed_placement
from edgeopt.metrics import save_result, load_result, gap_pct


def solve_hard_milp(net, cache, cfg):
    """Formulates and solves the exact MILP for server placement (y), routing (w), and modules (z)."""
    N = net.n                  # Number of candidate nodes (|V| = 26)
    S = len(net.src_nodes)     # Number of traffic demands (|D| = 650)
    M = net.m                  # Number of directed edges (|E| = 84)

    # 1. Extract all candidate paths across all (demand, candidate_server) pairs
    delay, col_src, col_tgt, col_edges = [], [], [], []
    for si, s in enumerate(net.src_nodes):
        for t in range(N):
            for d, eidx in cache.get(int(s), t):
                delay.append(d)
                col_src.append(si)
                col_tgt.append(t)
                col_edges.append(eidx)

    P = len(delay)
    if P == 0 or np.bincount(col_src, minlength=S).min() == 0:
        return dict(feasible=False, objective=np.inf)

    delay = np.array(delay, dtype=float)
    col_src = np.array(col_src, dtype=np.int64)
    col_tgt = np.array(col_tgt, dtype=np.int64)
    vcol = net.volume[col_src]

    # Decision vector layout: x = [ w (P continuous) | y (N binary) | z (M integer) ]
    nv = P + N + M
    idx_y = P
    idx_z = P + N

    # 2. Objective Function: min sum_{d, p} (demand_d * tau_p * w_{d,p})
    # Tiny tie-breakers on y and z prevent arbitrary unused hardware purchases within budget
    c = np.concatenate([
        vcol * delay,
        1e-6 * net.server_cost / max(net.server_cost.max(), 1e-12),
        1e-6 * net.mod_cost / max(net.mod_cost.max(), 1e-12)
    ])

    # 3. Equality Constraints (A_eq * x == b_eq)
    # (a) Demand Route Completion: sum_{p in P_d} w_{d,p} = 1  for all d
    A_dem = coo_matrix((np.ones(P), (col_src, np.arange(P))), shape=(S, nv)).tocsr()
    b_dem = np.ones(S)

    # (b) Exact Edge Server Count: sum_{i in V} y_i = K
    A_srv = coo_matrix((np.ones(N), (np.zeros(N, dtype=np.int64), idx_y + np.arange(N))), shape=(1, nv)).tocsr()
    b_srv = np.array([float(cfg.K)])

    A_eq = vstack([A_dem, A_srv]).tocsr()
    b_eq = np.concatenate([b_dem, b_srv])

    # 4. Inequality Constraints (A_ub * x <= b_ub)
    # (a) Active Server Destination Coupling: w_{d,p} - y_{target(p)} <= 0
    A_cpl = coo_matrix(
        (np.concatenate([np.ones(P), -np.ones(P)]),
         (np.concatenate([np.arange(P), np.arange(P)]),
          np.concatenate([np.arange(P), idx_y + col_tgt]))),
        shape=(P, nv)
    ).tocsr()
    b_cpl = np.zeros(P)

    # (b) Link Capacity & Module Expansion: sum_{d, p ni e} demand_d * w_{d,p} - C_mod * z_e <= C_base_e
    lens = np.array([len(e) for e in col_edges])
    cap_rows = np.concatenate(col_edges) if lens.sum() else np.empty(0, dtype=np.int64)
    cap_cols = np.repeat(np.arange(P), lens)
    cap_data = np.repeat(vcol, lens)

    A_cap = coo_matrix(
        (np.concatenate([cap_data, -np.full(M, cfg.module_capacity)]),
         (np.concatenate([cap_rows, np.arange(M)]),
          np.concatenate([cap_cols, idx_z + np.arange(M)]))),
        shape=(M, nv)
    ).tocsr()
    b_cap = net.cap_base

    # (c) CapEx Budget Limit: sum_i c_server_i * y_i + sum_e c_mod_e * z_e <= B
    A_bud = coo_matrix(
        (np.concatenate([net.server_cost, net.mod_cost]),
         (np.zeros(N + M, dtype=np.int64),
          np.concatenate([idx_y + np.arange(N), idx_z + np.arange(M)]))),
        shape=(1, nv)
    ).tocsr()
    b_bud = np.array([float(cfg.budget)])

    A_ub = vstack([A_cpl, A_cap, A_bud]).tocsr()
    b_ub = np.concatenate([b_cpl, b_cap, b_bud])

    # 5. Variable Domains & Integrality
    # w in [0, 1] (continuous = 0), y in {0, 1} (integer = 1), z in Z>=0 (integer = 1)
    integrality = np.concatenate([np.zeros(P), np.ones(N), np.ones(M)])
    lb = np.zeros(nv)
    ub = np.concatenate([np.ones(P), np.ones(N), np.full(M, np.inf)])

    # 6. Solve MILP via HiGHS
    t0 = time.perf_counter()
    res = milp(
        c,
        integrality=integrality,
        bounds=Bounds(lb, ub),
        constraints=[
            LinearConstraint(A_eq, b_eq, b_eq),
            LinearConstraint(A_ub, -np.inf, b_ub)
        ],
        options={
            "time_limit": float(cfg.milp_time_limit),
            "mip_rel_gap": 1e-6
        }
    )
    solve_time = time.perf_counter() - t0

    if res.x is None:
        return dict(feasible=False, objective=np.inf, solve_time_s=solve_time)

    # 7. Extract and Clean Solution
    w_opt = res.x[:P]
    y_opt = np.round(res.x[idx_y:idx_z]).astype(int)
    z_opt = np.round(res.x[idx_z:]).astype(int).clip(min=0)

    chosen_servers = np.where(y_opt == 1)[0].tolist()
    true_objective = float((vcol * delay) @ w_opt)
    total_capex = float(net.server_cost[chosen_servers].sum() + net.mod_cost @ z_opt)

    return dict(
        feasible=True,
        objective=true_objective,
        servers=chosen_servers,
        modules=z_opt,
        total_modules=int(z_opt.sum()),
        capex=total_capex,
        mip_gap=float(getattr(res, "mip_gap", 0.0)),
        solve_time_s=solve_time
    )


def main():
    cfg = Config.load()
    net = load_network(cfg.data_path, cfg)
    cache = PathCache(net, cfg.k_paths)

    print(f"Running Exact MILP (K={cfg.K}, Budget={cfg.budget}, k_paths={cfg.k_paths})...")
    sol = solve_hard_milp(net, cache, cfg)

    if not sol["feasible"]:
        print("MILP failed to find a feasible solution.")
        save_result("track2_hard_milp", sol, cfg)
        return

    print(f"Optimal Objective (Z_MILP*): {sol['objective']:.4f}")
    print(f"Chosen Servers: {sol['servers']}")
    print(f"CapEx Used: {sol['capex']:.2f} / {cfg.budget}")
    print(f"Solve Time: {sol['solve_time_s']:.3f}s")

    # 8. Verify Solution using Fixed-Placement Sub-MILP (routing.py)
    cfg.integer_modules = True
    ver = solve_fixed_placement(net, sol["servers"], cache, cfg)
    assert ver["feasible"], "Verification failed: Sub-MILP marked solution infeasible!"
    assert abs(ver["objective"] - sol["objective"]) < 1e-3, (
        f"Objective mismatch! MILP={sol['objective']}, Verifier={ver['objective']}"
    )
    print("Verification passed against routing.solve_fixed_placement.")

    # 9. Compute Price of Realism against Easy LP / Continuous Relaxation (if available)
    try:
        lp_res = load_result("track1_easy_lp", cfg)
        z_lp = lp_res.get("objective", None)
        if z_lp and np.isfinite(z_lp):
            por_pct = gap_pct(sol["objective"], z_lp)
            sol["price_of_realism_pct"] = por_pct
            sol["price_of_realism_ratio"] = por_pct / 100.0
            print(f"Price of Realism vs Track 1 ({z_lp:.4f}): {por_pct:.2f}%")
    except FileNotFoundError:
        print("track1_easy_lp.json not found yet; skipping Price of Realism calculation.")

    save_result("track2_hard_milp", sol, cfg)
    print("Saved results to results/track2_hard_milp.json")


if __name__ == "__main__":
    main()