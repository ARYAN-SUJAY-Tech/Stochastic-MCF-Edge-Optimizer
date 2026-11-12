"""Easy Track: continuous LP relaxation (arc-based multi-commodity flow).

Run:  python -m tracks.track1_easy_lp [--config config.json] [--data data/x.json] [--K 5] [--budget 150000]

NOTE on the flow-conservation constraint (4) in the Easy-track PDF:
  out - in = 1 - y_s at the source and -y_i elsewhere  sums to 1 - K over all nodes,
  but net flow must sum to 0, so it is infeasible for K > 1.
  Fix used here: explicit absorption variables a[s,i] in [0, y_i] with
      out - in + a[s,i] = [i == s]      (and sum_i a[s,i] = 1 follows automatically).
Demands are aggregated by origin node (exact, 650 demands -> |origins| commodities).
"""
import argparse
import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, vstack

from edgeopt.config import Config
from edgeopt.network import load_network
from edgeopt.metrics import save_result


def solve_relaxation(net, cfg, mode="latency"):
    S, N, M = len(net.src_nodes), net.n, net.m
    off_a, off_y = S * M, S * M + S * N
    off_z = off_y + N
    nv = off_z + M

    s_idx = np.repeat(np.arange(S), M)
    e_idx = np.tile(np.arange(M), S)
    x_col = s_idx * M + e_idx
    sN = np.arange(S * N)

    # objective: sum_s vol_s * delay_e * x[s,e]
    c = np.zeros(nv)
    c[:off_a] = net.volume[s_idx] * net.delay_ms[e_idx]
    # tiny CapEx tie-breaker so the LP never buys modules it does not need (excluded from reported Z)
    c[off_z:] = 1e-4 * net.mod_cost / max(net.mod_cost.max(), 1e-12)

    # equalities: flow conservation with absorption, then sum(y) = K
    rows = np.concatenate([s_idx * N + net.eu[e_idx], s_idx * N + net.ev[e_idx], sN, np.full(N, S * N)])
    cols = np.concatenate([x_col, x_col, off_a + sN, off_y + np.arange(N)])
    data = np.concatenate([np.ones(S * M), -np.ones(S * M), np.ones(S * N), np.ones(N)])
    A_eq = coo_matrix((data, (rows, cols)), shape=(S * N + 1, nv)).tocsr()
    b_eq = np.zeros(S * N + 1)
    b_eq[np.arange(S) * N + net.src_nodes] = 1.0
    b_eq[-1] = cfg.K

    # inequalities: a[s,i] <= y_i ; link capacity ; CapEx budget
    A_link = coo_matrix(
        (np.concatenate([np.ones(S * N), -np.ones(S * N)]),
         (np.concatenate([sN, sN]), np.concatenate([off_a + sN, off_y + np.tile(np.arange(N), S)]))),
        shape=(S * N, nv)).tocsr()
    A_cap = coo_matrix(
        (np.concatenate([net.volume[s_idx], -np.full(M, cfg.module_capacity)]),
         (np.concatenate([e_idx, np.arange(M)]), np.concatenate([x_col, off_z + np.arange(M)]))),
        shape=(M, nv)).tocsr()
    A_bud = coo_matrix(
        (np.concatenate([net.server_cost, net.mod_cost]),
         (np.zeros(N + M, dtype=int), np.concatenate([off_y + np.arange(N), off_z + np.arange(M)]))),
        shape=(1, nv)).tocsr()
    A_ub = vstack([A_link, A_cap, A_bud]).tocsr()
    b_ub = np.concatenate([np.zeros(S * N), net.cap_base, [cfg.budget]])

    if mode == "capex":                 # diagnostic: cheapest possible CapEx that carries the traffic
        c[:] = 0.0
        c[off_y:off_z] = net.server_cost
        c[off_z:] = net.mod_cost
        b_ub[-1] = 1e18

    ub = np.full(nv, np.inf)
    ub[off_y:off_z] = 1.0
    t0 = time.perf_counter()
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=np.column_stack([np.zeros(nv), ub]), method="highs")
    dt = time.perf_counter() - t0
    if not res.success:
        msg = f"LP relaxation failed ({res.message.split('(')[0].strip()})."
        if mode == "latency":
            try:
                mn = solve_relaxation(net, cfg, mode="capex")["objective"]
                msg += (f"\n  Budget B = {cfg.budget:,.0f}, but the cheapest way to carry the traffic with K = {cfg.K} "
                        f"costs at least {mn:,.0f}\n  (module_cost = {cfg.module_cost:,.0f}, module_capacity = {cfg.module_capacity}, "
                        f"total demand = {net.volume.sum():,.0f}).\n  Fix one of: raise --budget to >= {mn:,.0f}; "
                        f"lower module_cost; or set demand_scale if demand units differ from capacity units.")
            except RuntimeError:
                msg += " Check K (must be <= number of nodes) and that the network is connected."
        raise RuntimeError(msg)

    y, z = np.clip(res.x[off_y:off_z], 0.0, 1.0), res.x[off_z:]
    y[y < 1e-9] = 0.0
    obj = float(res.fun) if mode == "capex" else float(c[:off_a] @ res.x[:off_a])
    return dict(
        objective=obj, y_hat=y, z=z, solve_time_s=dt,
        mean_latency_ms=float(obj / net.volume.sum()),
        capex=dict(server=float(net.server_cost @ y), capacity=float(net.mod_cost @ z),
                   total=float(net.server_cost @ y + net.mod_cost @ z)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--data"); ap.add_argument("--K", type=int); ap.add_argument("--budget", type=float)
    a = ap.parse_args()
    cfg = Config.load(a.config, data_path=a.data, K=a.K, budget=a.budget)
    net = load_network(cfg.data_path, cfg)
    print(f"Loaded {net.name}: {net.n} nodes, {net.m} directed links, {len(net.src_nodes)} origins")

    try:
        r = solve_relaxation(net, cfg)
    except RuntimeError as e:
        raise SystemExit(f"ERROR: {e}")
    print(f"Z*_LP = {r['objective']:,.2f}  | mean latency {r['mean_latency_ms']:.3f} ms | "
          f"{r['solve_time_s']:.2f}s | modules {r['z'].sum():.2f} | CapEx {r['capex']['total']:,.0f}")
    top = np.argsort(-r["y_hat"])[:8]
    print("Top y_hat:", ", ".join(f"{net.node_ids[i]}={r['y_hat'][i]:.3f}" for i in top if r["y_hat"][i] > 1e-6))

    save_result("track1_easy_lp", dict(
        track="easy_lp", dataset=net.name, objective=r["objective"], lower_bound=r["objective"],
        mean_latency_ms=r["mean_latency_ms"], solve_time_s=r["solve_time_s"],
        modules_total=float(r["z"].sum()), capex=r["capex"]["total"], capex_breakdown=r["capex"],
        budget_utilisation=r["capex"]["total"] / cfg.budget,
        y_hat=dict(zip(net.node_ids, r["y_hat"])), config=cfg.to_dict()), cfg)


if __name__ == "__main__":
    main()