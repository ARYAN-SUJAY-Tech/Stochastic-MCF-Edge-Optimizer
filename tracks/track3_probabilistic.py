"""Hard Track 2: probabilistic LP rounding (Phase I relaxation -> Phase II Monte Carlo sub-LPs).

Run:  python -m tracks.track3_probabilistic [--trials 200] [--seed 42] [--method systematic|weighted]

Efficiency notes
  * Sampled placements are cached, so repeated draws cost nothing (LP mass concentrates on few sets).
  * Trial 0 is the deterministic top-K-by-y_hat set (a free greedy baseline).
  * 'systematic' sampling picks exactly K distinct nodes and keeps marginals P(i in S) = y_hat_i
    (nodes with y_hat = 1 are always chosen).  'weighted' is the plain sequential
    without-replacement sampling from the report.
"""
import argparse
import time
import numpy as np

from edgeopt.config import Config
from edgeopt.network import load_network
from edgeopt.paths import PathCache
from edgeopt.routing import solve_fixed_placement
from edgeopt.metrics import save_result, load_result, gap_pct
from tracks.track1_easy_lp import solve_relaxation


def sample_systematic(y, K, rng):
    n = len(y)
    y = np.clip(y, 0.0, 1.0)
    perm = rng.permutation(n)
    cum = np.cumsum(y[perm])
    cum *= K / cum[-1]
    pts = rng.random() + np.arange(K)
    chosen = set(perm[np.minimum(np.searchsorted(cum, pts, side="right"), n - 1)].tolist())
    for j in np.argsort(-y):                       # float-edge-case fill
        if len(chosen) >= K: break
        chosen.add(int(j))
    return tuple(sorted(chosen))


def sample_weighted(y, K, rng, eps=1e-6):
    p = np.clip(y, 0, None) + eps
    return tuple(sorted(rng.choice(len(y), size=K, replace=False, p=p / p.sum()).tolist()))


def run(net, cfg, relax=None):
    relax = relax or solve_relaxation(net, cfg)
    y_hat = relax["y_hat"]
    rng = np.random.default_rng(cfg.seed)
    sampler = sample_systematic if cfg.sampling == "systematic" else sample_weighted
    cache = PathCache(net, cfg.k_paths)

    evaluated, best, history = {}, None, []
    t0 = time.perf_counter()
    for n in range(cfg.n_trials + 1):
        S = tuple(sorted(np.argsort(-y_hat)[:cfg.K].tolist())) if n == 0 else sampler(y_hat, cfg.K, rng)
        if S not in evaluated:
            evaluated[S] = solve_fixed_placement(net, S, cache, cfg)
        r = evaluated[S]
        if r["feasible"] and (best is None or r["objective"] < best[1]["objective"]):
            best = (S, r)
        history.append(best[1]["objective"] if best else None)
    dt = time.perf_counter() - t0
    return dict(best=best, history=history, unique=len(evaluated),
                n_feasible=sum(r["feasible"] for r in evaluated.values()),
                phase2_time_s=dt, relax=relax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--data"); ap.add_argument("--K", type=int); ap.add_argument("--budget", type=float)
    ap.add_argument("--trials", type=int); ap.add_argument("--seed", type=int)
    ap.add_argument("--method", choices=["systematic", "weighted"]); ap.add_argument("--k_paths", type=int)
    a = ap.parse_args()
    cfg = Config.load(a.config, data_path=a.data, K=a.K, budget=a.budget,
                      n_trials=a.trials, seed=a.seed, sampling=a.method, k_paths=a.k_paths)
    net = load_network(cfg.data_path, cfg)
    try:
        out = run(net, cfg)
    except RuntimeError as e:
        raise SystemExit(f"ERROR: {e}")
    relax, best = out["relax"], out["best"]
    lb = relax["objective"]
    print(f"Phase I  Z*_Relaxed = {lb:,.2f}  ({relax['solve_time_s']:.2f}s)")
    print(f"Phase II {cfg.n_trials} trials, {out['unique']} unique placements, "
          f"{out['n_feasible']} feasible, {out['phase2_time_s']:.2f}s")
    if best is None:
        print("No feasible placement found - increase trials / budget or check module_cost.")
    else:
        S, r = best
        print(f"Best servers: {[net.node_ids[i] for i in S]}")
        print(f"Z_heuristic = {r['objective']:,.2f} | gap to LB = {gap_pct(r['objective'], lb):.3f}% | CapEx {r['capex']:,.0f}")

    result = dict(
        track="probabilistic_lp_rounding", dataset=net.name, lower_bound=lb,
        objective=best[1]["objective"] if best else None,
        servers=[net.node_ids[i] for i in best[0]] if best else None,
        modules_total=float(best[1]["modules"].sum()) if best else None,
        capex=best[1]["capex"] if best else None, feasible=best is not None,
        gap_to_lb_pct=gap_pct(best[1]["objective"], lb) if best else None,
        n_trials=cfg.n_trials, unique_placements=out["unique"], n_feasible=out["n_feasible"],
        total_time_s=relax["solve_time_s"] + out["phase2_time_s"],
        convergence=out["history"], config=cfg.to_dict())
    # If Binish's MILP result exists, report the true heuristic optimality gap too
    try:
        milp = load_result("track2_hard_milp", cfg)
        if best and milp.get("objective"):
            result["gap_to_milp_pct"] = gap_pct(best[1]["objective"], milp["objective"])
            print(f"Gap to MILP = {result['gap_to_milp_pct']:.3f}%")
    except FileNotFoundError:
        pass
    save_result("track3_probabilistic", result, cfg)


if __name__ == "__main__":
    main()