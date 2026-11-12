# Stochastic-MCF Edge Optimizer

Multi-paradigm mathematical optimization framework for **edge server placement and capacity expansion** on the SNDlib **Janos-US** backbone network, using LP, MILP, probabilistic LP rounding, and non-linear (convex and non-convex) solvers.

Team: Aatraya Mukherjee, Aryan Sujay, Binish Sood.

---

## 1. The problem in plain words

We have a telecom backbone: cities (nodes) connected by fibre links (edges), and traffic demands between cities.

We want to place **K edge servers** in K cities so that traffic reaches a server with **as little propagation delay as possible**. Links start with (almost) no bandwidth, so we also have to **buy capacity** in 64 Gbps modules. Everything (servers plus modules) must fit inside a **CapEx budget B**.

**Janos-US:** 26 nodes, 84 directed links (42 physical links, both directions), 650 demands.

**Decisions**

| Symbol | Meaning | Type |
|---|---|---|
| `y_i` | is there a server in city *i*? | binary (0/1) |
| `z_e` | number of 64 Gbps modules bought on link *e* | integer |
| `w_{d,p}` | fraction of demand *d* sent over path *p* | continuous in [0,1] |

**Objective:** minimise total demand-weighted latency, the sum over demands and paths of `demand_d * delay_p * w_{d,p}`.

**Constraints:** exactly K servers; every demand fully routed; a path may only be used if a server sits at its end; link flow is at most the installed capacity; server cost plus module cost is at most B.

---

## 2. The five tracks (who owns what)

All tracks solve the same underlying problem, with progressively harder mathematics.

| File | Track | Owner | Idea |
|---|---|---|---|
| `tracks/track1_easy_lp.py` | Easy LP | Aryan | Relax `y` and `z` to continuous and solve a multi-commodity-flow LP. Gives a **lower bound** and a fractional server vector `y_hat`. |
| `tracks/track2_hard_milp.py` | Hard Track 1 | Binish | Exact MILP (true binary `y`, integer `z`) with Yen's k-shortest paths. Gives the **global optimum** and the **Price of Realism**. |
| `tracks/track3_probabilistic.py` | Hard Track 2 | Aryan | Use `y_hat` from Track 1 as a probability distribution, sample server sets, solve a fast sub-LP for each, and keep the best. A **near-optimal heuristic**, much faster than the MILP. |
| `tracks/track4_custom_convex.py` | Custom Ext. I | Aatraya | Replace constant delay with **M/M/1 queuing delay** (convex). Solved by SOCP or piecewise linearisation. |
| `tracks/track5_custom_nonconvex.py` | Custom Ext. II | Aatraya | Capacity cost with **economies of scale** (`alpha * z^gamma`, concave, so non-convex). Solved with SOS2 or IPOPT multi-start. |

`compare_results.ipynb` loads every track's result file, prints comparison tables and plots.

> Status: Tracks 1 and 3 and the shared `edgeopt/` package are implemented. Tracks 2, 4, 5 and the notebook are owned by teammates (see section 9).

### How the bounds fit together

```
Z*_LP (Track 1)  <=  Z*_MILP (Track 2)  <=  Z_heuristic (Track 3)
   lower bound          true optimum          best sampled solution
```

- **Heuristic optimality gap** = (Z_heuristic - Z*_MILP) / Z*_MILP * 100%
- **Price of Realism** = (Z*_MILP - Z*_LP) / Z*_LP
- If the MILP has not been run, Track 3 reports its gap to the LP lower bound instead.

---

## 3. Repository layout

```
edge-server-optimization/
├── README.md
├── config.json              # all tunable parameters in one place
├── requirements.txt
├── data/                    # put janos-us.json (or any dataset) here
├── results/                 # each track writes <track>.json here (git-ignored)
├── edgeopt/                 # shared package used by every track
│   ├── config.py            # Config object (reads config.json + CLI overrides)
│   ├── network.py           # JSON loader, Haversine distance, Network object
│   ├── paths.py             # Yen's k-shortest paths with caching
│   ├── routing.py           # fixed-placement routing sub-LP
│   └── metrics.py           # save/load results, gap %
├── tracks/
│   ├── track1_easy_lp.py
│   ├── track2_hard_milp.py
│   ├── track3_probabilistic.py
│   ├── track4_custom_convex.py
│   └── track5_custom_nonconvex.py
└── compare_results.ipynb
```

Both `edgeopt/` and `tracks/` need an (empty) `__init__.py`.

---

## 4. Setup

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Solver: LPs are solved with **HiGHS** through SciPy, with sparse matrices. No separate solver install is needed. Track 4 and Track 5 additionally need their own solvers (CVXPY/MOSEK, IPOPT); add them to `requirements.txt` when you use them.

---

## 5. Running

Always run from the repository root, using `python -m` so the imports work.

```bash
# Track 1: Easy LP relaxation (writes results/track1_easy_lp.json)
python -m tracks.track1_easy_lp --data data/janos-us.json

# Track 3: probabilistic rounding (runs Track 1's LP internally as Phase I)
python -m tracks.track3_probabilistic --data data/janos-us.json --trials 200 --seed 42
```

Useful flags, available on both tracks unless noted:

| Flag | Meaning |
|---|---|
| `--config` | path to a config file (default `config.json`) |
| `--data` | dataset JSON |
| `--K` | number of servers |
| `--budget` | CapEx budget B |
| `--trials` | Monte Carlo trials (Track 3 only) |
| `--seed` | RNG seed, for reproducible runs (Track 3 only) |
| `--method systematic\|weighted` | sampling method (Track 3 only) |
| `--k_paths` | paths per (origin, server) pair (Track 3 only) |

Command-line flags override `config.json`.

---

## 6. Configuration (`config.json`)

| Key | Default | Meaning |
|---|---|---|
| `data_path` | `data/janos-us.json` | dataset |
| `K` | 5 | number of edge servers |
| `budget` | 150000 | CapEx budget B |
| `server_cost` | 1000 | cost per server, used if a node has no cost in the data |
| `module_capacity` | 64.0 | Gbps per capacity module |
| `module_cost` | 1000 | cost per module, used if a link has no cost in the data. **Placeholder: set the real value.** |
| `base_capacity` | 0.0 | pre-installed capacity per link (0 in the native Janos file) |
| `delay_us_per_km` | 5.0 | fibre propagation delay |
| `delay_from` | `haversine` | `haversine` computes link length from node coordinates (as in the report); `length` uses the link's own `dist`/`length` field |
| `demand_scale` | 1.0 | multiplies every demand volume. Use it to fix units (e.g. `0.001` if the file is in Mbps but capacity is in Gbps) |
| `links_directed` | `auto` | `auto` detects whether both directions are listed; or `yes` / `no` |
| `integer_modules` | `true` | Phase II solves capacity modules as whole numbers (small MILP with servers fixed). `false` = fast LP, then round modules up |
| `milp_time_limit` | 20 | seconds allowed per Phase II sub-problem when `integer_modules` is true |
| `k_paths` | 3 | Yen's k |
| `n_trials` | 200 | Monte Carlo trials |
| `seed` | 42 | RNG seed |
| `sampling` | `systematic` | `systematic` or `weighted` |
| `results_dir` | `results` | where result JSONs go |
| `field_map` | `{}` | rename JSON keys, e.g. `{"lat": "latitude"}` |

---

## 7. Using a different dataset

The code is not tied to Janos-US. Both `janos-us.json` and `germany50.json` already load without changes. They use the networkx node-link format:

```json
{"directed": false,
 "graph": {"name": "janos_us", "demands": {"0": {"1": 240.0, "2": 544.0}}},
 "nodes": [{"id": 0, "name": "Seattle", "pos": [-122.3, 47.45]}],
 "edges": [{"source": 0, "target": 2, "dist": 1093.37}]}
```

Here `pos` is `[longitude, latitude]`, `demands` is `{origin: {destination: volume}}`, and `"directed": false` makes the loader mirror each edge (janos-us: 42 edges become 84 directed links). These files carry no capacity or module cost, so `module_cost` and `base_capacity` come from `config.json`.

A flat layout is also accepted. Drop in any JSON of this shape:

```json
{
  "name": "my-network",
  "nodes":   [{"id": "Atlanta", "lat": 33.7, "lon": -84.4, "server_cost": 1000}],
  "links":   [{"source": "Atlanta", "target": "Chicago", "capacity": 0, "module_cost": 500}],
  "demands": [{"source": "Atlanta", "target": "Chicago", "volume": 12.5}]
}
```

- `server_cost`, `capacity`, `module_cost` and `length` (km) are optional; missing values fall back to `config.json`.
- Without `length`, link delay is computed from node `lat`/`lon` (Haversine distance times `delay_us_per_km`).
- Common alternative key names (`latitude`, `from`/`to`, `edges`, `volume`/`demand`, ...) are recognised automatically. For anything else, use `field_map` in `config.json`.
- Nodes or links may also be dicts keyed by id.
- Unknown node ids in links or demands raise a clear error.

If the loader cannot read a new file, fix `edgeopt/network.py` only. No track code should need to change.

---

## 8. How Tracks 1 and 3 work

### Track 1: Easy LP relaxation

Treats the problem as a **multi-commodity flow** and relaxes `y_i` to [0,1] and `z_e` to continuous values.

Each demand origin `s` sends one unit of flow. At every node, flow either passes through or is *absorbed* by a server:

```
outflow - inflow + a[s,i] = 1 if i is the origin s, else 0,      0 <= a[s,i] <= y_i
```

Then: `sum_i y_i = K`, link flow is at most `base + 64 * z_e`, and CapEx is at most B.

Why not the formulation in the original PDF? Its flow-conservation right-hand sides (`1 - y_i` at the source, `-y_i` elsewhere) sum to `1 - K`, but net flow must sum to 0, so it is infeasible for K > 1. The absorption variables `a[s,i]` fix that.

Outputs: `Z*_LP` (lower bound), the fractional vector `y_hat` (input to Track 3), fractional modules, CapEx breakdown, and mean latency.

### Track 3: Probabilistic LP rounding

**Phase I:** solve Track 1's LP to get `y_hat`, which sums to K.

**Phase II:** repeat for N trials:
1. **Sample** K distinct cities using `y_hat` as probabilities.
2. **Solve a small sub-problem** with those servers fixed (path fractions plus capacity modules). By default modules are whole numbers, so this is a small MILP, much smaller than the full MILP of Track 2. With `integer_modules: false` it is a pure LP: modules are rounded **up** afterwards, and if that breaks the CapEx budget the budget is shrunk by the overshoot and the LP re-solved.
3. Check the CapEx budget on the whole-module values.
4. Keep the best feasible solution.

Sampling methods:
- `systematic` (default): picks exactly K distinct cities and keeps `P(i selected) = y_hat_i`, so cities with `y_hat = 1` are always chosen.
- `weighted`: plain sequential sampling without replacement, as in the report.

Speed-ups:
- Sampled placements are cached, so repeated draws cost nothing.
- Trial 0 is the top-K cities by `y_hat`, a free greedy baseline.
- Paths are computed lazily, only to cities that actually get sampled.
- Demands are aggregated by origin. This is exact and shrinks the model a lot (650 demands become at most 26 origins).

### Modelling assumptions

- A demand is served by **one of the chosen servers**. Its target node in the dataset is ignored, so only the **origin** matters.
- Delay is propagation only (`km * 5 us/km`, in ms). Queuing delay belongs to Track 4.
- Capacity is purchasable on every link, so the budget is the only thing that limits capacity. Modules cost nothing in the objective, but a tiny CapEx tie-breaker is added so the LP does not buy modules it does not need. It is excluded from the reported Z.
- Link capacity is **per direction**.

---

## 9. Result file contract (for all tracks)

Every track must write `results/<track_name>.json` via `edgeopt.metrics.save_result`. The notebook relies on these keys:

| Key | Required | Meaning |
|---|---|---|
| `track` | yes | short track name |
| `dataset` | yes | dataset name |
| `objective` | yes | final objective value Z |
| `solve_time_s` or `total_time_s` | yes | wall-clock time |
| `capex` | yes | total CapEx as a single number |
| `servers` | if applicable | list of chosen city ids |
| `modules_total` | optional | total capacity modules |
| `lower_bound` | optional | Track 1 and Track 3 |
| `config` | optional | the config used, for reproducibility |

File names expected by the code and notebook:

```
results/track1_easy_lp.json
results/track2_hard_milp.json         <- Binish: Track 3 reads "objective" from this file to report the true gap
results/track3_probabilistic.json
results/track4_custom_convex.json
results/track5_custom_nonconvex.json
```

### For teammates writing tracks

- Load data and parameters with `Config.load()` and `load_network()`. Do not parse the JSON yourself.
- Reuse `PathCache` for Yen's paths.
- **Binish:** `edgeopt.routing.solve_fixed_placement(net, servers, cache, cfg)` solves the routing for a fixed server set. You can use it to cross-check your MILP's placement.
- **Aatraya:** your tracks use the same loader and config. Add new parameters (`gamma`, `alpha`, safety threshold) to `edgeopt/config.py` and `config.json`, not as hard-coded constants.
- If you need a change in `edgeopt/`, tell the team first. Every track imports it.

---

## 10. Troubleshooting

| Symptom | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: edgeopt` / `tracks` | Run from the repo root with `python -m tracks.<name>`, and check both folders have `__init__.py`. |
| `LP relaxation failed ... infeasible` | The budget cannot pay for the capacity the traffic needs. Raise `budget`, lower `module_cost`, or check units. |
| `No feasible placement found` | Raise `--trials` or `--budget`, or check `module_cost`. |
| `No 'nodes' section found` | Your JSON key names differ. Add them to `field_map` or extend the aliases in `edgeopt/network.py`. |
| `Link references unknown node` | Node ids in links or demands do not match the node list (check spelling and case). |
| Janos-US shows 42 links, not 84 | Set `links_directed` to `no` so each link is mirrored (normally automatic from the file's `directed` flag). |
| Janos-US is infeasible at budget 150000 | The traffic in the file (80,000 total) needs roughly 890 modules, so the budget or `module_cost` must be consistent with that, or `demand_scale` must fix the units. See section 12. |
| Results differ between runs | Fix `--seed`. Track 1 is deterministic; only Track 3 sampling is random. |

---

## 11. Notes for the report

- Regenerate every number in the PDFs (objective values, timings, CapEx, `y_hat` table) from an actual run. Do not reuse the example figures.
- State the Easy-track conservation fix (section 8) and how Track 3 handles modules (whole-number sub-MILP by default, or LP then round up).
- If the budget is large enough not to bind, the LP picks whole servers and Track 3 trivially matches it (one unique placement). Compare the tracks at budgets where the budget binds.
- Report the heuristic gap against the MILP when `track2_hard_milp.json` exists, and against the LP bound otherwise.
- Track 3 saves a `convergence` list (best objective so far after each trial) that the notebook can plot.

---

## 12. Parameters you must agree on as a team

The dataset files contain topology, distances and demands only. Capacity and cost parameters come from `config.json`, so every track must use the same values.

- **Demand units:** janos-us demands total 80,000 and germany50 totals 2,365. With 64-unit modules, janos-us needs about 890 modules on the LP optimum (K = 5). With `module_cost` = 1000 that is about 0.9M CapEx, far above B = 150,000. Decide whether the demand unit and module size are consistent (use `demand_scale` if not), what a module really costs, and what B should be.
- **Same values everywhere:** `K`, `budget`, `module_capacity`, `module_cost`, `server_cost` and `k_paths` must be identical across tracks, otherwise the comparison table is meaningless.