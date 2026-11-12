"""Dataset-agnostic loader: JSON -> Network (numpy arrays, integer node indices).

Expected (flexible) JSON layout:
{
  "nodes":   [{"id": "Atlanta", "lat": 33.7, "lon": -84.4, "server_cost": 1000?}, ...],
  "links":   [{"source": "A", "target": "B", "capacity": 0?, "module_cost": 500?, "length": km?}, ...],
  "demands": [{"source": "A", "target": "B", "volume": 12.5}, ...]
}
Also understood: networkx node-link JSON (what janos-us.json / germany50.json use):
  {"directed": false, "graph": {"name": ..., "demands": {"0": {"1": 240.0, ...}}},
   "nodes": [{"id": 0, "name": "Seattle", "pos": [lon, lat]}],
   "edges": [{"source": 0, "target": 2, "dist": 1093.37}]}
Key names are matched through the aliases below; override any of them with
Config.field_map (e.g. {"lat": "y_coord"}).  Nodes/links may also be dicts keyed by id.
"""
from dataclasses import dataclass
import json
import numpy as np

_ALIASES = {
    "id": ["id", "name", "node", "label", "city"],
    "lat": ["lat", "latitude", "y"],
    "lon": ["lon", "lng", "long", "longitude", "x"],
    "server_cost": ["server_cost", "deploy_cost", "facility_cost"],
    "src": ["source", "src", "from", "u", "origin"],
    "dst": ["target", "dst", "to", "v", "destination"],
    "cap": ["capacity", "base_capacity", "pre_installed_capacity", "preinstalled_capacity"],
    "mod_cost": ["module_cost", "cost_per_module", "capacity_cost", "unit_cost"],
    "length": ["length", "dist", "distance_km", "km"],
    "pos": ["pos", "position", "coords", "coordinates"],
    "volume": ["volume", "demand", "demand_value", "demandValue", "bandwidth", "value"],
}
_SECTIONS = {
    "nodes": ["nodes", "cities", "vertices"],
    "links": ["links", "edges", "arcs"],
    "demands": ["demands", "traffic", "commodities"],
}


@dataclass
class Network:
    name: str
    node_ids: list
    lat: np.ndarray
    lon: np.ndarray
    server_cost: np.ndarray   # (N,)
    eu: np.ndarray            # (M,) edge tail index
    ev: np.ndarray            # (M,) edge head index
    delay_ms: np.ndarray      # (M,)
    cap_base: np.ndarray      # (M,)
    mod_cost: np.ndarray      # (M,)
    src_nodes: np.ndarray     # (S,) distinct demand origins
    volume: np.ndarray        # (S,) total demand per origin (aggregated)

    @property
    def n(self): return len(self.node_ids)
    @property
    def m(self): return len(self.eu)


def haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi, dl = p2 - p1, np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _pick(rec, name, fmap, default=None):
    keys = [fmap[name]] if name in fmap else _ALIASES[name]
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return default


def _section(raw, name):
    for where in (raw, raw.get("graph", {}) if isinstance(raw.get("graph"), dict) else {}):
        for k in _SECTIONS[name]:
            if k in where:
                return where[k]
    raise KeyError(f"No '{name}' section found (looked for {_SECTIONS[name]}). "
                   f"Top-level keys: {list(raw)}")


def _records(sec):
    if isinstance(sec, dict):
        return [{"id": k, **(v if isinstance(v, dict) else {})} for k, v in sec.items()]
    return list(sec)


def load_network(path, cfg):
    with open(path) as f:
        raw = json.load(f)
    raw = raw.get("network", raw) if isinstance(raw, dict) else raw
    fm = cfg.field_map

    # ---- nodes (match links/demands by id; show the human-readable name when available)
    nodes = _records(_section(raw, "nodes"))
    keys = [str(_pick(r, "id", fm)) for r in nodes]
    labels = [str(r["name"]) if r.get("name") not in (None, "") else k for r, k in zip(nodes, keys)]
    if len(set(labels)) != len(labels):
        labels = keys
    idx = {k: i for i, k in enumerate(keys)}
    ids = labels

    def _coord(r, which):
        v = _pick(r, which, fm)
        if v is None:
            pos = _pick(r, "pos", fm)           # [lon, lat] (x, y) order
            if pos is not None:
                v = pos[0] if which == "lon" else pos[1]
        return float(v) if v is not None else np.nan

    lat = np.array([_coord(r, "lat") for r in nodes])
    lon = np.array([_coord(r, "lon") for r in nodes])
    scost = np.array([float(_pick(r, "server_cost", fm, cfg.server_cost)) for r in nodes])

    # ---- links (auto-detect whether the file already lists both directions)
    links = _records(_section(raw, "links"))
    pairs = []
    for r in links:
        u, v = str(_pick(r, "src", fm)), str(_pick(r, "dst", fm))
        if u not in idx or v not in idx:
            raise ValueError(f"Link references unknown node: {u} -> {v}")
        pairs.append((idx[u], idx[v], r))
    pset = {(u, v) for u, v, _ in pairs}
    if cfg.links_directed in ("yes", "no"):
        directed = cfg.links_directed == "yes"
    elif isinstance(raw.get("directed"), bool):
        directed = raw["directed"]               # networkx node-link files carry this flag
    else:
        directed = any((v, u) in pset for u, v in pset)

    eu, ev, delay, cap, mcost, seen = [], [], [], [], [], set()
    for u, v, r in pairs:
        length = None if cfg.delay_from == "haversine" else _pick(r, "length", fm)
        if length is None:
            if np.isnan([lat[u], lon[u], lat[v], lon[v]]).any():
                raise ValueError("Need node lat/lon or a link 'length' to compute delay.")
            length = haversine_km(lat[u], lon[u], lat[v], lon[v])
        c = float(_pick(r, "cap", fm, cfg.base_capacity))
        mc = float(_pick(r, "mod_cost", fm, cfg.module_cost))
        for a, b in ([(u, v)] if directed else [(u, v), (v, u)]):
            if (a, b) in seen or a == b:
                continue
            seen.add((a, b))
            eu.append(a); ev.append(b)
            delay.append(float(length) * cfg.delay_us_per_km / 1000.0)  # ms
            cap.append(c); mcost.append(mc)

    # ---- demands, aggregated by origin (exact: the server choice depends only on the origin)
    vol = np.zeros(len(ids))
    dsec = _section(raw, "demands")
    if isinstance(dsec, dict) and dsec and all(isinstance(v, dict) for v in dsec.values()) \
            and not any(k in next(iter(dsec.values())) for k in _ALIASES["volume"]):
        flat = [{"source": a, "target": b, "volume": w} for a, row in dsec.items() for b, w in row.items()]
    else:
        flat = _records(dsec)
    for r in flat:
        sk = str(_pick(r, "src", fm))
        if sk not in idx:
            raise ValueError(f"Demand references unknown node: {sk}")
        vol[idx[sk]] += float(_pick(r, "volume", fm, 0.0)) * cfg.demand_scale
    src_nodes = np.flatnonzero(vol > 0)

    return Network(
        name=str((raw.get("graph") or {}).get("name") or raw.get("name") or path),
        node_ids=ids, lat=lat, lon=lon, server_cost=scost,
        eu=np.array(eu), ev=np.array(ev), delay_ms=np.array(delay),
        cap_base=np.array(cap), mod_cost=np.array(mcost),
        src_nodes=src_nodes, volume=vol[src_nodes],
    )