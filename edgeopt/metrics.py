"""Shared result format: every track writes results/<track>.json so the notebook can load them all."""
import json
import os
import numpy as np


def _enc(o):
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, np.ndarray): return o.tolist()
    raise TypeError(type(o))


def save_result(name, data, cfg):
    os.makedirs(cfg.results_dir, exist_ok=True)
    path = os.path.join(cfg.results_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_enc)
    return path


def load_result(name, cfg):
    with open(os.path.join(cfg.results_dir, f"{name}.json")) as f:
        return json.load(f)


def gap_pct(value, reference):
    """(value - reference) / reference * 100.  Use for heuristic gap and Price of Realism."""
    return float("nan") if not reference else (value - reference) / reference * 100.0
