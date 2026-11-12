"""Single source of truth for every tunable parameter (shared by all tracks)."""
from dataclasses import dataclass, field, asdict
import json
import os


@dataclass
class Config:
    data_path: str = "data/janos-us.json"
    K: int = 5                       # number of edge servers
    budget: float = 150_000.0        # CapEx budget B
    server_cost: float = 1000.0      # fallback if a node has no server cost
    module_capacity: float = 64.0    # Gbps per capacity module (C_mod)
    module_cost: float = 1000.0      # fallback if a link has no module cost (PLACEHOLDER: use real value)
    base_capacity: float = 0.0       # fallback pre-installed link capacity
    delay_us_per_km: float = 5.0     # propagation delay in fibre
    demand_scale: float = 1.0        # multiply all demand volumes (unit conversion, e.g. 0.001 for Mbps -> Gbps)
    delay_from: str = "haversine"    # "haversine" (coords, as in the report) | "length" (link length/dist field if present)
    links_directed: str = "auto"     # "auto" | "yes" | "no"
    integer_modules: bool = True     # Phase II: solve modules as integers (small MILP, y fixed). False = fast LP + round up
    milp_time_limit: float = 20.0    # seconds per Phase II sub-problem when integer_modules is True
    k_paths: int = 3                 # Yen's k per (source, server) pair
    n_trials: int = 200              # Monte Carlo trials (Hard Track 2)
    seed: int = 42
    sampling: str = "systematic"     # "systematic" | "weighted"
    results_dir: str = "results"
    field_map: dict = field(default_factory=dict)  # override JSON key names, e.g. {"lat": "latitude"}

    @classmethod
    def load(cls, path="config.json", **overrides):
        cfg = cls()
        if path and os.path.exists(path):
            with open(path) as f:
                for k, v in json.load(f).items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
        for k, v in overrides.items():
            if v is not None and hasattr(cfg, k):
                setattr(cfg, k, v)
        return cfg

    def to_dict(self):
        return asdict(self)