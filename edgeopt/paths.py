"""Lazy, cached Yen's k-shortest paths (networkx.shortest_simple_paths is Yen's algorithm)."""
from itertools import islice
import numpy as np
import networkx as nx


class PathCache:
    def __init__(self, net, k):
        self.k = k
        self.G = nx.DiGraph()
        self.G.add_nodes_from(range(net.n))
        for i, (u, v) in enumerate(zip(net.eu, net.ev)):
            self.G.add_edge(int(u), int(v), delay=float(net.delay_ms[i]), idx=i)
        self._cache = {}

    def get(self, s, t):
        """List of (delay_ms, edge_index_array) for up to k loopless paths s -> t."""
        key = (int(s), int(t))
        if key not in self._cache:
            if s == t:
                out = [(0.0, np.empty(0, dtype=np.int64))]
            else:
                try:
                    paths = list(islice(nx.shortest_simple_paths(self.G, key[0], key[1], weight="delay"), self.k))
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    paths = []
                out = []
                for p in paths:
                    es = [self.G[a][b] for a, b in zip(p[:-1], p[1:])]
                    out.append((sum(e["delay"] for e in es), np.array([e["idx"] for e in es], dtype=np.int64)))
            self._cache[key] = out
        return self._cache[key]
