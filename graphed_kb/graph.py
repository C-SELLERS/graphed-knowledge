"""
Knowledge graph built from KBNode link relationships.

Supports:
  - Forward links (A links to B)
  - Backlinks (B is linked from A)
  - BFS/DFS context expansion
  - Personalized PageRank for scoring
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from .parser import KBNode, load_directory


class KnowledgeGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, KBNode] = {}
        # forward edges: node_id -> set of node_ids it links to
        self._edges: dict[str, set[str]] = {}
        # reverse edges: node_id -> set of node_ids that link to it
        self._backlinks: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, kb_path: str | Path) -> "KnowledgeGraph":
        """Load all markdown files from a directory."""
        self.nodes = load_directory(kb_path)
        self._build_edges()
        return self

    def add_node(self, node: KBNode) -> None:
        self.nodes[node.id] = node
        self._build_edges()

    def _build_edges(self) -> None:
        self._edges = {nid: set() for nid in self.nodes}
        self._backlinks = {nid: set() for nid in self.nodes}

        for nid, node in self.nodes.items():
            for target in node.links:
                if target in self.nodes:
                    self._edges[nid].add(target)
                    self._backlinks[target].add(nid)

    # ------------------------------------------------------------------
    # Graph queries
    # ------------------------------------------------------------------

    def neighbors(self, node_id: str, include_backlinks: bool = True) -> set[str]:
        """Return all direct neighbors (forward + optional backlinks)."""
        result = set(self._edges.get(node_id, set()))
        if include_backlinks:
            result |= self._backlinks.get(node_id, set())
        return result

    def bfs(
        self,
        seeds: list[str],
        max_hops: int = 2,
        include_backlinks: bool = True,
    ) -> dict[str, int]:
        """
        BFS from seed nodes. Returns {node_id: hop_distance}.
        Seeds themselves are at distance 0.
        """
        visited: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque()

        for seed in seeds:
            if seed in self.nodes:
                visited[seed] = 0
                queue.append((seed, 0))

        while queue:
            current, depth = queue.popleft()
            if depth >= max_hops:
                continue
            for neighbor in self.neighbors(current, include_backlinks):
                if neighbor not in visited:
                    visited[neighbor] = depth + 1
                    queue.append((neighbor, depth + 1))

        return visited

    def pagerank(
        self,
        personalization: dict[str, float] | None = None,
        damping: float = 0.85,
        iterations: int = 50,
        tol: float = 1e-6,
    ) -> dict[str, float]:
        """
        Compute (personalized) PageRank over the graph.

        personalization: optional seed weights {node_id: weight}. When provided
        this becomes Personalized PageRank — the score reflects relevance to
        the seed set rather than global importance.
        """
        n = len(self.nodes)
        if n == 0:
            return {}

        node_ids = list(self.nodes.keys())

        # Normalize personalization
        if personalization:
            total = sum(personalization.values())
            p_vec = {nid: personalization.get(nid, 0.0) / total for nid in node_ids}
        else:
            p_vec = {nid: 1.0 / n for nid in node_ids}

        # Initialize ranks
        rank = {nid: 1.0 / n for nid in node_ids}

        # Out-degree for normalization
        out_deg = {nid: len(self._edges.get(nid, set())) for nid in node_ids}

        for _ in range(iterations):
            new_rank: dict[str, float] = {}
            for nid in node_ids:
                incoming = self._backlinks.get(nid, set())
                link_score = sum(
                    rank[src] / out_deg[src]
                    for src in incoming
                    if out_deg.get(src, 0) > 0
                )
                new_rank[nid] = (1 - damping) * p_vec[nid] + damping * link_score

            # Normalize
            total = sum(new_rank.values()) or 1.0
            new_rank = {k: v / total for k, v in new_rank.items()}

            # Check convergence
            delta = sum(abs(new_rank[k] - rank[k]) for k in node_ids)
            rank = new_rank
            if delta < tol:
                break

        return rank

    def subgraph(self, node_ids: set[str]) -> "KnowledgeGraph":
        """Return a new KnowledgeGraph containing only the specified nodes."""
        sub = KnowledgeGraph()
        sub.nodes = {nid: self.nodes[nid] for nid in node_ids if nid in self.nodes}
        sub._build_edges()
        return sub

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        total_edges = sum(len(v) for v in self._edges.values())
        isolated = sum(
            1 for nid in self.nodes
            if not self._edges.get(nid) and not self._backlinks.get(nid)
        )
        return {
            "nodes": len(self.nodes),
            "edges": total_edges,
            "isolated_nodes": isolated,
            "avg_degree": total_edges / len(self.nodes) if self.nodes else 0,
        }

    def __repr__(self) -> str:
        s = self.stats()
        return f"KnowledgeGraph(nodes={s['nodes']}, edges={s['edges']})"
