"""
Retrieval strategies over a KnowledgeGraph.

Three strategies:
  1. keyword_search  - TF-IDF-style scoring, no graph structure
  2. graph_search    - Find seed nodes via keywords, expand via BFS, re-score
                       with personalized PageRank
  3. hybrid_search   - Linear combination of both scores

This lets you directly compare flat vs graph-augmented retrieval.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from .graph import KnowledgeGraph
from .parser import KBNode


@dataclass
class ScoredResult:
    node_id: str
    title: str
    score: float
    hop_distance: int | None        # None for keyword-only results
    matched_terms: list[str]
    snippet: str                    # first ~200 chars of plain_text

    def __repr__(self) -> str:
        hop = f"hop={self.hop_distance}" if self.hop_distance is not None else "keyword"
        return f"ScoredResult({self.node_id!r}, score={self.score:.4f}, {hop})"


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_idf(graph: KnowledgeGraph) -> dict[str, float]:
    """Compute inverse-document-frequency for every term in the KB."""
    n = len(graph.nodes)
    df: dict[str, int] = {}
    for node in graph.nodes.values():
        seen = set(_tokenize(node.plain_text + " " + node.title))
        for term in seen:
            df[term] = df.get(term, 0) + 1
    return {term: math.log((n + 1) / (count + 1)) + 1.0 for term, count in df.items()}


def _tf_score(tokens: list[str], query_terms: set[str]) -> float:
    if not tokens:
        return 0.0
    freq = {}
    for t in tokens:
        freq[t] = freq.get(t, 0) + 1
    return sum(freq.get(qt, 0) / len(tokens) for qt in query_terms)


def _snippet(plain_text: str, length: int = 200) -> str:
    return plain_text[:length] + ("..." if len(plain_text) > length else "")


class Retriever:
    def __init__(self, graph: KnowledgeGraph) -> None:
        self.graph = graph
        self._idf = _build_idf(graph)

    def rebuild_index(self) -> None:
        """Call after the graph is modified."""
        self._idf = _build_idf(self.graph)

    # ------------------------------------------------------------------
    # Strategy 1: keyword search (flat, no graph)
    # ------------------------------------------------------------------

    def keyword_search(self, query: str, top_k: int = 5) -> list[ScoredResult]:
        """
        Score every node by TF-IDF against the query. No graph traversal.
        This is the baseline 'flat' retrieval.
        """
        query_terms = set(_tokenize(query))
        if not query_terms:
            return []

        results: list[ScoredResult] = []
        for nid, node in self.graph.nodes.items():
            tokens = _tokenize(node.plain_text + " " + node.title)
            tf = _tf_score(tokens, query_terms)
            idf_boost = sum(self._idf.get(qt, 1.0) for qt in query_terms if qt in set(tokens))
            score = tf * idf_boost

            matched = [qt for qt in query_terms if qt in set(tokens)]
            if score > 0:
                results.append(ScoredResult(
                    node_id=nid,
                    title=node.title,
                    score=score,
                    hop_distance=None,
                    matched_terms=matched,
                    snippet=_snippet(node.plain_text),
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Strategy 2: graph search (graph-augmented retrieval)
    # ------------------------------------------------------------------

    def graph_search(
        self,
        query: str,
        top_k: int = 5,
        max_hops: int = 2,
        seed_k: int = 3,
    ) -> list[ScoredResult]:
        """
        1. Find top seed_k keyword-matching nodes.
        2. BFS-expand from seeds up to max_hops.
        3. Re-score with Personalized PageRank seeded on the keyword scores.
        4. Blend PPR score with keyword relevance.

        This surfaces nodes that are *connected* to relevant nodes even if
        they don't directly contain the query terms.
        """
        query_terms = set(_tokenize(query))
        if not query_terms:
            return []

        # Step 1: keyword seed scores
        kw_results = self.keyword_search(query, top_k=max(seed_k, top_k))
        seed_scores = {r.node_id: r.score for r in kw_results[:seed_k]}

        if not seed_scores:
            return []

        # Step 2: BFS expansion
        bfs_distances = self.graph.bfs(list(seed_scores.keys()), max_hops=max_hops)

        # Step 3: Personalized PageRank from seeds
        ppr = self.graph.pagerank(personalization=seed_scores)

        # Step 4: score all reached nodes
        kw_scores = {r.node_id: r.score for r in kw_results}
        max_kw = max(kw_scores.values()) if kw_scores else 1.0
        max_ppr = max(ppr.values()) if ppr else 1.0

        results: list[ScoredResult] = []
        for nid, hop in bfs_distances.items():
            node = self.graph.nodes.get(nid)
            if node is None:
                continue
            norm_kw = kw_scores.get(nid, 0.0) / max_kw
            norm_ppr = ppr.get(nid, 0.0) / max_ppr
            # Decay score by hop distance so direct matches score highest
            hop_decay = 1.0 / (1 + hop)
            score = (0.4 * norm_kw + 0.6 * norm_ppr) * hop_decay

            tokens = set(_tokenize(node.plain_text + " " + node.title))
            matched = [qt for qt in query_terms if qt in tokens]
            results.append(ScoredResult(
                node_id=nid,
                title=node.title,
                score=score,
                hop_distance=hop,
                matched_terms=matched,
                snippet=_snippet(node.plain_text),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Strategy 3: hybrid search
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.5,
        max_hops: int = 2,
    ) -> list[ScoredResult]:
        """
        Linear blend of keyword and graph scores.
        alpha=1.0 => pure keyword, alpha=0.0 => pure graph.
        """
        kw = {r.node_id: r for r in self.keyword_search(query, top_k=len(self.graph.nodes))}
        gr = {r.node_id: r for r in self.graph_search(query, top_k=len(self.graph.nodes), max_hops=max_hops)}

        max_kw_score = max((r.score for r in kw.values()), default=1.0)
        max_gr_score = max((r.score for r in gr.values()), default=1.0)

        all_ids = set(kw) | set(gr)
        results: list[ScoredResult] = []
        for nid in all_ids:
            norm_kw = (kw[nid].score / max_kw_score) if nid in kw else 0.0
            norm_gr = (gr[nid].score / max_gr_score) if nid in gr else 0.0
            score = alpha * norm_kw + (1 - alpha) * norm_gr

            base = kw.get(nid) or gr[nid]
            results.append(ScoredResult(
                node_id=nid,
                title=base.title,
                score=score,
                hop_distance=gr[nid].hop_distance if nid in gr else None,
                matched_terms=base.matched_terms,
                snippet=base.snippet,
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # Context assembly (formats results for injection into an LLM prompt)
    # ------------------------------------------------------------------

    def get_context(
        self,
        query: str,
        strategy: Literal["keyword", "graph", "hybrid"] = "hybrid",
        top_k: int = 5,
        max_hops: int = 2,
    ) -> str:
        """
        Returns a formatted context string ready to inject into an LLM prompt.
        """
        if strategy == "keyword":
            results = self.keyword_search(query, top_k=top_k)
        elif strategy == "graph":
            results = self.graph_search(query, top_k=top_k, max_hops=max_hops)
        else:
            results = self.hybrid_search(query, top_k=top_k, max_hops=max_hops)

        if not results:
            return "[No relevant knowledge found]"

        lines = [f"# Retrieved Knowledge ({strategy} strategy, query: {query!r})\n"]
        for i, r in enumerate(results, 1):
            hop_info = f" (hop={r.hop_distance})" if r.hop_distance is not None else ""
            lines.append(f"## {i}. {r.title}{hop_info}  [score={r.score:.3f}]")
            lines.append(r.snippet)
            if r.matched_terms:
                lines.append(f"_Matched terms: {', '.join(r.matched_terms)}_")
            lines.append("")

        return "\n".join(lines)
