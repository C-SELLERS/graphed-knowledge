"""
High-level agent memory interface over a KnowledgeGraph.

An AI agent uses this to:
  - remember()  : persist a new fact/note into the KB
  - recall()    : retrieve relevant context for a query
  - associate() : explicitly link two nodes
  - reflect()   : surface high-pagerank / hub nodes (what does the agent
                  "know best"?)
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from typing import Literal

from .graph import KnowledgeGraph
from .parser import KBNode, _slugify, parse_node
from .retrieval import Retriever, ScoredResult


class AgentMemory:
    def __init__(self, kb_path: str | Path) -> None:
        self.kb_path = Path(kb_path)
        self.kb_path.mkdir(parents=True, exist_ok=True)
        self.graph = KnowledgeGraph().load(self.kb_path)
        self.retriever = Retriever(self.graph)

    def reload(self) -> None:
        """Re-parse the KB directory (call after external changes)."""
        self.graph.load(self.kb_path)
        self.retriever.rebuild_index()

    # ------------------------------------------------------------------
    # remember — write a new node into the KB
    # ------------------------------------------------------------------

    def remember(
        self,
        title: str,
        content: str,
        tags: list[str] | None = None,
        links_to: list[str] | None = None,
        filename: str | None = None,
    ) -> KBNode:
        """
        Persist a new fact/note into the knowledge base.

        Args:
            title:     Human-readable title.
            content:   Markdown body text.
            tags:      Optional list of tag strings (without #).
            links_to:  List of node ids or titles to link to via [[wiki-links]].
            filename:  Override the generated filename stem.

        Returns the parsed KBNode.
        """
        slug = filename or _slugify(title)
        file_path = self.kb_path / f"{slug}.md"

        tag_line = ""
        if tags:
            tag_line = f"tags: [{', '.join(tags)}]\n"

        link_block = ""
        if links_to:
            lines = []
            for t in links_to:
                slug = _slugify(t)
                display = self.graph.nodes[slug].title if slug in self.graph.nodes else slug.replace("_", " ").title()
                lines.append(f"- [[{slug}|{display}]]")
            link_block = "\n\n## Related\n" + "\n".join(lines)

        timestamp = datetime.utcnow().strftime("%Y-%m-%d")
        frontmatter = f"---\ntitle: {title}\ndate: {timestamp}\n{tag_line}---\n\n"
        full_content = frontmatter + f"# {title}\n\n" + textwrap.dedent(content) + link_block

        file_path.write_text(full_content, encoding="utf-8")

        node = parse_node(file_path)
        self.graph.add_node(node)
        self.retriever.rebuild_index()
        return node

    # ------------------------------------------------------------------
    # recall — retrieve relevant context
    # ------------------------------------------------------------------

    def recall(
        self,
        query: str,
        strategy: Literal["keyword", "graph", "hybrid"] = "hybrid",
        top_k: int = 5,
        max_hops: int = 2,
        as_context: bool = False,
    ) -> list[ScoredResult] | str:
        """
        Retrieve relevant knowledge for a query.

        Args:
            query:      Natural-language query.
            strategy:   'keyword', 'graph', or 'hybrid'.
            top_k:      Max results to return.
            max_hops:   Graph expansion depth (graph/hybrid only).
            as_context: If True, return a formatted string for LLM injection.
        """
        if as_context:
            return self.retriever.get_context(query, strategy=strategy, top_k=top_k, max_hops=max_hops)
        if strategy == "keyword":
            return self.retriever.keyword_search(query, top_k=top_k)
        elif strategy == "graph":
            return self.retriever.graph_search(query, top_k=top_k, max_hops=max_hops)
        else:
            return self.retriever.hybrid_search(query, top_k=top_k, max_hops=max_hops)

    # ------------------------------------------------------------------
    # associate — add an explicit link between two nodes
    # ------------------------------------------------------------------

    def associate(self, from_node: str, to_node: str) -> bool:
        """
        Add a [[wiki-link]] from from_node -> to_node by editing the file.
        Returns True if the link was added, False if it already existed.
        """
        src_id = _slugify(from_node)
        dst_id = _slugify(to_node)

        if src_id not in self.graph.nodes or dst_id not in self.graph.nodes:
            raise KeyError(f"Node not found: {src_id!r} or {dst_id!r}")

        src_node = self.graph.nodes[src_id]
        if dst_id in src_node.links:
            return False  # already linked

        dst_title = self.graph.nodes[dst_id].title
        link_text = f"\n- [[{dst_id}|{dst_title}]]"

        content = src_node.path.read_text(encoding="utf-8")
        if "## Related" in content:
            content = content.rstrip() + link_text + "\n"
        else:
            content = content.rstrip() + f"\n\n## Related\n{link_text}\n"

        src_node.path.write_text(content, encoding="utf-8")
        self.reload()
        return True

    # ------------------------------------------------------------------
    # reflect — surface hub nodes (high PageRank)
    # ------------------------------------------------------------------

    def reflect(self, top_k: int = 5) -> list[tuple[str, float]]:
        """
        Return the top_k highest-PageRank nodes — the knowledge hubs
        the agent has built its understanding around.
        """
        pr = self.graph.pagerank()
        ranked = sorted(pr.items(), key=lambda x: x[1], reverse=True)
        return [(self.graph.nodes[nid].title, score) for nid, score in ranked[:top_k] if nid in self.graph.nodes]

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            **self.graph.stats(),
            "kb_path": str(self.kb_path),
        }
