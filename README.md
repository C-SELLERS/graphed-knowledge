# GraphedKnowledge

A graph-structured knowledge base for AI agents. Notes are plain Markdown files with YAML frontmatter and `[[wiki-links]]`; retrieval combines TF-IDF keyword search with Personalized PageRank graph traversal for context-aware results.

Designed to drop into any project and give any AI agent persistent memory across sessions via an MCP server.

## Features

- **Pure Python, zero runtime dependencies** — the core library (`graphed_kb`) has no third-party requirements
- **Obsidian-compatible storage** — nodes are `.md` files you can read and edit directly
- **Three retrieval strategies** — `keyword` (TF-IDF), `graph` (BFS + PageRank), `hybrid` (both, re-ranked)
- **MCP server** — exposes five tools (`kb_recall`, `kb_remember`, `kb_associate`, `kb_reflect`, `kb_stats`) to any MCP-compatible agent

## Installation

```bash
pip install graphed-knowledge
```

For MCP server support:

```bash
pip install "graphed-knowledge[mcp]"
```

## Quick start

### As a Python library

```python
from graphed_kb import AgentMemory

mem = AgentMemory("./my-kb")

# Store a note
mem.remember(
    title="Transformer Attention",
    content="Scaled dot-product attention: softmax(QKᵀ/√d)V ...",
    tags=["transformers", "attention"],
    links_to=["positional-encoding", "multi-head-attention"],
)

# Retrieve relevant context
results = mem.recall("how does attention scale with sequence length", strategy="hybrid")
for r in results:
    print(r.node.title, r.score)

# Get context string ready to inject into a prompt
context = mem.recall("attention mechanism", as_context=True)
```

### Run the MCP server manually

```bash
python mcp_server.py --kb-path /path/to/kb
```

## API reference

### `AgentMemory(kb_path)`

The main interface. Wraps `KnowledgeGraph` + `Retriever` with a simple high-level API.

| Method | Description |
|--------|-------------|
| `remember(title, content, tags, links_to)` | Write a new node to the KB |
| `recall(query, strategy, top_k, as_context)` | Retrieve relevant nodes |
| `associate(from_node, to_node)` | Add a wiki-link between two existing nodes |
| `reflect(top_k)` | Return the highest-PageRank nodes (knowledge hubs) |
| `stats()` | Node count, edge count, KB path |

### Retrieval strategies

| Strategy | When to use |
|----------|-------------|
| `keyword` | Exact term lookup, debugging |
| `graph` | Related concepts you might not name directly |
| `hybrid` | Default — best for open-ended questions |

### MCP tools (for agents)

| Tool | Description |
|------|-------------|
| `kb_recall` | Search the KB with keyword/graph/hybrid strategy |
| `kb_remember` | Persist a new note |
| `kb_associate` | Link two existing nodes |
| `kb_reflect` | Surface the most central nodes |
| `kb_stats` | KB size and path |

## Node format

Nodes are standard Markdown files the agent writes and reads:

```markdown
---
title: Attention Mechanism
date: 2025-01-15
tags: [transformers, attention, seeded]
---

# Attention Mechanism

Scaled dot-product attention computes ...

## Related
- [[Positional Encoding]]
- [[Multi-Head Attention]]
```

Files live in whatever directory you point the library at — no database, no lock files, fully inspectable and version-controllable.

## Project layout

```
graphed_kb/          # installable library
  parser.py          # Markdown + frontmatter → KBNode
  graph.py           # KnowledgeGraph, BFS, PageRank
  retrieval.py       # TF-IDF, graph search, hybrid
  agent_memory.py    # High-level AgentMemory interface
mcp_server.py        # FastMCP server (requires mcp extra)
```

## Requirements

- Python 3.9+
- No runtime dependencies for the core library
- `mcp>=1.0` for the MCP server (`pip install "graphed-knowledge[mcp]"`)

## License

MIT
