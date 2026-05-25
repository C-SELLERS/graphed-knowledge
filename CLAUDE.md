# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Run all tests:**
```bash
python -m pytest tests/
# or without pytest:
python -m unittest discover tests/
```

**Run a single test file:**
```bash
python -m pytest tests/test_parser.py
# or:
python tests/test_parser.py
```

**Run the interactive demo:**
```bash
python demo.py
```

**Run the interactive KB chat (multi-session):**
```bash
python chat.py           # new session or resume from list
python chat.py <id>      # resume specific session
python chat.py --new     # always start fresh
```

**Initialise a KB for any project:**
```bash
python -m graphed_kb init <project_path>   # seeds from README/CLAUDE.md, writes MCP config
python -m graphed_kb init .                # current directory
python -m graphed_kb init /path/to/proj --kb-dir .kb
```

**Run the MCP server (network / SSE — default):**
```bash
python mcp_server.py                                    # SSE on 0.0.0.0:8000
python mcp_server.py --port 9000                        # custom port
python mcp_server.py --host 127.0.0.1 --port 8000      # localhost only
python mcp_server.py --transport stdio                  # stdio fallback

# Seed a 'default' KB at startup (registers it if not already present):
python mcp_server.py --kb-path ./knowledge_base --kb-description "ML knowledge base"

# Custom registry file location:
python mcp_server.py --registry /data/my_registry.json
```

**Install dependencies:**
```bash
pip install mcp>=1.0    # required for mcp_server.py
pip install pytest      # optional, for tests only
```

## Architecture

The system turns a directory of Obsidian-style Markdown files into a queryable knowledge graph.

### Data flow

```
.md files  ->  parser.py (KBNode)  ->  graph.py (KnowledgeGraph)  ->  retrieval.py (Retriever)
                                                                              ^
                                                                    agent_memory.py (AgentMemory)
```

### Module responsibilities

- **`parser.py`** — Parses `.md` files into `KBNode` dataclasses. Extracts YAML frontmatter, `[[wiki-links]]` (converted to slugified node IDs), `#tags`, and strips Markdown for plain-text indexing. Node IDs are the filename stem lowercased with spaces replaced by underscores.

- **`graph.py`** — `KnowledgeGraph` holds nodes and forward/backlink edge sets. Key operations: `bfs()` for neighborhood expansion, `pagerank()` for global or personalized importance scoring, `subgraph()` for slicing.

- **`retrieval.py`** — `Retriever` implements three strategies against a `KnowledgeGraph`:
  1. `keyword_search` — flat TF-IDF scoring, no graph traversal (baseline)
  2. `graph_search` — keyword seeds + BFS expansion + Personalized PageRank re-scoring; surfaces nodes *connected* to relevant ones even without direct term matches
  3. `hybrid_search` — linear blend of keyword and graph scores (alpha parameter, default 0.5)

  `get_context()` formats results as a string ready for LLM prompt injection.

- **`agent_memory.py`** — `AgentMemory` is the high-level agent interface over a KB directory:
  - `remember()` — writes a new `.md` file and registers it in the graph
  - `recall()` — delegates to `Retriever` for retrieval
  - `associate()` — appends a `[[wiki-link]]` between two existing nodes by editing the source file
  - `reflect()` — returns top-PageRank nodes (what the agent "knows best")

- **`cli.py`** — `python -m graphed_kb init <path>` CLI. Seeds a new KB from the project's README and CLAUDE.md (splits by H2 sections into individual nodes), then writes an MCP server entry into `.claude/settings.local.json` pointing at the new KB directory.

- **`mcp_server.py`** — FastMCP server with multi-KB instance management. Exposes a `KBRegistry` (persisted to `~/.graphed_kb/registry.json` by default) and eight MCP tools: `kb_list`, `kb_create`, `kb_delete` for registry management, and `kb_recall`, `kb_remember`, `kb_associate`, `kb_reflect`, `kb_stats` for per-KB operations (all require a `kb_id` arg). Runs over SSE (HTTP) by default for network access; falls back to stdio with `--transport stdio`.

- **`chat.py`** — Interactive multi-session REPL. Sessions are persisted to `sessions/<id>.json` and can be resumed by ID.

### Knowledge base format

Nodes are plain `.md` files in `knowledge_base/`. Relationships are expressed as `[[wiki-links]]` in the body. The slug of the link target (lowercased, spaces to underscores) must match the target file's stem for the edge to be recognized. Tags come from YAML frontmatter (`tags: [a, b]`) or inline `#tag` syntax.

### Graph scoring internals

- `graph_search` blends normalized keyword score (40%) and normalized Personalized PageRank (60%), then applies a `1/(1+hop)` decay so direct keyword matches outscore graph-traversed neighbors.
- `hybrid_search` normalizes both `keyword_search` and `graph_search` scores independently before blending via alpha.
- After calling `graph.add_node()` or `associate()`, call `retriever.rebuild_index()` (done automatically inside `AgentMemory`).
