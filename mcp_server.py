"""
MCP server exposing GraphedKnowledge with per-user, multi-KB instance management.

Each Bearer token maps to a user ID. Each user gets their own isolated registry
and KB store under <MCP_DATA_DIR>/<user_id>/.

Token map file (MCP_TOKEN_MAP, default: <MCP_DATA_DIR>/tokens.json):
    {
      "<token>": "<user_id>",
      ...
    }

Environment variables:
    MCP_DATA_DIR    Base directory for all user data (default: ~/.graphed_kb)
    MCP_TOKEN_MAP   Path to token->user_id JSON map (default: <MCP_DATA_DIR>/tokens.json)
    MCP_HOST        Bind host (default: 0.0.0.0)
    MCP_PORT        Bind port (default: 8000)
    MCP_TRANSPORT   'sse' or 'stdio' (default: sse)
    MCP_KB_PATH     Seed path for a 'default' KB on first run (optional)
    MCP_KB_DESCRIPTION  Description for the seeded KB (default: "Default knowledge base")

Registry tools (no kb_id needed):
  kb_list       - list all registered knowledge bases with IDs and descriptions
  kb_create     - register and initialise a new knowledge base
  kb_delete     - unregister a knowledge base (files are NOT deleted from disk)

Per-KB tools (require kb_id — use kb_list to discover):
  kb_recall     - retrieve relevant context from a specific KB
  kb_remember   - persist a new note into a specific KB
  kb_associate  - add an explicit wiki-link between two nodes in a specific KB
  kb_reflect    - surface top hub nodes by PageRank in a specific KB
  kb_stats      - return graph stats for a specific KB
"""

from __future__ import annotations

import contextvars
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from mcp.server.fastmcp import FastMCP
from graphed_kb.agent_memory import AgentMemory


# ---------------------------------------------------------------------------
# KB Registry
# ---------------------------------------------------------------------------

class KBRegistry:
    def __init__(self, registry_path: Path) -> None:
        self.registry_path = registry_path
        self._entries: dict[str, dict[str, str]] = {}
        self._instances: dict[str, AgentMemory] = {}
        self._load()

    def _load(self) -> None:
        if not self.registry_path.exists():
            return
        try:
            data = json.loads(self.registry_path.read_text(encoding="utf-8"))
            for entry in data:
                self._entries[entry["id"]] = {
                    "description": entry["description"],
                    "path": entry["path"],
                }
        except Exception as exc:
            print(f"[KBRegistry] Warning: could not load registry: {exc}", file=sys.stderr)

    def _save(self) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = [
            {"id": kb_id, "description": v["description"], "path": v["path"]}
            for kb_id, v in self._entries.items()
        ]
        self.registry_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def list(self) -> list[dict[str, str]]:
        return [
            {"id": kb_id, "description": v["description"], "path": v["path"]}
            for kb_id, v in self._entries.items()
        ]

    def exists(self, kb_id: str) -> bool:
        return kb_id in self._entries

    def register(self, kb_id: str, description: str, path: str) -> None:
        self._entries[kb_id] = {"description": description, "path": path}
        self._instances.pop(kb_id, None)
        self._save()

    def unregister(self, kb_id: str) -> bool:
        if kb_id not in self._entries:
            return False
        del self._entries[kb_id]
        self._instances.pop(kb_id, None)
        self._save()
        return True

    def get(self, kb_id: str) -> AgentMemory:
        if kb_id not in self._entries:
            available = ", ".join(self._entries.keys()) or "none"
            raise KeyError(
                f"Knowledge base {kb_id!r} not found. "
                f"Available IDs: {available}. "
                f"Use kb_list to see all registered bases."
            )
        if kb_id not in self._instances:
            self._instances[kb_id] = AgentMemory(Path(self._entries[kb_id]["path"]))
        return self._instances[kb_id]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_HOST       = os.environ.get("MCP_HOST", "0.0.0.0")
_PORT       = int(os.environ.get("MCP_PORT", "8000"))
_TRANSPORT  = os.environ.get("MCP_TRANSPORT", "sse")

_DATA_DIR   = Path(os.environ.get("MCP_DATA_DIR", str(Path.home() / ".graphed_kb")))
_TOKEN_MAP_PATH = Path(os.environ.get("MCP_TOKEN_MAP", str(_DATA_DIR / "tokens.json")))

_KB_PATH        = os.environ.get("MCP_KB_PATH")
_KB_DESCRIPTION = os.environ.get("MCP_KB_DESCRIPTION", "Default knowledge base")


# ---------------------------------------------------------------------------
# Token -> user resolution
# ---------------------------------------------------------------------------

def _load_token_map() -> dict[str, str]:
    """Load {token: user_id} from the token map file."""
    if not _TOKEN_MAP_PATH.exists():
        return {}
    try:
        return json.loads(_TOKEN_MAP_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[auth] Warning: could not load token map: {exc}", file=sys.stderr)
        return {}

# Loaded once at startup; server restart required to pick up changes.
_TOKEN_MAP: dict[str, str] = _load_token_map()


def _resolve_user(token: str) -> str | None:
    """Return user_id for a token, or None if unknown."""
    return _TOKEN_MAP.get(token)


# ---------------------------------------------------------------------------
# Per-user registry cache
# ---------------------------------------------------------------------------

_registry_cache: dict[str, KBRegistry] = {}


def _user_dir(user_id: str) -> Path:
    return _DATA_DIR / user_id


def _get_registry(user_id: str) -> KBRegistry:
    if user_id not in _registry_cache:
        user_dir = _user_dir(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)
        reg = KBRegistry(user_dir / "registry.json")
        # Seed default KB for new users if MCP_KB_PATH is set
        if _KB_PATH and not reg.exists("default"):
            reg.register("default", _KB_DESCRIPTION, str(Path(_KB_PATH).resolve()))
        _registry_cache[user_id] = reg
    return _registry_cache[user_id]


def _kb_store(user_id: str) -> Path:
    return _user_dir(user_id) / "kbs"


# ---------------------------------------------------------------------------
# Request-scoped user context
# ---------------------------------------------------------------------------

_current_user: contextvars.ContextVar[str] = contextvars.ContextVar("current_user")


def _get_user_registry() -> tuple[KBRegistry | None, str | None]:
    user_id = _current_user.get(None)
    if not user_id:
        return None, "No authenticated user in context."
    return _get_registry(user_id), None


def _get_kb(kb_id: str) -> tuple[AgentMemory | None, str | None]:
    reg, err = _get_user_registry()
    if err:
        return None, err
    try:
        return reg.get(kb_id), None
    except KeyError as exc:
        return None, str(exc)


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("graphed-knowledge", host=_HOST, port=_PORT)


# ---------------------------------------------------------------------------
# Registry tools
# ---------------------------------------------------------------------------

@mcp.tool()
def kb_list() -> str:
    """List all registered knowledge bases with their IDs and descriptions.

    Always call this first to discover which knowledge bases are available
    before querying, writing, or inspecting any of them.
    """
    reg, err = _get_user_registry()
    if err:
        return f"Error: {err}"
    entries = reg.list()
    if not entries:
        return "No knowledge bases registered yet.\nUse kb_create to add one."
    lines = ["Registered knowledge bases:\n"]
    for e in entries:
        lines.append(f"  [{e['id']}]  {e['description']}")
        lines.append(f"           path: {e['path']}")
        lines.append("")
    return "\n".join(lines).rstrip()


@mcp.tool()
def kb_create(kb_id: str, description: str) -> str:
    """Register and initialise a new knowledge base.

    Storage is managed automatically by the server.

    Args:
        kb_id:       Unique identifier (e.g. 'ml-research'). Lowercase, digits, hyphens.
        description: Human-readable description shown in kb_list.
    """
    reg, err = _get_user_registry()
    if err:
        return f"Error: {err}"
    if reg.exists(kb_id):
        return (
            f"Error: knowledge base {kb_id!r} already exists. "
            f"Use kb_delete to remove it first, or choose a different ID."
        )
    user_id = _current_user.get()
    resolved = str((_kb_store(user_id) / kb_id).resolve())
    reg.register(kb_id, description, resolved)
    try:
        mem = reg.get(kb_id)
    except Exception as exc:
        reg.unregister(kb_id)
        return f"Error initialising KB at {resolved!r}: {exc}"
    s = mem.stats()
    return (
        f"Created knowledge base '{kb_id}'.\n"
        f"Description: {description}\n"
        f"Nodes loaded: {s['nodes']}"
    )


@mcp.tool()
def kb_delete(kb_id: str) -> str:
    """Unregister a knowledge base. Files on disk are NOT deleted.

    Args:
        kb_id: ID of the knowledge base to unregister.
    """
    reg, err = _get_user_registry()
    if err:
        return f"Error: {err}"
    if not reg.unregister(kb_id):
        available = ", ".join(e["id"] for e in reg.list()) or "none"
        return f"Error: knowledge base {kb_id!r} not found. Available IDs: {available}."
    return f"Knowledge base '{kb_id}' has been unregistered. Files on disk are preserved."


# ---------------------------------------------------------------------------
# Per-KB tools
# ---------------------------------------------------------------------------

@mcp.tool()
def kb_recall(kb_id: str, query: str, strategy: str = "hybrid", top_k: int = 5) -> str:
    """Retrieve relevant knowledge from a specific KB.

    Args:
        kb_id:     ID of the knowledge base (use kb_list to find IDs).
        query:     Natural-language query.
        strategy:  'keyword', 'graph', or 'hybrid' (default).
        top_k:     Maximum results to return (default 5).
    """
    mem, err = _get_kb(kb_id)
    if err:
        return f"Error: {err}"
    return mem.recall(query, strategy=strategy, top_k=top_k, as_context=True)


@mcp.tool()
def kb_remember(
    kb_id: str,
    title: str,
    content: str,
    tags: list[str] | None = None,
    links_to: list[str] | None = None,
) -> str:
    """Persist a new note into a specific knowledge base.

    Args:
        kb_id:     ID of the knowledge base (use kb_list to find IDs).
        title:     Human-readable title for the note.
        content:   Markdown body text.
        tags:      Optional list of tag strings (without leading #).
        links_to:  Optional list of existing node IDs to link to.
    """
    mem, err = _get_kb(kb_id)
    if err:
        return f"Error: {err}"
    node = mem.remember(title, content, tags=tags or [], links_to=links_to or [])
    return f"Stored '{node.id}' in knowledge base '{kb_id}' with {len(node.links)} outgoing link(s)."


@mcp.tool()
def kb_associate(kb_id: str, from_node: str, to_node: str) -> str:
    """Add an explicit link from one KB node to another.

    Args:
        kb_id:      ID of the knowledge base.
        from_node:  ID of the source node.
        to_node:    ID of the destination node.
    """
    mem, err = _get_kb(kb_id)
    if err:
        return f"Error: {err}"
    try:
        added = mem.associate(from_node, to_node)
        return f"Link from '{from_node}' -> '{to_node}' in '{kb_id}': {'added' if added else 'already existed'}."
    except KeyError as exc:
        return f"Error: {exc}"


@mcp.tool()
def kb_reflect(kb_id: str, top_k: int = 5) -> str:
    """Return the top hub nodes by PageRank.

    Args:
        kb_id:  ID of the knowledge base.
        top_k:  Number of top nodes to return (default 5).
    """
    mem, err = _get_kb(kb_id)
    if err:
        return f"Error: {err}"
    results = mem.reflect(top_k=top_k)
    if not results:
        return f"Knowledge base '{kb_id}' is empty."
    lines = [f"Top {len(results)} hub nodes in '{kb_id}':\n"]
    for title, score in results:
        lines.append(f"  {title}: {score:.4f}")
    return "\n".join(lines)


@mcp.tool()
def kb_search_all(query: str, strategy: str = "hybrid", top_k: int = 3) -> str:
    """Search across ALL registered knowledge bases and return merged results.

    Args:
        query:     Natural-language query.
        strategy:  'keyword', 'graph', or 'hybrid' (default).
        top_k:     Maximum results per KB (default 3).
    """
    reg, err = _get_user_registry()
    if err:
        return f"Error: {err}"
    entries = reg.list()
    if not entries:
        return "No knowledge bases registered."
    lines = []
    for e in entries:
        try:
            mem = reg.get(e["id"])
            result = mem.recall(query, strategy=strategy, top_k=top_k, as_context=True)
            if "[No relevant knowledge found]" not in result:
                lines.append(f"--- KB: {e['id']} ---\n{result}")
        except Exception:
            continue
    return "\n\n".join(lines) if lines else "[No relevant knowledge found across any KB]"


@mcp.tool()
def kb_stats(kb_id: str) -> str:
    """Return statistics about a specific knowledge graph.

    Args:
        kb_id: ID of the knowledge base.
    """
    mem, err = _get_kb(kb_id)
    if err:
        return f"Error: {err}"
    s = mem.stats()
    return (
        f"Knowledge base: {kb_id}\n"
        f"Nodes:          {s['nodes']}\n"
        f"Edges:          {s['edges']}\n"
        f"Isolated nodes: {s['isolated_nodes']}\n"
        f"Avg degree:     {s['avg_degree']:.2f}\n"
        f"Path:           {s['kb_path']}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if _TRANSPORT == "stdio":
        print("[graphed-knowledge] Starting MCP server (stdio transport)", file=sys.stderr)
        mcp.run(transport="stdio")
    else:
        import json as _json
        import uvicorn
        from starlette.types import Receive, Scope, Send

        app = mcp.streamable_http_app()
        _inner_app = app

        async def _read_body(receive: Receive) -> bytes:
            body = b""
            while True:
                msg = await receive()
                body += msg.get("body", b"")
                if not msg.get("more_body", False):
                    break
            return body

        async def _json_response(send: Send, data, status: int = 200) -> None:
            body = _json.dumps(data).encode()
            await send({"type": "http.response.start", "status": status, "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ]})
            await send({"type": "http.response.body", "body": body})

        async def _handle_api(path: str, method: str, receive: Receive, send: Send) -> None:
            parts = path.removeprefix("/api/").split("/")

            # GET /api/kbs
            if parts == ["kbs"] and method == "GET":
                reg, err = _get_user_registry()
                if err:
                    return await _json_response(send, {"error": err}, 500)
                return await _json_response(send, reg.list())

            # /api/kbs/{kb_id}/...
            if len(parts) >= 2 and parts[0] == "kbs":
                kb_id = parts[1]

                # GET /api/kbs/{kb_id}/nodes
                if len(parts) == 3 and parts[2] == "nodes" and method == "GET":
                    mem, err = _get_kb(kb_id)
                    if err:
                        return await _json_response(send, {"error": err}, 404)
                    nodes = []
                    for nid, node in mem.graph.nodes.items():
                        nodes.append({
                            "id": nid,
                            "title": node.title,
                            "tags": node.tags,
                            "links": node.links,
                            "snippet": node.content[:150],
                        })
                    stats = mem.stats()
                    return await _json_response(send, {"nodes": nodes, "stats": stats})

                # GET /api/kbs/{kb_id}/nodes/{node_id}
                if len(parts) == 4 and parts[2] == "nodes" and method == "GET":
                    node_id = parts[3]
                    mem, err = _get_kb(kb_id)
                    if err:
                        return await _json_response(send, {"error": err}, 404)
                    node = mem.graph.nodes.get(node_id)
                    if not node:
                        return await _json_response(send, {"error": "not found"}, 404)
                    return await _json_response(send, {
                        "id": node.id,
                        "title": node.title,
                        "content": node.content,
                        "tags": node.tags,
                        "links": node.links,
                    })

                # POST /api/kbs/{kb_id}/query
                if len(parts) == 3 and parts[2] == "query" and method == "POST":
                    mem, err = _get_kb(kb_id)
                    if err:
                        return await _json_response(send, {"error": err}, 404)
                    body = _json.loads(await _read_body(receive))
                    query = body.get("query", "")
                    strategy = body.get("strategy", "hybrid")
                    top_k = body.get("top_k", 10)
                    result = mem.recall(query, strategy=strategy, top_k=top_k, as_context=True)
                    return await _json_response(send, {"result": result})

            await _json_response(send, {"error": "not found"}, 404)

        async def _auth_app(scope: Scope, receive: Receive, send: Send) -> None:
            if scope["type"] in ("http", "websocket"):
                path = scope.get("path", "")
                if path.startswith("/.well-known/") or path == "/token" or path == "/register":
                    await send({"type": "http.response.start", "status": 404, "headers": [(b"content-length", b"9")]})
                    await send({"type": "http.response.body", "body": b"Not Found"})
                    return
                headers = dict(scope.get("headers", []))
                auth = headers.get(b"authorization", b"").decode()
                token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
                user_id = _resolve_user(token)
                if not user_id:
                    await send({"type": "http.response.start", "status": 401, "headers": [(b"www-authenticate", b"Bearer"), (b"content-length", b"12")]})
                    await send({"type": "http.response.body", "body": b"Unauthorized"})
                    return
                _current_user.set(user_id)
                if path.startswith("/api/"):
                    method = scope.get("method", "GET")
                    return await _handle_api(path, method, receive, send)
            await _inner_app(scope, receive, send)

        if _TOKEN_MAP:
            print(f"[graphed-knowledge] Token auth enabled ({len(_TOKEN_MAP)} user(s))", file=sys.stderr)
        else:
            print("[graphed-knowledge] Warning: token map is empty — all requests will be rejected", file=sys.stderr)

        print(f"[graphed-knowledge] Starting MCP server (SSE on {_HOST}:{_PORT})", file=sys.stderr)
        uvicorn.run(_auth_app, host=_HOST, port=_PORT)
