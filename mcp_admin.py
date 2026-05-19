"""
Admin MCP server for GraphedKnowledge.

Exposes user and KB management tools. Protected by a separate admin token.
Runs on a different port from the user-facing server.

Environment variables:
    MCP_DATA_DIR        Base directory for all user data (shared with mcp_server.py)
    MCP_TOKEN_MAP       Path to token->user_id JSON map (shared with mcp_server.py)
    MCP_ADMIN_TOKEN     Required Bearer token for this server
    MCP_ADMIN_HOST      Bind host (default: 0.0.0.0)
    MCP_ADMIN_PORT      Bind port (default: 8001)

Admin tools:
  user_list         - list all users and their token count
  user_create       - add a new user and generate a token
  user_delete       - remove a user and all their tokens
  token_add         - add an extra token for an existing user
  token_revoke      - revoke a specific token
  kb_list_all       - list all KBs across all users
  kb_delete_user    - force-unregister a KB from any user's registry
"""

from __future__ import annotations

import json
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from mcp.server.fastmcp import FastMCP
from graphed_kb.agent_memory import AgentMemory


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ADMIN_HOST  = os.environ.get("MCP_ADMIN_HOST", "0.0.0.0")
_ADMIN_PORT  = int(os.environ.get("MCP_ADMIN_PORT", "8001"))
_ADMIN_TOKEN = os.environ.get("MCP_ADMIN_TOKEN")

_DATA_DIR        = Path(os.environ.get("MCP_DATA_DIR", str(Path.home() / ".graphed_kb")))
_TOKEN_MAP_PATH  = Path(os.environ.get("MCP_TOKEN_MAP", str(_DATA_DIR / "tokens.json")))


# ---------------------------------------------------------------------------
# Token map helpers (read/write the shared tokens.json)
# ---------------------------------------------------------------------------

def _read_token_map() -> dict[str, str]:
    if not _TOKEN_MAP_PATH.exists():
        return {}
    return json.loads(_TOKEN_MAP_PATH.read_text(encoding="utf-8"))


def _write_token_map(token_map: dict[str, str]) -> None:
    _TOKEN_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    _TOKEN_MAP_PATH.write_text(json.dumps(token_map, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Registry helpers (read a user's registry.json without caching)
# ---------------------------------------------------------------------------

def _read_user_registry(user_id: str) -> list[dict]:
    reg_path = _DATA_DIR / user_id / "registry.json"
    if not reg_path.exists():
        return []
    try:
        return json.loads(reg_path.read_text(encoding="utf-8"))
    except Exception:
        return []


def _write_user_registry(user_id: str, entries: list[dict]) -> None:
    reg_path = _DATA_DIR / user_id / "registry.json"
    reg_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# MCP server
# ---------------------------------------------------------------------------

mcp = FastMCP("graphed-knowledge-admin", host=_ADMIN_HOST, port=_ADMIN_PORT)


@mcp.tool()
def user_list() -> str:
    """List all users and how many tokens each has."""
    token_map = _read_token_map()
    if not token_map:
        return "No users registered."
    counts: dict[str, int] = {}
    for user_id in token_map.values():
        counts[user_id] = counts.get(user_id, 0) + 1
    lines = ["Users:\n"]
    for user_id, count in sorted(counts.items()):
        user_dir = _DATA_DIR / user_id
        kbs = _read_user_registry(user_id)
        lines.append(f"  {user_id}  ({count} token(s), {len(kbs)} KB(s))")
        lines.append(f"    data: {user_dir}")
    return "\n".join(lines)


@mcp.tool()
def user_create(user_id: str) -> str:
    """Create a new user and generate an initial token for them.

    Args:
        user_id: Unique identifier for the user (e.g. 'alice').
    """
    token_map = _read_token_map()
    if user_id in token_map.values():
        return f"Error: user '{user_id}' already exists. Use token_add to issue more tokens."
    token = secrets.token_urlsafe(32)
    token_map[token] = user_id
    _write_token_map(token_map)
    (_DATA_DIR / user_id).mkdir(parents=True, exist_ok=True)
    return f"Created user '{user_id}'.\nToken: {token}\n\nShare this token with the user — it cannot be recovered."


@mcp.tool()
def user_delete(user_id: str) -> str:
    """Remove a user and revoke all their tokens. KB files on disk are NOT deleted.

    Args:
        user_id: ID of the user to remove.
    """
    token_map = _read_token_map()
    tokens_to_remove = [t for t, u in token_map.items() if u == user_id]
    if not tokens_to_remove:
        return f"Error: user '{user_id}' not found."
    for t in tokens_to_remove:
        del token_map[t]
    _write_token_map(token_map)
    return f"Deleted user '{user_id}' and revoked {len(tokens_to_remove)} token(s). KB files preserved on disk."


@mcp.tool()
def token_add(user_id: str) -> str:
    """Generate and add an additional token for an existing user.

    Args:
        user_id: ID of the user to issue a new token for.
    """
    token_map = _read_token_map()
    if user_id not in token_map.values():
        return f"Error: user '{user_id}' not found. Use user_create first."
    token = secrets.token_urlsafe(32)
    token_map[token] = user_id
    _write_token_map(token_map)
    return f"New token for '{user_id}': {token}\n\nShare this token with the user — it cannot be recovered."


@mcp.tool()
def token_revoke(token: str) -> str:
    """Revoke a specific token.

    Args:
        token: The token string to revoke.
    """
    token_map = _read_token_map()
    if token not in token_map:
        return "Error: token not found."
    user_id = token_map.pop(token)
    _write_token_map(token_map)
    return f"Revoked token for user '{user_id}'."


@mcp.tool()
def kb_list_all() -> str:
    """List all knowledge bases across all users."""
    token_map = _read_token_map()
    all_users = sorted(set(token_map.values()))
    if not all_users:
        return "No users registered."
    lines = []
    for user_id in all_users:
        entries = _read_user_registry(user_id)
        lines.append(f"[{user_id}] ({len(entries)} KB(s))")
        for e in entries:
            lines.append(f"  [{e['id']}]  {e['description']}")
            lines.append(f"             path: {e['path']}")
    return "\n".join(lines) if lines else "No knowledge bases found."


@mcp.tool()
def kb_create_user(user_id: str, kb_id: str, description: str) -> str:
    """Create a knowledge base for a specific user.

    Args:
        user_id:     ID of the user to create the KB for.
        kb_id:       Unique KB identifier (lowercase, digits, hyphens).
        description: Human-readable description of the KB.
    """
    token_map = _read_token_map()
    if user_id not in token_map.values():
        return f"Error: user '{user_id}' not found."
    entries = _read_user_registry(user_id)
    if any(e["id"] == kb_id for e in entries):
        return f"Error: KB '{kb_id}' already exists for user '{user_id}'."
    kb_path = _DATA_DIR / user_id / "kbs" / kb_id
    kb_path.mkdir(parents=True, exist_ok=True)
    entries.append({"id": kb_id, "description": description, "path": str(kb_path)})
    _write_user_registry(user_id, entries)
    # Eagerly init to surface errors
    try:
        from graphed_kb.agent_memory import AgentMemory
        mem = AgentMemory(kb_path)
        s = mem.stats()
    except Exception as exc:
        entries = [e for e in entries if e["id"] != kb_id]
        _write_user_registry(user_id, entries)
        return f"Error initialising KB: {exc}"
    return f"Created KB '{kb_id}' for user '{user_id}'.\nDescription: {description}\nNodes loaded: {s['nodes']}"


@mcp.tool()
def kb_delete_user(user_id: str, kb_id: str) -> str:
    """Force-unregister a KB from a user's registry. Files on disk are NOT deleted.

    Args:
        user_id: ID of the user who owns the KB.
        kb_id:   ID of the knowledge base to unregister.
    """
    entries = _read_user_registry(user_id)
    original_len = len(entries)
    entries = [e for e in entries if e["id"] != kb_id]
    if len(entries) == original_len:
        return f"Error: KB '{kb_id}' not found for user '{user_id}'."
    _write_user_registry(user_id, entries)
    return f"Unregistered KB '{kb_id}' from user '{user_id}'. Files preserved on disk."


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json as _json
    import uvicorn
    from starlette.types import Receive, Scope, Send

    if not _ADMIN_TOKEN:
        print("[graphed-knowledge-admin] ERROR: MCP_ADMIN_TOKEN is not set — refusing to start.", file=sys.stderr)
        sys.exit(1)

    app = mcp.streamable_http_app()
    _inner_app = app

    async def _auth_app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] in ("http", "websocket"):
            path = scope.get("path", "")
            if path.startswith("/.well-known/") or path in ("/token", "/register"):
                await send({"type": "http.response.start", "status": 404, "headers": [(b"content-length", b"9")]})
                await send({"type": "http.response.body", "body": b"Not Found"})
                return
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"").decode()
            token = auth.removeprefix("Bearer ").strip() if auth.startswith("Bearer ") else ""
            if token != _ADMIN_TOKEN:
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-length", b"12"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b"Unauthorized"})
                return
        await _inner_app(scope, receive, send)

    print(f"[graphed-knowledge-admin] Starting admin MCP server (SSE on {_ADMIN_HOST}:{_ADMIN_PORT})", file=sys.stderr)
    uvicorn.run(_auth_app, host=_ADMIN_HOST, port=_ADMIN_PORT)
