"""
Command-line interface for GraphedKnowledge.

Usage:
  python -m graphed_kb init [project_path] [--kb-dir .kb]
"""

import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# KB system-prompt section injected into the project's CLAUDE.md
# ---------------------------------------------------------------------------

_KB_SECTION_MARKER = "## Knowledge Base (GraphedKnowledge)"

_KB_CLAUDE_MD_SECTION = """\
## Knowledge Base (GraphedKnowledge)

This project has a GraphedKnowledge KB at `.kb/`. Use the MCP tools below to
consult and expand it during every session.

### When to use each tool

| Situation | Tool |
|-----------|------|
| Before answering any non-trivial question | `kb_recall` (strategy `hybrid`) |
| Learned something new or made a decision | `kb_remember` |
| Two existing nodes are related but unlinked | `kb_associate` |
| Start of session or domain shift | `kb_reflect` |
| Curious about KB health/size | `kb_stats` |

### Rules

- **Always `kb_recall` before answering** questions about this codebase.
- **`kb_remember`** after architectural decisions, gotchas, or useful insights.
  Good notes have a specific title, detailed Markdown body, relevant tags, and
  3-5 `links_to` existing node IDs.
- **Don't duplicate.** If a node already covers the topic, call `kb_associate`
  to add links instead of creating a redundant node.
- Use `strategy="hybrid"` by default — it combines keyword and graph search.
"""


# ---------------------------------------------------------------------------
# README / CLAUDE.md seeding helpers
# ---------------------------------------------------------------------------

def _split_by_h2(text: str) -> list[tuple[str, str]]:
    """Split a Markdown document into (heading, body) sections at H2 boundaries.
    Returns the preamble (before first H2) as the first section with the
    document title (first H1) or 'Overview' as the heading.
    """
    h1 = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    raw_title = h1.group(1).strip() if h1 else "Overview"
    doc_title = raw_title[:-3] if raw_title.lower().endswith(".md") else raw_title

    parts = re.split(r"^(##\s+.+)$", text, flags=re.MULTILINE)
    sections = []

    # preamble (before first ##)
    preamble = parts[0].strip()
    if preamble:
        sections.append((doc_title, preamble))

    # paired (## heading, body) chunks
    for i in range(1, len(parts) - 1, 2):
        heading = parts[i].lstrip("#").strip()
        body = parts[i + 1].strip()
        if body:
            sections.append((heading, body))

    return sections


def _seed_from_file(memory, source_path: Path, tag: str) -> list[str]:
    """Parse a Markdown file into sections and store each as a KB node.
    Returns the list of node IDs created.
    """
    from .parser import _slugify

    text = source_path.read_text(encoding="utf-8")
    sections = _split_by_h2(text)
    created = []

    for title, body in sections:
        # Skip very short sections (likely just a subheading with no content)
        if len(body) < 40:
            continue
        # Avoid duplicate IDs if the same title appears twice
        node = memory.remember(
            title=title,
            content=body,
            tags=[tag, "seeded"],
        )
        created.append(node.id)

    return created


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------

def cmd_init(project_path: Path, kb_dir: str = ".kb") -> None:
    from .agent_memory import AgentMemory

    project_path = project_path.resolve()
    kb_path = project_path / kb_dir
    kb_path.mkdir(parents=True, exist_ok=True)

    memory = AgentMemory(kb_path)
    seeded: list[str] = []

    # Seed from README.md
    for readme_name in ("README.md", "readme.md", "README.rst"):
        readme = project_path / readme_name
        if readme.exists():
            ids = _seed_from_file(memory, readme, tag="readme")
            seeded += [f"{readme_name} → '{nid}'" for nid in ids]
            break

    # Seed from CLAUDE.md
    claude_md = project_path / "CLAUDE.md"
    if claude_md.exists():
        ids = _seed_from_file(memory, claude_md, tag="claude-md")
        seeded += [f"CLAUDE.md → '{nid}'" for nid in ids]

    # Write .claude/settings.local.json with MCP server config
    # Using settings.local.json because paths are machine-specific
    claude_dir = project_path / ".claude"
    claude_dir.mkdir(exist_ok=True)
    settings_path = claude_dir / "settings.local.json"

    settings: dict = {}
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
        except json.JSONDecodeError:
            pass

    settings.setdefault("mcpServers", {})
    settings["mcpServers"]["project-kb"] = {
        "command": sys.executable,
        "args": [
            str(Path(__file__).parent.parent / "mcp_server.py"),
            "--kb-path", str(kb_path),
        ],
    }
    settings_path.write_text(json.dumps(settings, indent=2))

    # Inject KB instructions into project CLAUDE.md
    claude_md_path = project_path / "CLAUDE.md"
    if claude_md_path.exists():
        existing = claude_md_path.read_text(encoding="utf-8")
    else:
        existing = "# CLAUDE.md\n\nThis file provides guidance to Claude Code when working in this project.\n"

    if _KB_SECTION_MARKER not in existing:
        separator = "\n" if existing.endswith("\n") else "\n\n"
        claude_md_path.write_text(existing + separator + _KB_CLAUDE_MD_SECTION,
                                  encoding="utf-8")
        claude_md_updated = True
    else:
        claude_md_updated = False

    # Summary
    stats = memory.stats()
    print(f"\nKB initialized at {kb_path}")
    print(f"  {stats['nodes']} nodes, {stats['edges']} edges")
    if seeded:
        print("\nSeeded from:")
        for item in seeded:
            print(f"  {item}")
    else:
        print("\nNo README.md or CLAUDE.md found — KB is empty, ready for the agent to fill.")

    print(f"\nMCP config written to {settings_path}")
    if claude_md_updated:
        print(f"KB instructions appended to {claude_md_path}")
    else:
        print(f"CLAUDE.md already contains KB instructions — skipped")
    print("""
Next steps:
  1. Open a Claude Code session in this project
  2. The agent will have these tools: kb_recall, kb_remember,
     kb_associate, kb_reflect, kb_stats
  3. The KB usage instructions have been added to CLAUDE.md automatically
""")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m graphed_kb",
        description="GraphedKnowledge CLI",
    )
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="Initialise a KB for a project")
    p_init.add_argument(
        "project_path",
        nargs="?",
        default=".",
        help="Path to the project root (default: current directory)",
    )
    p_init.add_argument(
        "--kb-dir",
        default=".kb",
        help="Subdirectory name for the KB (default: .kb)",
    )

    args = parser.parse_args()

    if args.command == "init":
        cmd_init(Path(args.project_path), kb_dir=args.kb_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
