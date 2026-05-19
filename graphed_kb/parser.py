"""
Parse Obsidian-style Markdown files into KBNode objects.

Handles:
  - YAML frontmatter (between --- delimiters)
  - [[wiki-links]] and [[wiki-links|aliases]]
  - #tags (inline)
  - Plain text extraction
"""

import re
from pathlib import Path
from dataclasses import dataclass, field


_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)

# Minimal YAML-subset parser — handles the frontmatter we actually use:
#   scalar strings, inline lists [a, b, c], and integers/booleans.
_FM_LIST_RE = re.compile(r"^\[(.+)\]$")
_FM_KEY_RE = re.compile(r"^(\w[\w -]*):\s*(.*)")


def _parse_frontmatter(text: str) -> dict:
    result: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _FM_KEY_RE.match(line)
        if not m:
            continue
        key, value = m.group(1).strip(), m.group(2).strip()
        # inline list: [a, b, c]
        lm = _FM_LIST_RE.match(value)
        if lm:
            result[key] = [v.strip().strip("'\"") for v in lm.group(1).split(",")]
        elif value.lower() == "true":
            result[key] = True
        elif value.lower() == "false":
            result[key] = False
        elif value.isdigit():
            result[key] = int(value)
        else:
            result[key] = value.strip("'\"")
    return result
_WIKILINK_RE = re.compile(r"\[\[([^\]|#]+)(?:[|#][^\]]*)?\]\]")
_TAG_RE = re.compile(r"(?<!\w)#([a-zA-Z][a-zA-Z0-9_/-]*)")
_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)", re.MULTILINE)
_MARKDOWN_NOISE_RE = re.compile(r"[`*_~\[\]#>|]|!\[.*?\]\(.*?\)|\[.*?\]\(.*?\)")


@dataclass
class KBNode:
    id: str                          # filename stem, lowercased, spaces->underscores
    title: str                       # human-readable title (H1 or filename)
    content: str                     # raw markdown body (no frontmatter)
    plain_text: str                  # markdown stripped, for indexing
    links: list[str]                 # [[wiki-link]] targets as node ids
    tags: list[str]                  # #tag names (no leading #)
    metadata: dict                   # parsed frontmatter
    path: Path


def _slugify(name: str) -> str:
    """Convert a link target or filename to a stable node id."""
    return name.strip().lower().replace(" ", "_")


def _strip_markdown(text: str) -> str:
    text = _MARKDOWN_NOISE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_node(path: Path) -> KBNode:
    raw = path.read_text(encoding="utf-8")

    # --- frontmatter ---
    metadata: dict = {}
    body = raw
    m = _FRONTMATTER_RE.match(raw)
    if m:
        metadata = _parse_frontmatter(m.group(1))
        body = raw[m.end():]

    # --- title: frontmatter > first H1 > filename ---
    title = metadata.get("title", "")
    if not title:
        h1 = _HEADING_RE.search(body)
        title = h1.group(1) if h1 else path.stem.replace("_", " ").title()

    # --- links ---
    links = [_slugify(t) for t in _WIKILINK_RE.findall(body)]

    # --- tags: frontmatter list + inline #tags ---
    fm_tags = metadata.get("tags", [])
    if isinstance(fm_tags, str):
        fm_tags = [fm_tags]
    inline_tags = _TAG_RE.findall(body)
    tags = list(dict.fromkeys(fm_tags + inline_tags))  # deduplicate, preserve order

    node_id = _slugify(path.stem)
    plain_text = _strip_markdown(body)

    return KBNode(
        id=node_id,
        title=title,
        content=body,
        plain_text=plain_text,
        links=links,
        tags=tags,
        metadata=metadata,
        path=path,
    )


def load_directory(kb_path: str | Path) -> dict[str, KBNode]:
    """Recursively load all .md files under kb_path. Returns {node_id: KBNode}."""
    kb_path = Path(kb_path)
    nodes: dict[str, KBNode] = {}
    for md_file in sorted(kb_path.rglob("*.md")):
        node = parse_node(md_file)
        nodes[node.id] = node
    return nodes
