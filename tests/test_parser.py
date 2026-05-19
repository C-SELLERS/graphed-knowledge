import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from graphed_kb.parser import parse_node, load_directory, _slugify


def write_md(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


class TestSlugify(unittest.TestCase):
    def test_spaces(self):
        self.assertEqual(_slugify("Machine Learning"), "machine_learning")

    def test_strip(self):
        self.assertEqual(_slugify("  RAG  "), "rag")

    def test_preserves_hyphen(self):
        self.assertEqual(_slugify("Knowledge-Graph"), "knowledge-graph")


class TestParseFrontmatter(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_title_and_tags(self):
        p = write_md(self.tmp, "test.md", "---\ntitle: My Note\ntags: [a, b]\n---\n\nBody text.\n")
        node = parse_node(p)
        self.assertEqual(node.title, "My Note")
        self.assertIn("a", node.tags)
        self.assertIn("b", node.tags)

    def test_wikilinks(self):
        p = write_md(self.tmp, "test.md", "See [[Other Node]] and [[Alias|Display]] for more.\n")
        node = parse_node(p)
        self.assertIn("other_node", node.links)
        self.assertIn("alias", node.links)

    def test_inline_tags(self):
        p = write_md(self.tmp, "test.md", "This is #important and #ai related.\n")
        node = parse_node(p)
        self.assertIn("important", node.tags)
        self.assertIn("ai", node.tags)

    def test_title_fallback_h1(self):
        p = write_md(self.tmp, "some_file.md", "# My Heading\n\nContent here.\n")
        node = parse_node(p)
        self.assertEqual(node.title, "My Heading")

    def test_title_fallback_filename(self):
        p = write_md(self.tmp, "my_note.md", "Just content, no heading.\n")
        node = parse_node(p)
        self.assertEqual(node.title, "My Note")

    def test_load_directory(self):
        write_md(self.tmp, "alpha.md", "# Alpha\n\nLinks to [[Beta]].\n")
        write_md(self.tmp, "beta.md", "# Beta\n\nContent.\n")
        nodes = load_directory(self.tmp)
        self.assertIn("alpha", nodes)
        self.assertIn("beta", nodes)
        self.assertIn("beta", nodes["alpha"].links)

    def test_plain_text_strips_markdown(self):
        p = write_md(self.tmp, "test.md", "# Heading\n\n**bold** and _italic_.\n")
        node = parse_node(p)
        self.assertNotIn("**", node.plain_text)


if __name__ == "__main__":
    unittest.main()
