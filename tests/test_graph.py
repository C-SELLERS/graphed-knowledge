import sys
import unittest
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from graphed_kb.graph import KnowledgeGraph


def make_graph(tmp_path: Path) -> KnowledgeGraph:
    (tmp_path / "a.md").write_text("# A\n\nLinks to [[B]] and [[C]].\n")
    (tmp_path / "b.md").write_text("# B\n\nLinks to [[C]].\n")
    (tmp_path / "c.md").write_text("# C\n\nNo outgoing links.\n")
    (tmp_path / "d.md").write_text("# D\n\nIsolated node.\n")
    return KnowledgeGraph().load(tmp_path)


class TestGraphLoad(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.g = make_graph(self.tmp)

    def test_node_count(self):
        self.assertEqual(len(self.g.nodes), 4)

    def test_forward_edges(self):
        self.assertIn("b", self.g._edges["a"])
        self.assertIn("c", self.g._edges["a"])
        self.assertIn("c", self.g._edges["b"])
        self.assertEqual(len(self.g._edges["c"]), 0)

    def test_backlinks(self):
        self.assertIn("a", self.g._backlinks["b"])
        self.assertIn("a", self.g._backlinks["c"])
        self.assertIn("b", self.g._backlinks["c"])

    def test_neighbors_with_backlinks(self):
        n = self.g.neighbors("c", include_backlinks=True)
        self.assertIn("a", n)
        self.assertIn("b", n)

    def test_bfs_depth_1(self):
        dist = self.g.bfs(["a"], max_hops=1)
        self.assertEqual(dist["a"], 0)
        self.assertEqual(dist["b"], 1)
        self.assertEqual(dist["c"], 1)
        self.assertNotIn("d", dist)

    def test_bfs_chain_depth_2(self):
        (self.tmp / "e.md").write_text("# E\nLinks to [[F]].\n")
        (self.tmp / "f.md").write_text("# F\nLinks to [[G]].\n")
        (self.tmp / "g.md").write_text("# G\nEnd.\n")
        g = KnowledgeGraph().load(self.tmp)
        dist = g.bfs(["e"], max_hops=2, include_backlinks=False)
        self.assertEqual(dist["e"], 0)
        self.assertEqual(dist["f"], 1)
        self.assertEqual(dist["g"], 2)

    def test_pagerank_sums_to_one(self):
        pr = self.g.pagerank()
        self.assertAlmostEqual(sum(pr.values()), 1.0, places=4)

    def test_personalized_pagerank(self):
        pr = self.g.pagerank(personalization={"a": 1.0})
        self.assertGreater(pr["a"], pr["d"])

    def test_stats(self):
        s = self.g.stats()
        self.assertEqual(s["nodes"], 4)
        self.assertEqual(s["edges"], 3)
        self.assertEqual(s["isolated_nodes"], 1)


if __name__ == "__main__":
    unittest.main()
