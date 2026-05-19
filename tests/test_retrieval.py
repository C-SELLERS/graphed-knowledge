import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from graphed_kb import KnowledgeGraph, Retriever

KB_PATH = Path(__file__).parent.parent / "knowledge_base"


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        g = KnowledgeGraph().load(KB_PATH)
        cls.r = Retriever(g)

    def test_keyword_finds_attention(self):
        results = self.r.keyword_search("attention mechanism query key", top_k=3)
        self.assertGreater(len(results), 0)
        self.assertIn("attention_mechanism", [r.node_id for r in results])

    def test_keyword_top_result_relevant(self):
        results = self.r.keyword_search("transformer attention", top_k=1)
        self.assertIn(results[0].node_id, {"transformers", "attention_mechanism"})

    def test_graph_search_expands_context(self):
        results = self.r.graph_search("vector database embeddings", top_k=5, max_hops=2)
        ids = {r.node_id for r in results}
        self.assertTrue(ids & {"memory_systems", "knowledge_graphs", "retrieval_augmented_generation"})

    def test_graph_surfaces_hub_nodes(self):
        results = self.r.graph_search("agent memory storage", top_k=5)
        ids = {r.node_id for r in results}
        self.assertTrue("memory_systems" in ids or "agent_architectures" in ids)

    def test_hybrid_recall_gte_keyword(self):
        relevant = {"memory_systems", "knowledge_graphs", "retrieval_augmented_generation", "vector_databases"}
        kw = {r.node_id for r in self.r.keyword_search("long term memory ai", top_k=5)}
        hy = {r.node_id for r in self.r.hybrid_search("long term memory ai", top_k=5)}
        self.assertGreaterEqual(len(hy & relevant), len(kw & relevant))

    def test_scores_normalized(self):
        results = self.r.hybrid_search("neural network training", top_k=5)
        for r in results:
            self.assertLessEqual(r.score, 1.0 + 1e-6)
            self.assertGreaterEqual(r.score, 0.0)

    def test_get_context_returns_string(self):
        ctx = self.r.get_context("knowledge graph memory", strategy="hybrid", top_k=3)
        self.assertIsInstance(ctx, str)
        self.assertIn("Retrieved Knowledge", ctx)

    def test_empty_query_returns_empty(self):
        results = self.r.keyword_search("", top_k=5)
        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
