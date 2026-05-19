"""
Benchmark: flat keyword retrieval vs graph-augmented retrieval.

Each test case defines:
  - query:          natural language question
  - relevant_nodes: ground-truth node ids that should be in the results

Metrics computed:
  - Precision@k:  fraction of top-k results that are relevant
  - Recall@k:     fraction of relevant nodes found in top-k
  - MRR:          mean reciprocal rank of first relevant result
  - Hit@1:        1 if the top result is relevant

Run: python -m tests.benchmark  (from the project root)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from graphed_kb import KnowledgeGraph, Retriever

KB_PATH = Path(__file__).parent.parent / "knowledge_base"
TOP_K = 5

# -------------------------------------------------------------------
# Test cases: (query, ground_truth_node_ids)
# Ground truth was determined by human inspection of the KB.
# -------------------------------------------------------------------
TEST_CASES = [
    {
        "query": "how do transformers work",
        "relevant": {"transformers", "attention_mechanism", "neural_networks"},
    },
    {
        "query": "agent memory and knowledge storage",
        "relevant": {"memory_systems", "agent_architectures", "knowledge_graphs", "vector_databases"},
    },
    {
        "query": "retrieval augmented generation RAG pipeline",
        "relevant": {"retrieval_augmented_generation", "vector_databases", "memory_systems"},
    },
    {
        "query": "long term memory for AI",
        "relevant": {"memory_systems", "knowledge_graphs", "vector_databases", "retrieval_augmented_generation"},
    },
    {
        "query": "graph traversal knowledge base",
        "relevant": {"knowledge_graphs", "memory_systems", "retrieval_augmented_generation"},
    },
    {
        "query": "context window limitation solution",
        "relevant": {"transformers", "retrieval_augmented_generation", "memory_systems"},
    },
    {
        "query": "attention query key value",
        "relevant": {"attention_mechanism", "transformers"},
    },
    {
        "query": "semantic similarity embeddings nearest neighbor",
        "relevant": {"vector_databases", "retrieval_augmented_generation"},
    },
]


# -------------------------------------------------------------------
# Metric helpers
# -------------------------------------------------------------------

def precision_at_k(results: list, relevant: set, k: int) -> float:
    top = [r.node_id for r in results[:k]]
    hits = sum(1 for nid in top if nid in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(results: list, relevant: set, k: int) -> float:
    top = [r.node_id for r in results[:k]]
    hits = sum(1 for nid in top if nid in relevant)
    return hits / len(relevant) if relevant else 0.0


def mrr(results: list, relevant: set) -> float:
    for i, r in enumerate(results, 1):
        if r.node_id in relevant:
            return 1.0 / i
    return 0.0


def hit_at_1(results: list, relevant: set) -> float:
    return 1.0 if results and results[0].node_id in relevant else 0.0


# -------------------------------------------------------------------
# Runner
# -------------------------------------------------------------------

def run_benchmark(verbose: bool = True) -> dict:
    graph = KnowledgeGraph().load(KB_PATH)
    retriever = Retriever(graph)

    strategies = {
        "keyword": lambda q: retriever.keyword_search(q, top_k=TOP_K),
        "graph":   lambda q: retriever.graph_search(q, top_k=TOP_K, max_hops=2),
        "hybrid":  lambda q: retriever.hybrid_search(q, top_k=TOP_K, max_hops=2),
    }

    agg: dict[str, dict[str, list[float]]] = {
        s: {"p@k": [], "r@k": [], "mrr": [], "hit@1": []}
        for s in strategies
    }

    if verbose:
        print(f"Knowledge graph: {graph}")
        print(f"Running {len(TEST_CASES)} test cases, top_k={TOP_K}\n")
        print("=" * 80)

    for tc in TEST_CASES:
        query = tc["query"]
        relevant = tc["relevant"]

        if verbose:
            print(f"\nQuery: {query!r}")
            print(f"Ground truth: {sorted(relevant)}")

        for name, search_fn in strategies.items():
            results = search_fn(query)
            p = precision_at_k(results, relevant, TOP_K)
            r = recall_at_k(results, relevant, TOP_K)
            m = mrr(results, relevant)
            h = hit_at_1(results, relevant)

            agg[name]["p@k"].append(p)
            agg[name]["r@k"].append(r)
            agg[name]["mrr"].append(m)
            agg[name]["hit@1"].append(h)

            if verbose:
                top_ids = [res.node_id for res in results]
                hit_markers = ["[HIT]" if nid in relevant else "     " for nid in top_ids]
                result_str = "  ".join(f"{m} {nid}" for m, nid in zip(hit_markers, top_ids))
                print(f"  {name:8s} P@{TOP_K}={p:.2f}  R@{TOP_K}={r:.2f}  MRR={m:.2f}  H@1={h:.0f}  | {result_str}")

    # Aggregate
    summary: dict[str, dict[str, float]] = {}
    for name, metrics in agg.items():
        summary[name] = {k: sum(v) / len(v) for k, v in metrics.items()}

    if verbose:
        print("\n" + "=" * 80)
        print(f"\nAGGREGATE RESULTS (mean over {len(TEST_CASES)} queries, top_k={TOP_K})")
        print(f"{'Strategy':<10} {'P@k':>6} {'R@k':>6} {'MRR':>6} {'Hit@1':>6}")
        print("-" * 36)
        for name, metrics in summary.items():
            print(f"{name:<10} {metrics['p@k']:>6.3f} {metrics['r@k']:>6.3f} {metrics['mrr']:>6.3f} {metrics['hit@1']:>6.3f}")

        # Improvement over baseline
        base_p = summary["keyword"]["p@k"]
        base_r = summary["keyword"]["r@k"]
        print("\nImprovement over keyword baseline:")
        for name in ("graph", "hybrid"):
            dp = (summary[name]["p@k"] - base_p) / base_p * 100 if base_p else 0
            dr = (summary[name]["r@k"] - base_r) / base_r * 100 if base_r else 0
            print(f"  {name}: P@k {dp:+.1f}%  R@k {dr:+.1f}%")

    return summary


if __name__ == "__main__":
    run_benchmark(verbose=True)
