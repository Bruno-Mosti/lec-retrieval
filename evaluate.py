# the pourpose of this file is to run my 20 querires through any retrivial method
#and compute Recall@5, MRR, and p95 latency.
# you use this for all three methods (BM24, dense, hybrid).

import json
import numpy as np

# load querires

with open("queries.json") as f:
    queries = json.load(f)


#metric functions

def compute_recall_at_k(results: list, relevant: list, k: int = 5) -> float:
    """
    the whole purpose is to caluculate how many re;evamt doc appear in the top k result.
    results -> the list of filenames i search returned (in order)
    relevant -> the list of correct filenames for this query
    k -> how many results to check (default 5)
    returns a number betwenn 0.0 and 1.01
    """

    #take only the first k results
    top_k_results = results[:k]

    #it count how many items in 'relevant' also appear in top_k_results.
    hits = sum(1 for fn in relevant if fn in top_k_results)

    if len(relevant) == 0:
        return 0.0
    return hits / len(relevant)


def compute_rr(results: list, relevant: list) -> float:
    """Reciprocal rank for one query: 1/rank of first hit, else 0.0."""
    for i, filename in enumerate(results):
        if filename in relevant:
            return 1.0 / (i + 1)
    return 0.0


def compute_mrr(rr_scores: list) -> float:
    """
    average the reciprocal rank scores across all queries.
    rr_scores -> list of per-query RR values (eacg between 0 and 1)
    returns the mean Reciprocal rank.
    
    
    """
    #one line. return the average of rr_scores.
    return sum(rr_scores) / len(rr_scores)


def compute_p95(latencies_ms: list) -> float:
    """the whole purpose is to find the 95th percentile latency across all queries.
    latencies_ms -> list of query times in milliseconds
    returns the p95 value. Must be under 1000ms to pass the assigment.
    """
    #np. percentile and not max(): give the worst single case
    return float(np.percentile(latencies_ms, 95))


#evaluation loop

def evaluate(retrieve_fn, config_name: str) -> dict:
    recalls, rr_scores, latencies = [], [], []

    for q in queries:
        run_latencies = []
        for _ in range(3):
            results, lat = retrieve_fn(q["query"])
            run_latencies.append(lat)
        stable_latency = float(np.median(run_latencies))
        latencies.append(stable_latency)
        recalls.append(compute_recall_at_k(results, q["relevant_filenames"]))
        rr_scores.append(compute_rr(results, q["relevant_filenames"]))

    result = {
        "config":         config_name,
        "recall_at_5":    round(float(np.mean(recalls)), 4),
        "mrr":            round(compute_mrr(rr_scores), 4),
        "p95_latency_ms": round(compute_p95(latencies), 2),
    }
    print(f"\n{'='*50}")
    print(f"Config:      {config_name}")
    print(f"Recall@5:    {result['recall_at_5']}")
    print(f"MRR:         {result['mrr']}")
    print(f"p95 (ms):    {result['p95_latency_ms']}")
    print(f"{'='*50}")
    return result


if __name__ == "__main__":
    from retrieve_bm25 import retrieve as bm25_retrieve
    from retrieve_dense import retrieve as dense_retrieve
    from retrieve_hybrid import retrieve as hybrid_retrieve

    results_table = [evaluate(bm25_retrieve, "BM25")]
    results_table.append(evaluate(dense_retrieve, "Dense"))
    results_table.append(evaluate(hybrid_retrieve, "Hybrid_RRF"))

    import os
    os.makedirs("results", exist_ok=True)
    with open("results/table.json", "w") as f:
        json.dump(results_table, f, indent=2)
    print("\nResults saved to results/table.json")
    print("\nIf Recall@5 is 0.0: check that filenames in queries.json")
    print("match filenames in corpus.json exactly.")