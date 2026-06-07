# Combine BM25 and Dense results using Reciprocal Rank Fusion.

import time
from retrieve_bm25 import retrieve as bm25_retrieve
from retrieve_dense import retrieve as dense_retrieve

RRF_K = 60  # standard constant

def retrieve(query: str, top_k: int = 5):
    """
    PURPOSE: Get candidates from both BM25 and Dense, fuse them with RRF,
             return the top_k results from the combined ranking.
    """
    t0 = time.perf_counter()

    # get more candidates than top_k from each method.
    # after fusion, some documents that ranked low individually might rank high when combined
    bm25_results, _ = bm25_retrieve(query, top_k=20)
    dense_results, _ = dense_retrieve(query, top_k=20)

    # RRF fusion
    scores = {}

    for rank, filename in enumerate(bm25_results, start=1):
        contribution = 1 / (RRF_K + rank)
        scores[filename] = scores.get(filename, 0) + contribution

    for rank, filename in enumerate(dense_results, start=1):
        contribution = 1 / (RRF_K + rank)
        scores[filename] = scores.get(filename, 0) + contribution

    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = [filename for filename, score in sorted_docs[:top_k]]

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000
    return results, latency_ms


if __name__ == "__main__":
    test_query = "Fail2Ban SSH hardening block repeated logins"
    results, latency = retrieve(test_query)
    print(f"Top-5:   {results}")
    print(f"Latency: {latency:.2f} ms")
