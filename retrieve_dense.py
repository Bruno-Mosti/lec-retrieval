# PURPOSE: Given a text query, find the top-5 most relevant documents

import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# load corpus
with open("corpus.json") as f:
    corpus = json.load(f)

filenames = [d["filename"] for d in corpus]
texts = [d["combined_text"] for d in corpus]

# load model

# the purpose: load the embedding model into memory.
# first run: downloads ~90MB to a local cache. subsequent runs: instant.

print("____Loading the embedding model could take a few seconds")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("model loaded.")

# embed the corpus (with caching)

CACHE_PATH = Path("embeddings.npy")

if CACHE_PATH.exists():
    print("Loading cached embeddings from disk...")
    corpus_embeddings = np.load(CACHE_PATH)
else:
    # compute embeddings for all documents.
    # model.encode() returns shape (N, 384) one row per document.

    print("Embedding corpus (takes 1 to 3 minutes, only happens once)...")
    corpus_embeddings = model.encode(texts, show_progress_bar=True)
    np.save(CACHE_PATH, corpus_embeddings)
    print(f"Embeddings saved to {CACHE_PATH}. Will load from cache next time.")

print(f"Corpus embeddings shape: {corpus_embeddings.shape}")

# Expected: (208, 384) — one 384-number row per document (N = len(corpus))

#nromalise corpus embeddings

#before the program cosine similarity, normalise each row to unit length (length 1.0)

#why normalise?
 #osine similarity = (A · B) / (||A|| × ||B||)
 # If both vectors have length 1, this simplifies to: similarity = A · B (dot product)


 #compute the L2 norm of each row

norms = np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
# axis=1: norm per document row, not the whole matrix.

#divide each row by its norm.
corpus_embeddings_norm = corpus_embeddings / norms

#retrival function

def retrieve(query: str, top_k: int = 5):
    """
    PURPOSE: Given a query, embed it and find the most similar documents.
    Returns: (list_of_filenames, latency_in_ms)
    """
    t0 = time.perf_counter()

    query_vec = model.encode(query)
    #result shape (384,) this is a single vector

    #now i need to normalise the query vector
    query_vec_norm = query_vec / np.linalg.norm(query_vec)

    similarities= corpus_embeddings_norm @ query_vec_norm
        # Result shape: (N,) — one score per document. Higher = more similar.

    top_indices = np.argsort(similarities)[::-1][:top_k]
    results = [filenames[i] for i in top_indices]

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000

    return results, latency_ms


#smoke test

if __name__ == "__main__":
    #Test with the SAME query you used in retrieve_bm25.py.
    # Compare top-5 to BM25. Note differences — start of your failure analysis.
    test_query = "Fail2Ban SSH hardening block repeated logins"
    results, latency = retrieve(test_query)
    print(f"Query:   {test_query!r}")
    print(f"Top-5:   {results}")
    print(f"Latency: {latency:.2f} ms")
