# retrieve_bm25.py
# PURPOSE: Given a text query, find the top-5 most relevant documents

import json
import time
import re
import numpy as np
from rank_bm25 import BM25Okapi    #library

#load corpus
# #PURPOSE: Read the documents you exported from Neo4j.
# This runs once when the script loads, not on every search.

with open("corpus.json") as f:
    corpus = json.load(f)

# PURPOSE: Keep a separate list of just the filenames.
# When BM25 says "document 42 is most relevant", you need to know
# that document 42 is called "Phase9B_Neo4j_setup.md".

filenames = [d["filename"] for d in corpus]


#tokenisation
# It needs text split into a LIST of individual words (called "tokens").
# Example: "rate limiting Flask" → ["rate", "limiting", "flask"]
# You must apply the same tokenisation to both the corpus AND every query.
# If you tokenise differently, BM25 will not find matches.

# Option C: re.findall(r'\w+', text.lower())
# Finds all word-character sequences after lowercasing.
# Splits on hyphens and all non-word characters.
# Flask-Login → ["flask", "login"], RTX-2060 → ["rtx", "2060"]
# Cleaner than Option B (re.split) — no empty string filtering needed.
# Trade-off: improves recall (query "flask login" finds Flask-Login docs)
# but loses compound-term specificity.
# Chosen because PSB corpus has many hyphenated terms queried
# in plain natural language without hyphens.

def tokenise(text: str) -> list[str]:
    return re.findall(r'\w+', text.lower())

#search index
# PURPOSE: BM25 needs to "learn" the corpus before it can answer queries.
# This happens once when the script loads — not on every search.
print("Building BM25 index...")

#apply your tokenise() fucntion toe very dociument's combined_text
#it will result in a list of list where eac inner list is one tokenised document
tokenised_corpus = [tokenise(d["combined_text"]) for d in corpus]


#PURPOSE: This line does the actual indexing — reads all tokenised documents
#          and builds the internal frequency tables BM25 needs.
index = BM25Okapi(tokenised_corpus)


print(f"BM25 index built. {len(corpus)} documents indexed.")


#retrieval function
def retrieve(query:str, top_k: int=5):
    """
    PURPOSE: Given a query string, return the top_k most relevant filenames.
    Returns a tuple: (list_of_filenames, latency_in_milliseconds)
    """

#purpuse: Record the start time so we can measure how long the search takes.
    t0 = time.perf_counter()
    tokens = tokenise(query)
    scores = index.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:top_k]
    results = [filenames[i] for i in top_indices]

    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000
    return results, latency_ms

if __name__ == "__main__":
    test_query = "software engineering process models"
    results, latency = retrieve(test_query)
    print(f"Query:   {test_query}")
    print(f"Top-5:   {results}")
    print(f"Latency: {latency:.2f} ms")
