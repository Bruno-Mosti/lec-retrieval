# Retrieval, Honest Comparison — Bruno Mosti
**LEC AI — Assignment 1 | GitHub upload: Sunday 7 June 2026, 12:30**

---

## 1. Why I picked Assignment 1

PSB already had a complete document ingestion pipeline. When I upload a file through the dashboard, Flask receives it and drops it into a Redis queue. An RQ worker — a separate subprocess — picks it up, extracts the full text using PyMuPDF for PDFs and python-docx for Word files, then sends it to Ollama (Qwen3:8b) for summarisation and concept extraction. The result lands in Neo4j as a :Document node. The worker pattern was deliberate — if Flask handled the AI processing inline, the entire UI would freeze for minutes on every upload. I built this in about three days after the interview question I couldn't fully answer.

What PSB could not do was retrieve documents intelligently at query time. The only search that existed was a Neo4j Cypher CONTAINS match on node metadata — name, topic, title — not the document bodies themselves. The AI assistant answered from its system prompt and its own training. No RAG. No BM25. No embeddings. Documents were ingested and stored, but never actually searched.

I chose this assignment because it was exactly the missing layer. I already had 208 real documents I had written and ingested myself — I know what they contain, I know which ones share vocabulary, I know which concepts only appear in one specific session log. That made it possible to design genuinely hard queries and produce a failure analysis grounded in real content, not a public dataset I had no relationship with.

Three days of building this opened a part of my PSB roadmap I had not reached yet — the intelligence layer. I now understand what RAG is and how to build it. The plan is to plug hybrid retrieval directly back into PSB so the assistant can search over real document content instead of answering from training data alone.

---

## 2. Corpus

The 208 documents come from three areas of my personal knowledge base: PSB build logs and session notes from every phase of development, university module materials (COMP1765, COMP1821, MATH1179 and others), and AI and software engineering reference documents I collected and ingested over time. Everything was personally written or curated — no public datasets, no scraped content.

**Why PSB and not Wikipedia:** I know these documents. I wrote most of them. That means I could design adversarial queries with real precision — I know which documents share vocabulary, which concepts span multiple files, and which terms only appear buried in one specific log. A public corpus would have required reading thousands of pages to get to the same level of query design accuracy.

**Honest corpus caveat:** Each document in `corpus.json` is not the full original file. PSB stores a `combined_text` field built from topic + Ollama-generated summary + first ~4000 characters of text — a deliberate hardware decision. My server runs an RTX 2060 with 6 GB VRAM. Processing full documents without this limit was already producing hallucinations and fragmented output. PSB stores a brain map, not a photocopy of every file. The honest consequence: if an important term only appears deep in a long PDF, it will not be in `combined_text`, and no config — BM25, dense, or hybrid — can find it. Retrieval cannot surface what was never fully indexed.

**Effect on metrics:** In practice this limitation helped rather than hurt — most PSB documents are short and keyword-dense, so important concepts are already concentrated in the excerpt. Overall scores are strong, but implicit queries still fail — see Section 7.

---

## 3. Experiment Design

All three configurations — BM25, Dense, and Hybrid_RRF — were evaluated on the same `corpus.json` (208 documents) and the same `queries.json` (20 queries). This is what makes the comparison fair and honest. If each config faced different queries the results would reflect the difficulty of the questions, not the quality of retrieval — the data would be unreliable and the analysis meaningless.

**Queries:** 20 hand-written labelled queries in `queries.json`, each mapping a natural-language question to one gold filename from the corpus. Each query retrieves top-k = 5 documents. 5 of the 20 are deliberately hard, designed to expose different failure modes:

| ID | Type | Gold document |
|---|---|---|
| Q03 | Paraphrase | Bruno_Mosti_CV_v3.pdf |
| Q05 | Temporal | PSB_KnowledgeGraph.docx |
| Q08 | Implicit | PSB_Phase8_Gemini.docx |
| Q09 | Implicit | PATCH_INSTRUCTIONS.txt |
| Q10 | Implicit | psb_server_interface_chat-2026-05-11-23-23-26.md |

**Metrics:**

- **Recall@5** — did the system find the right document anywhere in the top 5 results? Score is 1.0 if yes, 0.0 if no, averaged across all 20 queries.
- **MRR (Mean Reciprocal Rank)** — how high up did the right document land? Computed as the average of 1/rank across all queries. Two systems can have identical Recall@5 but different MRR — the one that puts the gold document at position 1 more often wins.
- **p95 latency** — the 95th percentile query time in milliseconds. Each query was timed 3 times and the median was taken to eliminate noise from cold cache or system spikes. Recall and MRR use the retrieved top-5 from the same evaluation run.

---

## 4. Configurations

**BM25 — keyword retrieval**
Tokeniser: `re.findall(r'\w+', text.lower())` — lowercases everything and splits on any non-word character including hyphens. `Flask-Login` becomes `["flask", "login"]` and `Fail-2ban` becomes `["fail", "2ban"]`, so a natural-language query like "flask login" still finds documents that wrote the hyphenated form. Trade-off: compound terms are split apart, which can reduce precision. Ruled out: space-split only (`text.lower().split()`) which would keep hyphens intact and miss those queries.

**Dense — semantic search**
Model: `sentence-transformers/all-MiniLM-L6-v2` — small (~90 MB), runs on CPU, 384-dimensional embeddings. I chose it because it kept p95 well under 1s on my hardware; larger models like `e5-large` (768+ dims) would likely be slower — I did not benchmark e5-large, but MiniLM was sufficient for 208 documents. Corpus embeddings are L2-normalised and cached to `embeddings.npy` (208 × 384); similarity is a dot product at query time.

**Hybrid — Reciprocal Rank Fusion (k=60)**
BM25 returns scores like 12.4; dense returns scores like 0.82 — different scales, not directly comparable. RRF ignores scores entirely and fuses positions: each document scores `1 / (60 + rank)` from each method, contributions are summed. The top 20 candidates from each method are pooled before fusion; the final result is the top 5 from the merged ranking. k=60 prevents the top-ranked document from dominating too aggressively. Ruled out: weighted score blending — any weighting would be arbitrary given the scale mismatch.

**Bonus — Reranker (not built)**
A cross-encoder reranker on Hybrid's top-20 candidates was not implemented within the time available. The planned approach is a two-stage orchestration pipeline combining SPLADE (sparse learned retrieval) and RoBERTa as the cross-encoder reranker, running in parallel and merging results — described further in Section 8.

---

## 5. Results

Recall@5 and MRR were identical across three full runs; p95 latency varied by ~1ms with system load. The table below is from the final run saved to `results/table.json`.

| Config | Recall@5 | MRR | p95 (ms) |
|---|---|---|---|
| BM25 | 0.70 | 0.5142 | 1.12 |
| Dense | 0.80 | 0.6417 | 4.64 |
| Hybrid_RRF | **0.85** | **0.7142** | **6.12** |

Hybrid_RRF leads on both quality metrics — +0.15 Recall@5 over BM25 and +0.05 over Dense, with MRR improving at each step. BM25 is the fastest at 1.21ms but the quality gap is significant. All three configs remain well under the 1000ms p95 constraint, making latency a non-issue for the winner decision.

---

## 6. Best Configuration

Hybrid_RRF is the best configuration: it achieves the highest Recall@5 (0.85) and MRR (0.7142) across all 20 queries while keeping p95 latency at 6.12ms — well within the 1000ms constraint.

---

## 7. Where the Best Config Still Loses

The hardest queries in this experiment are implicit ones — where I remember a topic but not the filename or the exact words in the document. Q08 is the clearest example: I asked "Which doc used to talk about some security and UI improvement I totally forgot?" and the gold document is `PSB_Phase8_Gemini.docx`. Hybrid_RRF returned `COMP_1765_-_Research_Task.docx`, `LEC_AI_Assessment1_Learning_Doc.md`, `Revision_Doc.docx`, `SECURITY.md`, and `Tutorial_Dynamic_Modelling_Solution_notes.pdf` — the correct file was not in the top 5. BM25 matched generic words like "security" in the wrong documents, not the Phase 8 doc. Dense understood the security theme but still pulled the wrong files. RRF combined two weak ranked lists and still missed. This test shows the limit when the query has meaning but not precise keywords, and `combined_text` (summary + excerpt only) does not contain enough context to anchor the right document. A fix would be full-text ingest or a cross-encoder reranker on hybrid's top-20 — both out of scope here.

---

## 8. With Another Week

With more time I would start by implementing full-text ingest so the complete document body is searchable rather than just the summary and excerpt — this would directly address the implicit query failures in Section 7. I would also scale towards a more advanced hybrid: an orchestration strategy combining SPLADE (sparse learned retrieval) and RoBERTa as a cross-encoder reranker, running retrieval methods in parallel and merging results through a two-stage pipeline rather than simple RRF fusion.

---

## 9. Reproduce

```bash
git clone https://github.com/brunomosti/lec-retrieval.git
cd lec-retrieval
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
make run
```

First run downloads the sentence-transformers model (~90 MB). Allow a few minutes on first `make run`.

`make run` runs `python evaluate.py` and writes `results/table.json`. No Neo4j password needed — graders use `corpus.json` and `embeddings.npy` directly from the repo.


---

## AI Usage

I used Cursor and Claude throughout this project as a learning tool and tutor — explaining concepts like BM25 scoring, RRF fusion, MRR, and p95 latency; helping debug code; and structuring README sections from my own answers. This was my first time building a retrieval system and I needed that support to move fast enough to meet the deadline. All retrieval code was implemented by me following a step-by-step guide. The 20 queries were designed by me from reading my own corpus. All results come from running `python evaluate.py` on my machine. The failure paragraph is based on my own Q08 terminal output. I understand the core concepts and can explain my decisions, but I acknowledge I am still consolidating this knowledge — three days was enough to build and ship, not enough to master everything.
