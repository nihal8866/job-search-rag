# Project plan

## Working approach

Work in small steps: explain the purpose, implement or run a small piece, inspect the output, and confirm understanding before advancing. Record engineering decisions as they are made.

## Requirements checklist

- [x] Read the assignment and inspect the supplied dataset.
- [x] Create the project structure and documentation scaffold.
- [x] Initialize Git and connect the GitHub remote.
- [ ] Audit missing values, duplicates, field types, and description lengths.
- [ ] Clean HTML and chunk descriptions while preserving job metadata.
- [ ] Generate embeddings.
- [ ] Store embeddings in a vector database.
- [ ] Implement configurable top-N vector retrieval.
- [ ] Implement BM25 keyword retrieval and rank fusion (hybrid search).
- [ ] Add a cross-encoder reranker.
- [ ] Consolidate chunks into distinct job listings.
- [ ] Integrate an LLM with a grounded, concise prompt.
- [ ] Implement FastAPI `POST /api/query`.
- [ ] Test retrieval quality, grounding, and endpoint behavior.
- [ ] Compare vector-only, hybrid, and reranked retrieval.
- [ ] Document reproducible setup and real request/response examples.
- [ ] Finish architecture, decisions, assumptions, limitations, and future work.
- [ ] Transfer the final report to Google Docs.
- [ ] Run final verification and confirm reviewer access to GitHub.

## Milestones

1. Repository and documentation setup.
2. Data inspection and preprocessing.
3. Vector retrieval baseline.
4. Hybrid retrieval and reranking.
5. LLM integration and API.
6. Evaluation, report, and submission.

## Scope note

The assignment includes an isolated sentence about diabetes classification in its dataset description. Its title and implementation requirements specify job retrieval with RAG. Treat the diabetes sentence as an apparent copy-paste error.
