# Project plan

## Working approach

Work in small steps: explain the purpose, implement or run a small piece, inspect the output, and confirm understanding before advancing. Record engineering decisions as they are made.

## Requirements checklist

- [x] Read the assignment and inspect the supplied dataset.
- [x] Create the project structure and documentation scaffold.
- [x] Initialize Git and connect the GitHub remote.
- [x] Audit missing values, duplicates, field types, and description lengths.
- [x] Clean HTML while preserving original descriptions and job metadata.
- [x] Export cleaned jobs to JSONL and verify reloaded values after normalizing missing values for comparison.
- [x] Chunk descriptions while preserving job metadata.
- [x] Generate embeddings and validate numerical values and reload consistency.
- [ ] Save and verify the final chunking notebook with current cells and outputs in execution order.
- [x] Store embeddings in a persistent cosine-distance Chroma collection.
- [x] Demonstrate top-N vector retrieval in a notebook.
- [x] Implement BM25 keyword retrieval and rank fusion in a notebook.
- [x] Add a multilingual cross-encoder reranker in a notebook.
- [x] Present distinct jobs using the best retrieved chunk per job.
- [ ] Verify notebooks 03 and 04 from fresh kernels after separating their cells.
- [ ] Move the notebook pipeline into reusable Python modules with configurable retrieval settings.
- [ ] Implement explicit location-filter handling.
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
