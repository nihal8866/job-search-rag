# Engineering decision log

Record a decision when evidence supports it. Distinguish planned choices from implemented behavior.

## 2026-10-20: Project scope

- Decision: build job retrieval with RAG, including both optional enhancements.
- Reason: the assignment explicitly requests job search, and both hybrid search and reranking were selected for this project.
- Scope: Python, FastAPI, vector retrieval plus BM25, Reciprocal Rank Fusion, a cross-encoder reranker, and an LLM response stage.
- Status: planned; implementation and model choices are pending.

## 2026-10-20: Repository organization

- Decision: separate original data, experiments, reusable code, scripts, tests, and documentation.
- Reason: preserve original inputs and keep experiments separate from the eventual API implementation.
- Documentation: maintain Markdown during development and transfer the final report to Google Docs for submission.

## Pending decisions

- Python version and dependency versions.
- Embedding model and vector database.
- HTML cleaning strategy and chunk boundaries.
- Candidate counts, fusion parameters, and job-level aggregation.
- Reranker model and latency budget.
- LLM provider, model, prompt, and response schema.
- Labeled evaluation queries and acceptance criteria.
