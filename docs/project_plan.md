# Completion Checklist

## Implemented

- [x] Repository structure and GitHub connection.
- [x] Data audit: missing values, duplicates, types, coverage, and dates.
- [x] HTML cleaning with original-data preservation and JSONL export.
- [x] Multilingual token analysis and overlapping chunks with source metadata.
- [x] GPU embeddings with shape, finite-value, and normalization checks.
- [x] Reload verification of values, missingness, and row alignment.
- [x] Persistent Chroma storage and vector-search baseline.
- [x] BM25 retrieval and Reciprocal Rank Fusion.
- [x] Multilingual cross-encoder reranking with pair-length checks.
- [x] Distinct-job presentation and exploratory evidence inspection.

## Review and reproducibility

- [x] Separate notebook responsibilities and remove temporary diagnostic cells.
- [x] Organize direct dependencies, notebook tools, and the environment snapshot.
- [x] Execute all four revised notebooks from fresh kernels.
- [ ] Verify installation in a newly created environment.

## Remaining implementation

- [ ] Move notebook logic into reusable Python modules.
- [ ] Add configurable retrieval settings and explicit location handling.
- [ ] Integrate an LLM with grounded responses and source references.
- [ ] Implement FastAPI POST /api/query with validation and error handling.
- [ ] Evaluate vector-only, hybrid, and reranked retrieval on labeled queries.
- [ ] Check latency, grounding, no-match behavior, and pipeline/API failures.
- [ ] Add verified setup and API request/response examples.
- [ ] Complete the report, limitations, and future enhancements.
- [ ] Prepare the Google Docs report and verify reviewer access.
- [ ] Verify the complete codebase and submit the GitHub repository link.

## Scope

The assignment's title and requirements specify job-data RAG. An isolated sentence
about diabetes classification appears unrelated and is treated as a copy-paste error.

## Verification record - October 28, 2026

All four revised notebooks completed in separate fresh kernels using the existing
project environment and cached models. The run regenerated cleaned data, chunks,
and GPU embeddings; verified artifact reloads and Chroma storage; and completed
BM25, fusion, and reranking. Chroma contained 3,618 records, the embedding array
was (3618, 384), and the example query produced 34 fused candidates.

`pip check` passed. Direct dependency pins match the installed environment, and
the requirements files passed an installed-environment dry run. Notebook schemas,
Python syntax, and documentation file links were checked. A fresh installation
into a new environment and broader relevance evaluation remain pending.
